"""Callbacks for flow training state and optional checkpoint snapshots."""

import os

import torch
from transformers import TrainerCallback

from src.utils.distributed import unwrap_model


def _print_main(state, message):
    if state.is_world_process_zero:
        print(message)


def _checkpoint_dir(args, state):
    path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
    os.makedirs(path, exist_ok=True)
    return path


class TimeWarpCallback(TrainerCallback):
    """Track the training step, log time samples and grids, and save the warper."""

    def __init__(self, plot_every: int = 500, hist_every: int = 1000):
        self.plot_every = plot_every
        self.hist_every = hist_every
        self._t_accum = []

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        m = unwrap_model(model)
        if m is not None and hasattr(m, "_global_step"):
            m._global_step = state.global_step

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or not (self.plot_every or self.hist_every):
            return
        m = unwrap_model(model)
        if m is None:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            logs = {}
            wp = getattr(m, "time_warper", None)
            if (
                wp is not None
                and wp.per_position
                and m._warp_active()
                and self.plot_every
                and state.global_step % self.plot_every == 0
            ):
                try:
                    logs["time_warp/per_position_grid"] = wandb.Image(
                        wp.get_per_position_grid_plot(
                            num_steps=32, current_step=state.global_step
                        )
                    )
                except Exception as exc:
                    print(f"[TimeWarp] grid plot skipped: {exc}")
            if self.hist_every:
                st = getattr(m, "_last_sampled_t", None)
                if st is not None and st.numel() > 0:
                    self._t_accum.append(st)
                if state.global_step % self.hist_every == 0 and self._t_accum:
                    try:
                        allt = torch.cat(self._t_accum).numpy()
                        logs["time_warp/sampled_t_hist"] = wandb.Image(
                            self._sampled_t_image(allt, state.global_step)
                        )
                    except Exception as exc:
                        print(f"[TimeWarp] sampled-t histogram skipped: {exc}")
                    finally:
                        self._t_accum = []
            if logs:
                wandb.log(logs, step=state.global_step)
        except Exception as exc:
            print(f"[TimeWarp] wandb log skipped: {exc}")

    def _sampled_t_image(self, t, step):
        """Plot a histogram of sampled flow times."""
        import io
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import PIL.Image

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            t,
            bins=100,
            range=(0, 1),
            density=True,
            color="tab:gray",
            alpha=0.6,
            label=f"empirical sampled t, all positions (n={len(t)})",
        )
        ax.set_xlabel("Flow Time t (0:Noise -> 1:Data)")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_title(f"Sampled-t histogram (step {step})")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return PIL.Image.open(buf)

    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        m = unwrap_model(model)
        wp = getattr(m, "time_warper", None) if m is not None else None
        if wp is not None:
            torch.save(
                wp.state_dict(),
                os.path.join(_checkpoint_dir(args, state), "time_warper.pt"),
            )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        resume_from = getattr(args, "resume_from_checkpoint", None)
        if not resume_from:
            return
        path = os.path.join(resume_from, "time_warper.pt")
        m = unwrap_model(model)
        if m is None or not os.path.exists(path):
            return
        device = next(m.parameters()).device
        m._ensure_time_warper(device)
        m.time_warper.load_state_dict(torch.load(path, map_location=device))
        _print_main(state, f"[TimeWarp] restored warper from {path}")


class EmbedNormProjectCallback(TrainerCallback):
    """Capture embedding norms and restore them after optimizer steps."""

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        m = unwrap_model(model)
        if m is not None and getattr(m, "_embed_norm_project", False):
            m._capture_embed_norms()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        m = unwrap_model(model)
        if m is not None and getattr(m, "_embed_norm_project", False):
            m._post_step_project()


class EMACallback(TrainerCallback):
    """Update model weight EMA and save or restore its checkpoint sidecar."""

    def __init__(self, start_step: int = 0):
        self.start_step = start_step

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        m = unwrap_model(model)
        if m is None or getattr(m, "ema_decay", 0.0) <= 0:
            return
        resume_from = getattr(args, "resume_from_checkpoint", None)
        path = os.path.join(resume_from, "ema.pt") if resume_from else None
        if path and os.path.exists(path):
            m.load_ema_state_dict(torch.load(path, map_location="cpu"))
            _print_main(state, f"[EMA] restored EMA from {path}")
        else:
            m.ema_initialize()
            _print_main(state, f"[EMA] initialized (decay={m.ema_decay})")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        m = unwrap_model(model)
        if (
            m is not None
            and getattr(m, "ema_decay", 0.0) > 0
            and state.global_step >= self.start_step
        ):
            m.ema_update()

    def on_save(self, args, state, control, model=None, **kwargs):
        m = unwrap_model(model)
        if m is None or not getattr(m, "has_ema", False):
            return
        # ZeRO-3 gathers EMA shards here. Every rank must enter this call.
        ema_state = m.ema_state_dict()
        if state.is_world_process_zero and ema_state is not None:
            torch.save(
                ema_state,
                os.path.join(_checkpoint_dir(args, state), "ema.pt"),
            )


class EpochCheckpointCallback(TrainerCallback):
    """Save an epoch model snapshot and its time warper and EMA sidecars."""

    def __init__(self):
        self.trainer = None

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return
        epoch = int(round(state.epoch))
        path = os.path.join(args.output_dir, f"epoch-{epoch}")
        self.trainer.save_model(path)

        m = unwrap_model(self.trainer.model)
        # Like EMACallback.on_save, all ranks must gather their EMA shards.
        ema_state = (
            m.ema_state_dict()
            if m is not None and getattr(m, "has_ema", False)
            else None
        )
        if not state.is_world_process_zero:
            return
        os.makedirs(path, exist_ok=True)
        wp = getattr(m, "time_warper", None) if m is not None else None
        if wp is not None:
            torch.save(wp.state_dict(), os.path.join(path, "time_warper.pt"))
        if ema_state is not None:
            torch.save(ema_state, os.path.join(path, "ema.pt"))
        print(f"[EpochCheckpoint] saved epoch-{epoch} -> {path}")
