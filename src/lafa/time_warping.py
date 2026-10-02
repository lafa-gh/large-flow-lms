"""EMA loss histogram used to sample and schedule flow time.

Bins cover reverse flow time (1 is noise, 0 is data). Each bin tracks the
observed loss with an exponential moving average. The normalized bin losses
form a sampling density; one histogram per block position allows all positions
in a block to share a quantile while following their own time schedules.
"""

import numpy as np
import torch


class TimeWarping:
    """Sample denoising times from EMA loss histograms."""

    def __init__(
        self,
        num_bins=100,
        t_min=0.0,
        t_max=1.0,
        device="cuda",
        per_position=False,
        bd_size=8,
    ):
        if num_bins < 1 or bd_size < 1 or t_min >= t_max:
            raise ValueError("Expected positive bin/block counts and t_min < t_max")
        self.num_bins = num_bins
        self.t_min = t_min
        self.t_max = t_max
        self.device = device
        self.per_position = per_position
        self.bd_size = bd_size
        self.ema_decay = 0.99
        shape = (bd_size, num_bins) if per_position else (num_bins,)
        self.loss_ema = torch.ones(shape, device=device, dtype=torch.float32)

    def to(self, device):
        self.device = device
        self.loss_ema = self.loss_ema.to(device)
        return self

    def state_dict(self):
        return {"dist": "ema", "loss_ema": self.loss_ema.detach().cpu()}

    def load_state_dict(self, state_dict):
        """Load an EMA warper, including checkpoints with obsolete optimizer keys.

        A shared eval grid can pool a per-position training checkpoint. A
        per-position warper can also start from a shared EMA histogram.
        """
        if state_dict.get("dist", "ema") != "ema":
            raise ValueError("Only EMA time warper checkpoints are supported")
        if "loss_ema" not in state_dict:
            raise ValueError("EMA time warper checkpoint is missing loss_ema")
        loss_ema = (
            state_dict["loss_ema"].detach().to(device=self.device, dtype=torch.float32)
        )
        if loss_ema.ndim not in (1, 2) or loss_ema.shape[-1] != self.num_bins:
            raise ValueError(
                f"Expected loss_ema with {self.num_bins} bins, got {tuple(loss_ema.shape)}"
            )
        if not torch.isfinite(loss_ema).all():
            raise ValueError("loss_ema contains non-finite values")
        if self.per_position:
            if loss_ema.ndim == 1:
                loss_ema = loss_ema.repeat(self.bd_size, 1)
            elif loss_ema.shape[0] != self.bd_size:
                raise ValueError(
                    f"Expected {self.bd_size} position histograms, got {loss_ema.shape[0]}"
                )
        elif loss_ema.ndim == 2:
            loss_ema = loss_ema.mean(dim=0)
        self.loss_ema = loss_ema.clone()

    def _ema_edges(self, loss_row):
        """Return uniform time bins and the EMA-based sampling CDF."""
        bin_widths = torch.full(
            (self.num_bins,),
            1.0 / self.num_bins,
            device=self.device,
            dtype=torch.float32,
        )
        bin_edges = torch.cat((bin_widths.new_zeros(1), bin_widths.cumsum(dim=0)))
        bin_edges[-1] = 1.0

        density = loss_row.clamp_min(1e-12)
        bin_mass = density / density.sum()
        cdf = torch.cat((bin_mass.new_zeros(1), bin_mass.cumsum(dim=0)))
        cdf[-1] = 1.0
        return bin_widths, bin_edges, bin_mass, cdf

    def _pos_edges(self, pos):
        if not self.per_position:
            raise ValueError("A shared warper has no per-position density")
        return self._ema_edges(self.loss_ema[pos])

    def _get_bin_widths_and_edges(self):
        loss_row = (
            self.loss_ema.mean(dim=0) if self.loss_ema.ndim == 2 else self.loss_ema
        )
        return self._ema_edges(loss_row)

    def _invert_u_row(self, u, bin_widths, bin_edges, bin_mass, cdf):
        """Map a quantile to denoising time using one EMA histogram."""
        bin_index = torch.clamp(
            torch.searchsorted(cdf, u.contiguous()) - 1, 0, self.num_bins - 1
        )
        fraction = (u - cdf[bin_index]) / bin_mass[bin_index].clamp_min(1e-12)
        reverse_time = bin_edges[bin_index] + fraction * bin_widths[bin_index]
        denoising_time = self.t_max - reverse_time * (self.t_max - self.t_min)
        return denoising_time.clamp(self.t_min, self.t_max)

    def _forward_u_row(self, t, bin_widths, bin_edges, bin_mass, cdf):
        """Map denoising time to its quantile in one EMA histogram."""
        reverse_time = (self.t_max - t.clamp(self.t_min, self.t_max)) / (
            self.t_max - self.t_min
        )
        bin_index = torch.clamp(
            torch.searchsorted(bin_edges, reverse_time.contiguous()) - 1,
            0,
            self.num_bins - 1,
        )
        fraction = (reverse_time - bin_edges[bin_index]) / bin_widths[bin_index]
        return (cdf[bin_index] + fraction * bin_mass[bin_index]).clamp(0.0, 1.0)

    def coupled_fit_times(self, t, anchor):
        """Give each block one uniform-time anchor and a shared quantile.

        The anchor position stays at its uniform input time, so every position
        receives full time coverage as anchors rotate over blocks.
        """
        if not self.per_position:
            raise ValueError("Coupled times require per-position histograms")
        output_device = t.device
        t = t.to(self.loss_ema.device)
        anchor = anchor.to(self.loss_ema.device)
        position_edges = [self._pos_edges(i) for i in range(self.bd_size)]
        tau = torch.empty_like(t)
        for i in range(self.bd_size):
            mask = anchor == i
            if mask.any():
                tau[mask] = self._forward_u_row(t[mask], *position_edges[i])
        times_by_position = [self._invert_u_row(tau, *edge) for edge in position_edges]
        return torch.stack(times_by_position, dim=1).to(output_device)

    def sample_tau_coupled(self, tau, device=None):
        """Map one quantile per block to a denoising time per position."""
        if not self.per_position:
            raise ValueError("Coupled sampling requires per-position histograms")
        output_device = self.loss_ema.device if device is None else device
        tau = tau.to(self.loss_ema.device)
        position_edges = [self._pos_edges(i) for i in range(self.bd_size)]
        times = torch.stack(
            [self._invert_u_row(tau, *edge) for edge in position_edges], dim=1
        )
        return times.to(output_device)

    def _euler_grid(self, quantiles, edges):
        grid = self._invert_u_row(quantiles, *edges).sort().values
        grid[0], grid[-1] = self.t_min, self.t_max
        return grid

    def grid_per_position(self, num_steps, device=None):
        """Return ascending Euler time grids with pinned endpoints, one per position."""
        if not self.per_position:
            raise ValueError("Per-position grids require per-position histograms")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        output_device = self.loss_ema.device if device is None else device
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=self.loss_ema.device)
        grids = []
        for i in range(self.bd_size):
            grids.append(self._euler_grid(u, self._pos_edges(i)))
        return torch.stack(grids, dim=0).to(output_device)

    def sample(self, batch_size, device=None):
        """Draw denoising times from the pooled EMA loss density."""
        output_device = self.loss_ema.device if device is None else device
        u = torch.rand(batch_size, device=self.loss_ema.device)
        return self._invert_u_row(u, *self._get_bin_widths_and_edges()).to(
            output_device
        )

    def grid(self, num_steps, device=None):
        """Return an ascending pooled Euler time grid with pinned endpoints."""
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        output_device = self.loss_ema.device if device is None else device
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=self.loss_ema.device)
        return self._euler_grid(u, self._get_bin_widths_and_edges()).to(output_device)

    def _bin_statistics(self, t, loss_val):
        counts = torch.zeros(self.num_bins, device=self.device, dtype=torch.float32)
        sums = torch.zeros_like(counts)
        if t.numel():
            reverse_time = (
                (self.t_max - t.detach().reshape(-1).to(self.device).float())
                / (self.t_max - self.t_min)
            ).clamp(0.0, 1.0)
            bin_index = (
                (reverse_time * self.num_bins).long().clamp_(0, self.num_bins - 1)
            )
            losses = loss_val.detach().reshape(-1).to(self.device, dtype=torch.float32)
            counts.scatter_add_(0, bin_index, torch.ones_like(losses))
            sums.scatter_add_(0, bin_index, losses)
        return counts, sums

    @torch.no_grad()
    def update_warping(self, t, loss_val, sync_ddp=True, pos=None):
        """Update observed bins by EMA, reducing counts and losses across ranks.

        Every rank calls the same collective even for an empty position bucket.
        This keeps the histograms identical and prevents DDP deadlocks.
        """
        if self.per_position != (pos is not None):
            raise ValueError("Position must be set exactly for per-position warpers")
        if t.numel() != loss_val.numel():
            raise ValueError("t and loss_val must contain the same number of samples")
        row = self.loss_ema[pos] if self.per_position else self.loss_ema
        counts, sums = self._bin_statistics(t, loss_val)
        if sync_ddp:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                packed = torch.stack((counts, sums))
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                counts, sums = packed[0], packed[1]
        seen = counts > 0
        if seen.any():
            mean_loss = sums[seen] / counts[seen]
            row[seen] = self.ema_decay * row[seen] + (1.0 - self.ema_decay) * mean_loss

    def get_per_position_grid_plot(self, num_steps=32, current_step=None):
        """Show the Euler time schedule for each block position."""
        import io
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import PIL.Image

        grid = self.grid_per_position(num_steps).detach().cpu().numpy()
        steps = np.arange(grid.shape[1])
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("viridis")
        for i in range(self.bd_size):
            ax.plot(
                steps,
                grid[i],
                marker="o",
                markersize=3,
                color=cmap(i / max(1, self.bd_size - 1)),
                linewidth=1.6,
                alpha=0.85,
                label=f"pos {i}",
            )
        ax.set(
            xlabel=f"Euler step (0 to {num_steps})",
            ylabel="Flow time t (0:noise, 1:data)",
            ylim=(self.t_min, self.t_max),
            title=f"Per-position EMA generation grid (step {current_step})",
        )
        ax.legend(loc="upper left", ncol=self.bd_size, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return PIL.Image.open(buf)
