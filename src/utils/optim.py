"""Native Muon and AdamW exposed as one Trainer-compatible optimizer."""

import torch


class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        matrix_groups, auxiliary_groups = [], []
        for group in param_groups:
            group = dict(group)
            use_muon = group.pop("use_muon")
            (matrix_groups if use_muon else auxiliary_groups).append(group)
        self.muon = torch.optim.Muon(
            matrix_groups,
            adjust_lr_fn="match_rms_adamw",
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
        )
        self.adamw = torch.optim.AdamW(auxiliary_groups) if auxiliary_groups else None
        groups = self.muon.param_groups + (self.adamw.param_groups if self.adamw else [])
        super().__init__(groups, {})
        self._matrix_group_count = len(matrix_groups)
        self.muon.state = self.state
        if self.adamw:
            self.adamw.state = self.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        if self.adamw:
            self.adamw.step()
        return loss

    def state_dict(self):
        state = super().state_dict()
        state["muon_backend"] = "torch_match_rms_adamw"
        return state

    def load_state_dict(self, state_dict):
        if state_dict.get("muon_backend") != "torch_match_rms_adamw":
            raise ValueError(
                "Optimizer checkpoint is not RMS-matched native Muon; "
                "load model weights as initialization instead of resuming optimizer state."
            )
        super().load_state_dict(state_dict)
        self.muon.param_groups = self.param_groups[: self._matrix_group_count]
        self.muon.state = self.state
        if self.adamw:
            self.adamw.param_groups = self.param_groups[self._matrix_group_count :]
            self.adamw.state = self.state
