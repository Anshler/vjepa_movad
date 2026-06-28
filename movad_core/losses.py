import torch


def build_loss(cfg):
    loss_type = cfg.get('loss_type', 'CrossEntropyLoss')
    if loss_type != 'CrossEntropyLoss':
        raise Exception('Not supported loss {}'.format(loss_type))
    smoothing = cfg.get('smoothing', 0.0)
    weight = torch.tensor(cfg.class_weights, dtype=torch.float32).to(cfg.device)
    return WeightAgnosticCELoss(weight=weight, label_smoothing=smoothing)


class WeightAgnosticCELoss(torch.nn.CrossEntropyLoss):
    """CrossEntropyLoss that casts weight to match input dtype (needed for AMP)."""

    def forward(self, input, target):
        if self.weight is not None and self.weight.dtype != input.dtype:
            self.weight = self.weight.to(input.dtype)
        return super().forward(input, target)
