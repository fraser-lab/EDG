import torch
import torch.nn.functional as F


def batched_correlation_coefficient(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    batch_size = pred.shape[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)

    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean

    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1)
    )

    return numerator / (denominator + 1e-8)


def batched_density_kl_loss(
    pred: torch.Tensor, target: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    batch_size = pred.shape[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)

    pred_soft = F.softmax(pred_flat / temperature, dim=1)
    target_soft = F.softmax(target_flat / temperature, dim=1)

    kl_per_sample = F.kl_div(pred_soft.log(), target_soft, reduction="none").sum(dim=1)

    return kl_per_sample


def batched_hybrid_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.7
) -> torch.Tensor:
    corr = batched_correlation_coefficient(pred, target)
    corr_loss = 1 - corr

    kl_loss = batched_density_kl_loss(pred, target)

    return alpha * corr_loss + (1 - alpha) * kl_loss


def batched_anticorrelation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = -(pred * target).sum(dim=[-3, -2, -1])
    return loss


def batched_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = ((pred - target) ** 2).sum(dim=[-3, -2, -1])
    return loss
