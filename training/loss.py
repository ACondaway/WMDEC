import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Combined diffusion loss with optional semantic consistency loss."""

    def __init__(self, lambda_sem: float = 0.1):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.mse = nn.MSELoss()

    def diffusion_loss(self, eps_pred: torch.Tensor, eps_target: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and target noise."""
        return self.mse(eps_pred, eps_target)

    def semantic_loss(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """
        Cosine-distance semantic consistency loss.

        Args:
            z_pred: (B, D) SigLIP embedding of reconstructed image
            z_target: (B, D) SigLIP embedding of original image
        """
        cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1)
        return (1.0 - cos_sim).mean()

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        z_pred: torch.Tensor = None,
        z_target: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute total loss.

        Returns:
            dict with 'total', 'diffusion', and optionally 'semantic' losses
        """
        l_diff = self.diffusion_loss(eps_pred, eps_target)
        result = {"diffusion": l_diff, "total": l_diff}

        if z_pred is not None and z_target is not None and self.lambda_sem > 0:
            l_sem = self.semantic_loss(z_pred, z_target)
            result["semantic"] = l_sem
            result["total"] = l_diff + self.lambda_sem * l_sem

        return result
