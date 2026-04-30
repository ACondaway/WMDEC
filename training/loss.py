import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Combined diffusion loss with optional Qwen visual semantic consistency loss."""

    def __init__(self, lambda_sem: float = 0.1):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.mse = nn.MSELoss()

    def diffusion_loss(self, eps_pred: torch.Tensor, eps_target: torch.Tensor) -> torch.Tensor:
        return self.mse(eps_pred, eps_target)

    def semantic_loss(
        self,
        patch_pred: torch.Tensor,
        patch_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine-distance semantic consistency loss in Qwen patch embedding space.
        Mean-pools patches to a single vector per image before computing cosine similarity.

        Args:
            patch_pred:   (B, N_patches, D) Qwen embeddings of reconstructed images
            patch_target: (B, N_patches, D) Qwen embeddings of original images
        """
        z_pred = patch_pred.mean(dim=1)    # (B, D)
        z_target = patch_target.mean(dim=1)
        cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1)
        return (1.0 - cos_sim).mean()

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_target: torch.Tensor,
        patch_pred: torch.Tensor = None,
        patch_target: torch.Tensor = None,
    ) -> dict:
        l_diff = self.diffusion_loss(eps_pred, eps_target)
        result = {"diffusion": l_diff, "total": l_diff}

        if patch_pred is not None and patch_target is not None and self.lambda_sem > 0:
            l_sem = self.semantic_loss(patch_pred, patch_target)
            result["semantic"] = l_sem
            result["total"] = l_diff + self.lambda_sem * l_sem

        return result
