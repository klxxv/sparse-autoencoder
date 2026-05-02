"""Sparse Autoencoder (SAE) — PyTorch Lightning implementation.

References:
    - Cunningham et al. (ICLR 2024): Sparse Autoencoders Find Highly Interpretable Features
    - Anthropic: Towards Monosemanticity (2023), Scaling Monosemanticity (2024)
    - SAELens: https://github.com/jbloomAus/SAELens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class SparseAutoencoder(nn.Module):
    """Standard SAE with ReLU + L1 sparsity penalty.

    Architecture:
        encoder:  x → ReLU(W_enc @ (x - b_pre) + b_enc)
        decoder:  f → W_dec @ f + b_dec  (with tied decoder normalization)
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        tied_decoder: bool = True,
        activation_fn: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        # Decoder
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        if tied_decoder:
            # W_dec = W_enc^T normalized
            self.W_dec.requires_grad = False

        self._init_weights()
        self._activation_fn = activation_fn

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, a=0.0)
        nn.init.kaiming_uniform_(self.W_dec, a=0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features."""
        x_centered = x - self.b_pre
        pre_acts = x_centered @ self.W_enc + self.b_enc

        if self._activation_fn == "relu":
            return F.relu(pre_acts)
        elif self._activation_fn == "topk":
            return pre_acts  # handled in forward
        raise ValueError(f"Unknown activation: {self._activation_fn}")

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to reconstruction."""
        if not self.W_dec.requires_grad:
            # Tied: W_dec = W_enc.normalized()
            W_dec = self.W_enc.t() / self.W_enc.t().norm(dim=-1, keepdim=True)
        else:
            W_dec = self.W_dec
        return features @ W_dec + self.b_dec

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """Forward pass with feature sparsity.

        Returns:
            (reconstruction, feature_acts) if return_features
            else reconstruction
        """
        features = self.encode(x)
        recon = self.decode(features)

        if return_features:
            return recon, features
        return recon


class LitSAE(L.LightningModule):
    """Lightning wrapper for SAE training."""

    def __init__(
        self,
        d_model: int = 768,
        d_sae: int = 768 * 8,
        l1_coefficient: float = 8e-5,
        lr: float = 4e-4,
        l1_warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sae = SparseAutoencoder(
            d_model=d_model,
            d_sae=d_sae,
            tied_decoder=True,
        )
        self.l1_coefficient = l1_coefficient
        self.lr = lr
        self.l1_warmup_steps = l1_warmup_steps

    def forward(self, x):
        return self.sae(x)

    def training_step(self, batch, batch_idx):
        activations = batch  # pre-computed model activations [batch, d_model]

        recon, features = self.sae(activations, return_features=True)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, activations)

        # Sparsity loss (L1) with warmup
        current_l1 = self.l1_coefficient * min(
            1.0, self.global_step / max(1, self.l1_warmup_steps)
        )
        sparsity_loss = current_l1 * features.norm(p=1, dim=-1).mean()

        loss = recon_loss + sparsity_loss

        # Metrics
        l0 = (features > 0).float().sum(dim=-1).mean()
        dead_ratio = (features.sum(dim=0) == 0).float().mean()

        self.log_dict({
            "train/loss": loss,
            "train/recon_loss": recon_loss,
            "train/sparsity_loss": sparsity_loss,
            "train/l0": l0,
            "train/dead_features": dead_ratio,
        }, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.sae.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100_000, eta_min=self.lr / 10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    @classmethod
    def from_pretrained(cls, release: str, sae_id: str, device: str = "cpu"):
        """Load a pre-trained SAE from SAELens."""
        from sae_lens import SAE as SAELensSAE

        sae_lens, cfg_dict, _ = SAELensSAE.from_pretrained(
            release=release, sae_id=sae_id, device=device
        )
        # Convert SAELens SAE to our LitSAE format
        d_model = cfg_dict["d_in"]
        d_sae = cfg_dict["d_sae"]

        model = cls(d_model=d_model, d_sae=d_sae)
        model.sae.W_enc.data = sae_lens.W_enc.data.clone()
        model.sae.b_enc.data = sae_lens.b_enc.data.clone()
        model.sae.b_pre.data = sae_lens.b_pre.data.clone() if hasattr(sae_lens, "b_pre") else 0.0
        model.sae.W_dec.data = sae_lens.W_dec.data.clone()
        model.sae.b_dec.data = sae_lens.b_dec.data.clone()

        return model
