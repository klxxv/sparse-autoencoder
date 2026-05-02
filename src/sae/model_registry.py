"""Multi-model SAE loading for Llama 3, Qwen 3, DeepSeek V4, Gemma 4.

Provides unified interface to load SAEs from different model families
using SAELens and TransformerLens.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

MODEL_REGISTRY = {
    "llama3": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "sae_release": "llama-scope-8b",
        "d_model": 4096,
        "n_layers": 32,
        "description": "Llama 3 8B — LlamaScope SAE (32k/131k width)",
    },
    "qwen3": {
        "model_name": "Qwen/Qwen3-8B-Base",
        "sae_release": "qwen-scope-8b",
        "d_model": 4096,
        "n_layers": 32,
        "description": "Qwen 3 8B — QwenScope SAE (32k-128k width)",
    },
    "deepseek-v4": {
        "model_name": "deepseek-ai/DeepSeek-V4-Base",  # placeholder
        "sae_release": "deepseek-r1-sae",  # TODO: update for V4
        "d_model": 7168,
        "n_layers": 61,
        "description": "DeepSeek V4 — DeepSeek SAE",
    },
    "gemma4": {
        "model_name": "google/gemma-4-9b",  # placeholder
        "sae_release": "gemma-scope-2-9b-pt-res",
        "d_model": 3584,
        "n_layers": 42,
        "description": "Gemma 4 — Gemma Scope 2 SAE",
    },
}

SUPPORTED_HOOKS = [
    "hook_resid_pre",
    "hook_resid_post",
    "hook_mlp_out",
    "hook_attn_out",
]


@dataclass
class ModelSAEConfig:
    """Configuration for a model + SAE pair."""
    model_key: str
    model_name: str
    sae_release: str
    layer: int = 12
    hook_name: str = "hook_resid_post"
    d_model: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def get_hook_point(self) -> str:
        return f"blocks.{self.layer}.{self.hook_name}"

    @classmethod
    def from_model_key(cls, model_key: str, layer: int = 12, **kwargs) -> "ModelSAEConfig":
        info = MODEL_REGISTRY[model_key]
        return cls(
            model_key=model_key,
            model_name=info["model_name"],
            sae_release=info["sae_release"],
            layer=layer,
            d_model=info["d_model"],
            **kwargs,
        )


def load_model_and_sae(config: ModelSAEConfig):
    """Load TransformerLens model and SAELens SAE for a given config.

    Returns:
        (model, sae, cfg_dict, sparsity)
    """
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Load model
    model = HookedTransformer.from_pretrained(
        config.model_name,
        device=config.device,
    )

    # Load SAE
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=config.sae_release,
        sae_id=config.get_hook_point(),
        device=config.device,
    )

    return model, sae, cfg_dict, sparsity


def list_available_models() -> Dict[str, str]:
    """List all available models with descriptions."""
    return {k: v["description"] for k, v in MODEL_REGISTRY.items()}
