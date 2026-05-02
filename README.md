# Sparse AutoEncoder (SAE)

> **论文**: [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600) (ICLR 2024)
> **系列**: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) (Anthropic 2023) | [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Anthropic 2024)
> **框架**: PyTorch Lightning + SAELens

## 复现目标

1. 训练 SAE 在 GPT-2 Small 的 residual stream 上
2. 复现关键指标: L0 sparsity、CE loss recovery、dead features ratio
3. 特征可解释性分析（auto-interpretability）
4. 特征 steering 实验

## 核心架构

```
Input Activation → Encoder → Sparse Features → Decoder → Reconstructed Activation
    (d_model)       ↓       (d_sae = 8× d_model)          (d_model)
                 L1 sparsity                     MSE reconstruction
```

**Loss**: `MSE(x, x̂) + λ × ‖f‖₁`

## 复现路线

### Phase 1: 加载预训练 SAE（验证环境）
```bash
cd scripts
uv run python analyze_pretrained_sae.py
```

### Phase 2: 训练自定义 SAE
```bash
uv run python train_sae.py --model gpt2-small --layer 8 --expansion 8
```

### Phase 3: 特征分析
- Feature dashboard / auto-interpretability
- Logit attribution
- Feature steering

## 硬件配置

**当前**: Intel Core Ultra 7 155H (CPU only on WSL)
- Intel Extension for PyTorch (IPEX) 用于 CPU 优化
- BF16/Int8 推理加速

**推荐**: NVIDIA GPU (A100/H100) 用于完整训练

## 目录结构

```
src/sae/
├── __init__.py
├── model.py          # SAE PyTorch implementation
├── training.py       # LightningModule
├── config.py         # Hyperparameter configs
└── analysis.py       # Feature analysis utilities
scripts/
├── train_sae.py
└── analyze_pretrained_sae.py
configs/
└── default.yaml
```
