<div align="center">

# FPU-Kernel

**融合隐私更新算子 — 基于 Triton 加速的 DP-SGD + CKKS 槽位打包融合核**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/triton-2.0%2B-76B900.svg)](https://triton-lang.org/)
[![License: Research](https://img.shields.io/badge/license-research%20only-lightgrey.svg)](#许可证)

</div>

---

## 概述

在隐私保护联邦学习中，每个客户端的模型更新在同态加密前需要经过一系列处理：

$$
\text{展平} \;\rightarrow\; \text{L2 裁剪} \;\rightarrow\; w_i\!\cdot\!\Delta \;\rightarrow\; +\;\mathcal{N}(0,\,\sigma^2) \;\rightarrow\; \text{量化} \;\rightarrow\; \text{CKKS 槽位打包}
$$

朴素实现中，每一步都是独立的 CUDA kernel 启动，导致全局内存被反复读写。
**FPU-Kernel** 将这些操作融合为**单次 Triton kernel 调度**（L2 范数归约除外），
将内存带宽开销降低约 2-3 倍，kernel 启动开销降低约 5 倍。

### 处理流程

```
                         ┌─── 主机端 ──┐
                         │ clip_scale  │
                         │ = min(1,    │
   state_dict ──展平────│  C/‖Δ‖₂)   │
         │               └─────┬──────┘
         ▼                     │
  ┌──────────────┐    ┌───────▼────────────────────────────────┐
  │  Triton      │    │     融合 Triton Kernel（单次启动）      │
  │  L2 范数     │    │                                        │
  │  （两趟      │───▶│  x = Δ × clip_scale × client_weight   │
  │   归约）     │    │  x = x + noise × σ        （若启用 DP）│
  └──────────────┘    │  x = deq(clamp(round(x/s))) （若量化） │
                      │  packed[bundle, slot] = x              │
                      └────────────────────────────────────────┘
                                       │
                                       ▼
                              packed_slots [B, N]
                              → CKKS 加密 → ...
```

## 特性

| 特性 | 说明 |
|------|------|
| **DP-SGD 裁剪** | 可配置阈值的 L2 范数裁剪 |
| **高斯噪声注入** | 校准噪声：$\sigma = \texttt{noise\_multiplier} \times C$ |
| **对称量化** | 可选 int8/int16 量化-反量化，适配 HE 槽位效率 |
| **CKKS 槽位打包** | 基于 LayoutPlan 的 `[bundles, slot_capacity]` 缓冲区填充 |
| **客户端加权** | 面向异构联邦学习的逐客户端重要性权重 |
| **两趟 L2 范数** | 数值稳定的 block 归约 → 树形归约（无 atomic_add） |
| **三种后端** | PyTorch 参考实现 / 多 kernel Triton / 融合单 kernel Triton |

## 安装

```bash
# 前置要求：CUDA GPU、PyTorch >= 2.0
pip install torch

# Triton（Linux）
pip install triton
# Triton（Windows）
pip install triton-windows

# 安装 FPU-Kernel（可编辑模式）
cd fused_private_update
pip install -e .
```

## 快速上手

```python
import torch
from fpu import FusedUpdateConfig, fused_private_update_triton

state_dict = {
    "layer1.weight": torch.randn(64, 32, device="cuda"),
    "layer1.bias":   torch.randn(64, device="cuda"),
    "layer2.weight": torch.randn(10, 64, device="cuda"),
    "layer2.bias":   torch.randn(10, device="cuda"),
}

config = FusedUpdateConfig(
    clip_norm=1.0,
    noise_multiplier=0.1,
    quant_bits=0,            # 0 = 不量化，8 = int8
    client_weight=1.0,
    noise_seed=42,           # 固定种子保证可复现
)

num_params = sum(t.numel() for t in state_dict.values())
slot_capacity = 4096         # CKKS poly_modulus_degree / 2
bundle_count = (num_params + slot_capacity - 1) // slot_capacity

output = fused_private_update_triton(
    state_dict, config, bundle_count, slot_capacity,
    device=torch.device("cuda"),
    measure_time=True,
)

print(f"输出形状:    {output.packed_slots.shape}")   # [bundles, 4096]
print(f"槽位利用率:  {output.utilization:.1%}")
print(f"L2 范数:     {output.metadata['l2_norm']:.4f}")
```

## API 文档

### `FusedUpdateConfig` — 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `clip_norm` | `float` | `1.0` | L2 裁剪阈值 $C$ |
| `noise_multiplier` | `float` | `0.0` | DP 噪声缩放系数（0 = 关闭） |
| `quant_bits` | `int` | `0` | 量化位宽（0 = 关闭，可选 8、16） |
| `client_weight` | `float` | `1.0` | 客户端重要性权重 $w_i$ |
| `ckks_scale` | `float` | $2^{40}$ | CKKS 编码尺度（仅元数据） |
| `noise_seed` | `int\|None` | `None` | 噪声随机种子（`None` = 不固定） |

### 三种后端

```python
from fpu import fused_private_update_ref          # PyTorch 参考实现（CPU/GPU）
from fpu import fused_private_update_triton       # 多 kernel Triton（GPU）
from fpu import fused_private_update_triton_v2    # 融合单 kernel Triton（GPU）
```

| 后端 | kernel 启动次数 | 适用场景 |
|------|----------------|---------|
| `_ref` | 0（纯 PyTorch 算子） | 正确性基线、CPU 回退 |
| `_triton` | 5（norm + clip + noise + quant + pack） | 逐阶段性能分析与调试 |
| `_triton_v2` | 2 个阶段（reduction + 融合；大张量 reduction 会分层启动） | **最大吞吐量，论文报告推荐** |

`_triton_v2` 接受预展平的 tensor，避免额外拷贝：

```python
from fpu.torch_ref import flatten_state_dict

flat = flatten_state_dict(state_dict).cuda()
output = fused_private_update_triton_v2(flat, config, bundle_count, slot_capacity)
```

### `FusedUpdateOutput` — 输出结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `packed_slots` | `Tensor` | `[bundles, slot_capacity]` — 可直接送入 CKKS 加密 |
| `utilization` | `float` | 有效参数占槽位总数的比例 |
| `metadata` | `dict` | 包含 `l2_norm`、`clip_scale`、`noise_std` 及各阶段耗时 |

注意：`quant_bits > 0` 当前语义是先量化再反量化回 float32 CKKS slot buffer，`packed_slots`
的物理存储仍是 float32。benchmark 会同时报告 `output_storage_mb` 和
`logical_quantized_payload_mb`，前者用于真实 GPU I/O 带宽估算，后者用于观察理想量化 payload 大小。

## 验证与基准测试

```bash
# 正确性验证：多配置下 Triton 与 PyTorch 参考实现的数值对比
python -m fpu.validate

# 性能基准：模型规模 scaling + 配置扫描，结果保存到 results/
python run_benchmark.py --mode both --output results/

# 仅测试单个模型
python run_benchmark.py --mode benchmark --model tiny_cnn --output results/

# 复现实验中的大规模 v2 路径，并复用输出/噪声 buffer
python run_benchmark.py --mode benchmark --synthetic-params 3000000000 \
  --backend triton_v2 --clip-norm 1.0 --noise-multiplier 0.0 \
  --quant-bits 8 --reuse-buffers --warmup 10 --iterations 100 \
  --output results/
```

优化后，无噪声量化路径的 `quant_stat_source` 应为 `l2_abs_max`，`quant_stat_ms`
应接近 0；如果打开 DP 噪声，则仍会使用 `transformed_abs_max` 统计路径来保证量化尺度正确。

## 与 hefl_runtime 集成

```python
from hefl.types import CryptoParams
from hefl.layout import HeuristicLayoutPlanner
from fpu import FusedUpdateConfig, fused_private_update_triton

crypto = CryptoParams(slots=4096)
planner = HeuristicLayoutPlanner(crypto)
layout = planner.plan(state_dict, strategy="layer_order")

config = FusedUpdateConfig(clip_norm=1.0, noise_multiplier=0.1)
output = fused_private_update_triton(
    state_dict, config, layout.bundle_count, layout.slot_capacity
)
# output.packed_slots → 送入 CKKS 加密流程
```

## 项目结构

```
fused_private_update/
├── fpu/
│   ├── __init__.py          # 公共 API，Triton 条件导入
│   ├── types.py             # FusedUpdateConfig, FusedUpdateOutput, KernelMetrics
│   ├── torch_ref.py         # PyTorch 参考实现
│   ├── triton_kernel.py     # Triton GPU kernel（多 kernel + 融合版）
│   ├── validate.py          # 正确性验证套件
│   └── benchmark.py         # 性能基准工具
├── configs/
│   └── benchmark.json       # 基准测试预设配置
├── results/                 # 基准测试输出目录
├── setup.py
├── requirements.txt
├── run_benchmark.py         # 基准测试 CLI 入口
├── example_integration.py   # 使用示例
└── README.md
```

## 参考文献

- Abadi et al., [*Deep Learning with Differential Privacy*](https://arxiv.org/abs/1607.00133)（DP-SGD）
- Cheon et al., [*Homomorphic Encryption for Approximate Arithmetic*](https://eprint.iacr.org/2016/421)（CKKS 方案）
- Tillet et al., [*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*](https://triton-lang.org/)

## 许可证

仅限科研用途。
