# CLIP-Gen-CZSL —— 冻结 CLIP 特征空间生成式增强的组合零样本学习（研究原型）

> **定位声明**：本仓库为**完全自写的研究原型（research prototype）**，不含任何
> 上游代码（CLIP 权重加载与数据集读取通过本地路径依赖外部 CLIP / CAMS 代码，
> 见"复现步骤"）。仓库未附实验结果文件，实测数字以实验记录为准。

## 1. 研究动机

CLIP-based CZSL 方法通常以文本嵌入作为 unseen 组合的唯一表示，缺少该组合的
视觉特征监督。本项目的思路是：在**冻结的 CLIP 特征空间**中训练一个条件生成
模型，为每个未见属性–物体组合合成"伪视觉特征"，再用
"真实已见特征 + 合成未见特征"训练组合分类器，并在推理时与 CLIP 判别式打分
融合。整体为两阶段流程：

- **Stage A（生成器训练）**：以 FiLM 组合条件为 conditioning，在 CLIP 图像特征
  流形上训练 cVAE-GAN；
- **Stage B（分类器训练）**：对每个 unseen 组合采样 k 个合成特征，与真实 seen
  特征一起训练线性组合分类器；
- **融合评估**：CAMS 风格的判别式打分（属性×对象×组合）与生成式分类器打分
  逐样本归一化后加权融合，权重在验证集上扫描。

## 2. 方法模块

| 模块 | 文件 | 说明 |
|---|---|---|
| 特征抽取 | `extract_features.py` | 离线抽取 CLIP ViT-L/14 图像特征（L2 归一化）与属性/对象/组合文本特征库，缓存到 `clip_feats/<dataset>/` |
| 残差适配器 | `models/adapter.py` `FeatureAdapter` | 冻结 CLIP 特征上的轻量残差 MLP，`ratio` 控制 CLIP 语义保留程度 |
| 组合条件调制 | `models/modulator.py` `CompositionModulator` | FiLM 风格：属性嵌入生成 (γ, β) 调制对象嵌入，得到上下文相关的组合条件（而非朴素拼接） |
| 条件 VAE | `models/vae_gan.py` `Encoder` | (真实特征, 条件) → (μ, logvar)，logvar 截断防 KL 发散 |
| 条件生成器 | `models/vae_gan.py` `Generator` | (z ~ N(0,I), 条件) → 合成特征，输出 L2 归一化以与 CLIP 特征同处单位球面 |
| 判别器 | `models/vae_gan.py` `Discriminator` | WGAN-GP 判别器（特征 + 条件 → 标量，n_critic=5，λ_gp=10） |
| 解耦辅助头 | `models/classifier.py` `DisentangleHeads` | 生成器训练时要求合成特征能解码头还原属性/物体标签（解耦一致性约束） |
| 损失函数 | `losses.py` | `L_G = L_gan + λ_r·L_recon + λ_kl·L_kl + λ_clip·L_clip(τ=0.07 InfoNCE) + λ_dis·L_dis + λ_cyc·L_cycle(延迟启用) + λ_div·L_div`；`L_D` 为 WGAN-GP |
| 组合分类器 | `models/classifier.py` `CompositionClassifier` | 全组合空间上的线性分类头（Stage B） |
| 评估 | `evaluate.py` / `evaluate_fusion.py` | Seen/Unseen/HM/AUC；支持 open-world（CAMS feasibility 掩码）；融合评估报告 CLIP-only / gen-only / fused 三档 |

## 3. 仓库结构

```
CLIP-Gen-CZSL/
├── README.md
├── LICENSE                    # MIT
├── extract_features.py        # Stage 0：CLIP 特征离线抽取与缓存
├── train_generator.py         # Stage A：cVAE-GAN 生成器训练
├── train_classifier.py        # Stage B：合成 unseen 特征 + 训练组合分类器
├── evaluate.py                # Seen/Unseen/HM/AUC 评估（closed/open-world）
├── evaluate_fusion.py         # CLIP 判别分 × 生成式分类分 融合评估
├── losses.py                  # 全部损失函数
├── utils.py                   # yml 参数加载 / 随机种子 / checkpoint
├── models/
│   ├── adapter.py             # FeatureAdapter（残差适配器）
│   ├── modulator.py           # CompositionModulator（FiLM 组合条件）
│   ├── vae_gan.py             # Encoder / Generator / Discriminator
│   └── classifier.py          # DisentangleHeads / CompositionClassifier
└── configs/
    ├── mit_states.yml         # 每数据集的生成器/分类器/评估超参
    ├── utzappos.yml           #（eval 节含 open-world 开关与 feasibility 阈值）
    └── cgqa.yml
```

## 4. 复现步骤

> 注意：代码按研究原型风格编写，`PROJECT_ROOT`、CLIP 路径、数据路径以本地
> 硬编码路径书写（如 `./lyt/czsl/CLIP_Gen_CZSL`、`./lyt/czsl/CLIP-main`、
> `./lyt/czsl/CAMS-main/CAMS`），复现前需统一改为自己环境中的路径。
> `extract_features.py` / `evaluate*.py` 依赖外部 `clip` 包（CLIP-main）与
> CAMS 的 `dataset.py`（compositional-split-natural 划分）。

### 4.1 环境

- PyTorch + CUDA，`pyyaml`、`tqdm`、`numpy`、`scipy`；
- OpenAI CLIP `ViT-L-14.pt` 权重（`extract_features.py` 中指定路径）。

### 4.2 数据集

MIT-States、UT-Zappos、C-GQA（compositional-split-natural 划分，与 CAMS/Troika
一致），整理为 `data/<dataset>/`。

### 4.3 三步流程

```bash
# Stage 0：抽取并缓存 CLIP 特征（每数据集一次）
python extract_features.py --dataset ut-zappos

# Stage A：训练 cVAE-GAN 生成器
python train_generator.py --config configs/utzappos.yml --dataset ut-zappos

# Stage B：合成 unseen 特征并训练组合分类器
python train_classifier.py --config configs/utzappos.yml --dataset ut-zappos \
    --k_syn 300

# 评估：融合打分（报告 CLIP-only / gen-only / fused）
python evaluate_fusion.py --config configs/utzappos.yml --dataset ut-zappos
# 仅生成式分类器（closed/open-world、feasibility 掩码）
python evaluate.py --config configs/utzappos.yml --dataset ut-zappos
```

主要超参（默认值 / yml）：`d_c=512`、`d_z=512`、`adapter_ratio=0.0~0.2`、
`epochs=100`、`n_critic=5`、`lambda_gp=10`、`k_syn=300`、`epochs_clf=30`。
open-world 的 feasibility 阈值按数据集写在 `configs/*.yml` 的 `eval.threshold`。

## 5. 已知限制（如实说明）

- 研究原型代码：多处本地路径硬编码，未做打包级抽象；`train_classifier.py` 的
  open-world 候选扩展在未传入 feasibility 张量与阈值时按闭世界 unseen 集处理
  （该分支为 open-world 筛选预留，open-world 评估主要依赖 `evaluate.py` 的
  CAMS feasibility 文件）；
- 仓库未附特征缓存、checkpoint 与实验结果文件，请勿据此引用任何数字。

## 6. License

本项目以 [MIT](LICENSE) 协议发布：`Copyright (c) 2025 Yyitan`。
