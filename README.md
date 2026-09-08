# AIC 2026 无人机低空航拍图像语义分割

本仓库覆盖初赛和复赛阶段的完整工作流：官方压缩包解压、数据体检、类别与外观分布分析、固定验证划分、单模型训练、验证、断点续训、测试集推理、提交格式校验和 ZIP 打包。训练需要由参赛者本人执行。

## 1. 官方任务与硬约束

- 训练集：6996 张图像及对应标签。
- 测试集：测试集 1 为 500 张，测试集 2 为 1300 张；测试集 3 在半决赛发布。
- 图像与标签均为 `1024x1024` patch。
- 标签 ID：`0-8`，其中 `0` 为 Ignore，不参与 mIoU。
- 指标：类别 `1-8` 的平均交并比 mIoU。
- 仅可使用官方数据；禁止额外训练数据。
- 预训练权重必须来自公开学术模型。
- 最终结果只能来自一个模型；本方案不使用模型集成、权重平均、预测平均或测试时增强。
- 提交标签必须与测试图片同名、同尺寸，是像素值为 `0-8` 的单通道 `L` 模式 PNG；不能是 RGB 或 Palette PNG。

类别定义：

| ID | 类别 | ID | 类别 |
|---:|---|---:|---|
| 0 | Ignore | 5 | Barren / 荒地 |
| 1 | Background / 背景 | 6 | Vegetation / 森林与植被 |
| 2 | Building / 建筑 | 7 | Agricultural / 农田 |
| 3 | Road / 道路 | 8 | Vehicle / 车辆 |
| 4 | Water / 水体 |  |  |

## 2. 模型方案

当前官方测试集1最高分模型为单个 **DINOv3 ViT-S+/16 Boundary Segmenter**：B00
官方mIoU=0.673862，高于A00的0.669362。B01计划在相同DINOv3结构下换回A00原始
划分，仍只使用官方数据。

### 2.1 网络结构

1. 编码器采用公开学术权重 `nvidia/mit-b2`，通过 Hugging Face Transformers 加载。
   MiT-B2 四阶段通道数为 `64/128/320/512`，Transformer block 数为 `3/4/6/3`，注意力头数为 `1/2/5/8`。
2. MiT-B2 输出四个尺度的特征，每层使用 `1x1 Conv + GroupNorm + GELU` 投影到 192 通道。
3. 所有特征上采样至 `1/4` 分辨率后拼接并融合。
4. 使用深度可分离卷积细化局部纹理，减少额外参数量。
5. 从最高分辨率特征预测边界，并以边界概率门控融合特征，改善建筑、道路、车辆和水岸边缘。
6. 最终输出 9 类 logits；边界分支仅作为同一模型内部的训练辅助任务，提交仍只有一个分割模型和一张类别图。

模型实现见 `src/uavseg/model.py`。A00检查点核验：MiT-B2 编码器24,196,288参数，本地解码器553,258参数，总计24,749,546（约24.75M），全部参与微调。第一轮结果见 `docs/EXPERIMENTS.md`，逐文件复盘见 `docs/A00_REVIEW.md`。

### 2.2 损失函数

```text
L = Weighted Cross Entropy + 0.3 * Soft Dice + 0.1 * Balanced Boundary BCE
```

- 加权交叉熵缓解类别不平衡，类别权重由正式训练集像素统计自动生成。
- Dice Loss 直接优化区域重叠，对小类别更友好。
- Balanced Boundary BCE 对边界正样本动态加权并约束类别交界处；Ignore 与有效类的直接交界不作为边界来源，Ignore 像素不参与边界损失。
- 交叉熵使用 `label_smoothing=0.05`。

### 2.3 数据增强

训练阶段：

- 随机尺度缩放：`0.75-1.50`。
- 随机裁剪：`640x640`。
- 水平翻转概率：`0.5`。
- 垂直翻转概率：`0.5`。
- 随机旋转：`0/90/180/270` 度。
- 亮度、对比度、饱和度抖动：幅度 `0.2`。
- 高斯模糊概率：`0.15`，半径 `0.1-1.2`。
- 类别感知裁剪概率：`0.5`；优先选择至少含 `0.5%` 荒地、农田或车辆像素的 crop，最多尝试 8 次。
- ImageNet 均值和标准差归一化。

### 2.4 B线：DINOv3 ViT-S+/16候选

B00使用公开学术权重`dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`作为单个
编码器，从block 2/5/8/11提取Token特征，恢复为1/16二维特征图，并与一个真实1/4
分辨率的轻量RGB细节支路融合。融合结果继续使用本项目的边界门控局部细化和单个9类
分类头，因此最终推理仍是一个模型，不包含A线模型、集成、权重平均、预测平均或TTA。

B00沿用A03 scene-guard代理划分、train-only类别权重、legacy裁剪、现有增强以及
`CE + 0.3 Dice + 0.1 Boundary BCE`。A04官方成绩表明直接关闭边界损失会从
0.664582降至0.661388，因此首次DINOv3实验不同时改边界损失。

实现位于`src/uavseg/model.py`，训练/恢复、验证诊断和预测脚本均可根据检查点中的
`architecture`字段重建A线或B线模型。完整源码版本、权重哈希、服务器路径、冒烟、
正式训练、验证和提交命令见`docs/B00_DINOV3_GUIDE.md`。B00在A03口径最佳验证
mIoU=0.7863238948（epoch 54），完整重算完全一致；官方测试集1为0.673862。

### 2.5 C线：LoveDA研究隔离路线

C00研究LoveDA预适配后再用官方训练集微调。LoveDA是规则禁止的外部训练数据，因此
C00及派生检查点不得用于AIC提交。训练入口会写入research-only标记，拒绝把这类检查点
当普通比赛模型初始化或预测。完整流程见`docs/C00_LOVEDA_RESEARCH_GUIDE.md`。

验证和测试不使用随机增强；验证使用完整 `1024x1024` 图像。

## 3. RTX 5070 环境

不要继续使用原来的 Python 3.7、PyTorch 1.6.0、CUDA 10.1 环境。RTX 5070 属于 Blackwell 架构，CUDA 12.8 才开始提供 Blackwell 支持；旧 PyTorch/CUDA 二进制不能可靠运行在该显卡上。

推荐新建独立环境，不需要安装 MMCV 或 MMSegmentation：

```powershell
conda create -n aic-uav python=3.11 -y
conda activate aic-uav

python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-train.txt
```

验证环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

最后一项必须输出 `True`。`nvidia-smi` 显示的 CUDA 13.0 是驱动能支持的最高 CUDA 版本，不代表旧环境中的 CUDA 10.1 能支持 RTX 5070。

## 4. 数据准备

官方文件当前位于：

```text
D:\AIC-2026\2026-低空图像语义分割赛道-训练集
├── train.zip
├── test_1.zip
├── Examples.zip
└── Label.txt
```

进入仓库：

```powershell
cd D:\AIC-2026\aicomp-uav-segmentation
```

安全解压并核对数量：

```powershell
python tools/prepare_dataset.py `
  --archives "D:\AIC-2026\2026-低空图像语义分割赛道-训练集" `
  --output data
```

解压后应为：

```text
data/
├── train/
│   ├── images/       # 6996
│   └── masks/        # 6996
├── test1/
│   └── images/       # 500
└── Examples/
    ├── images/       # 11
    └── masks/        # 11
```

数据目录已被 `.gitignore` 排除，严禁提交或传播官方数据。

## 5. 数据体检、分布分析与验证划分

先运行完整性检查：

```powershell
python tools/check_dataset.py `
  --images data/train/images `
  --masks data/train/masks `
  --output outputs/dataset_check
```

再执行完整统计并生成固定划分：

```powershell
python tools/profile_dataset.py `
  --images data/train/images `
  --masks data/train/masks `
  --output outputs/dataset_profile `
  --val-fraction 0.10 `
  --trials 512 `
  --seed 3407
```

该步骤会读取全部标签，可能需要一段时间，输出：

- `profile.json`：类别像素比例、图片出现率、RGB 统计和训练权重。
- `split.json`：固定的 90%/10% 训练验证划分。
- `class_weights.json`：训练使用的类别权重。

由于官方已经重新排序 patch 且没有提供原始场景 ID，无法严格按航线或原图分组。工具会尝试 512 个固定种子的候选划分，并选择类别出现、类别面积和图像外观统计最接近全体分布的一组。若赛事方后续提供来源/场景字段，应立即改为按来源分组划分。

## 6. 已观察到的协变量偏移

对压缩包内 256 张训练图和 256 张测试集 1 图像进行等间隔抽样，得到：

| 统计量 | 训练抽样 | 测试集1抽样 |
|---|---:|---:|
| R 均值 | 0.417 | 0.374 |
| G 均值 | 0.423 | 0.387 |
| B 均值 | 0.390 | 0.365 |
| 总体亮度均值 | 0.410 | 0.375 |

测试集 1 整体更暗。另抽查 512 张训练标签，车辆约占像素 `0.75%`、荒地约 `1.70%`；官方统计又显示农田、荒地等在测试集的图片出现率明显高于训练集。

本方案采用以下防御措施：

- 颜色抖动覆盖亮度、对比度和饱和度变化。
- 随机尺度和 90 度旋转适应不同飞行高度与视角。
- 类别权重、Dice 和类别感知裁剪保护稀有类。
- GroupNorm 避免 batch size 1 时 BatchNorm 统计不稳定。
- 无位置编码的 MiT 编码器适应训练 crop 与测试整图的尺寸差异。
- 验证划分同时匹配标签与低阶外观统计，而非只做一次随意随机划分。

不要根据测试集人工伪造标签，也不要使用测试图自训练；测试集只用于最终推理和无标签分布诊断。

## 7. 正式训练

以下命令保留A00历史基线。2026-09-04新增A01/A02受控实验：仅训练集类别权重与可切换的指定类别裁剪，详见 `docs/A01_A02_TRAINING_GUIDE.md`。`--crop-strategy legacy`（默认）保留原裁剪；`targeted`启用候选策略，尚无正式训练增益结论。

先用 11 张示例数据完成一次模型下载、前向、反向、验证和检查点冒烟测试：

```powershell
python scripts/train.py `
  --images data/Examples/images `
  --masks data/Examples/masks `
  --output outputs/s01_b2_boundary_smoke `
  --model nvidia/mit-b2 `
  --epochs 1 `
  --batch-size 1 `
  --gradient-accumulation 1 `
  --crop-size 512 `
  --workers 0
```

冒烟测试只用于发现环境、显存和代码问题，不能用于选择超参数，也不能从它的权重继续正式训练。

如果权重已经下载完成，但 Hugging Face 的联网检查失败，可以在命令末尾添加 `--local-files-only`，直接读取本机缓存。

RTX 5070 8GB 的推荐启动命令：

```powershell
python scripts/train.py `
  --images data/train/images `
  --masks data/train/masks `
  --split outputs/dataset_profile/split.json `
  --class-weights outputs/dataset_profile/class_weights.json `
  --output outputs/a00_b2_boundary `
  --model nvidia/mit-b2 `
  --epochs 60 `
  --batch-size 1 `
  --gradient-accumulation 4 `
  --crop-size 640 `
  --encoder-lr 0.00003 `
  --decoder-lr 0.0001 `
  --weight-decay 0.01 `
  --warmup-ratio 0.05 `
  --power 0.9 `
  --decoder-channels 192 `
  --dice-weight 0.3 `
  --boundary-weight 0.1 `
  --label-smoothing 0.05 `
  --scale-min 0.75 `
  --scale-max 1.5 `
  --color-jitter 0.2 `
  --rare-crop-probability 0.5 `
  --seed 3407 `
  --workers 4
```

关键超参数：

| 参数 | 值 |
|---|---:|
| 训练轮数 | 60 |
| 物理 batch size | 1 |
| 梯度累积 | 4 |
| 有效 batch size | 4 |
| 训练 crop | 640 |
| 验证分辨率 | 1024 |
| 编码器学习率 | 3e-5 |
| 解码器学习率 | 1e-4 |
| 权重衰减 | 0.01 |
| 预热比例 | 5% |
| 学习率策略 | Polynomial，power=0.9 |
| 混合精度 | FP16 |
| 梯度裁剪 | 1.0 |
| 随机种子 | 3407 |

输出目录包含：

- `config.json`：完整启动参数。
- `runtime.json`：PyTorch/CUDA/GPU 信息以及实际模型参数量。
- `metrics.jsonl`：逐轮训练 loss、验证 loss、mIoU、各类别 IoU 和学习率。
- `last.pt`：最近一轮的模型、优化器、调度器和混合精度状态。
- `best.pt`：固定验证集上 mIoU 最好的单个检查点。

如果显存不足，先把 `--crop-size 640` 改为 `512`，不要同时修改其他超参数，并在实验表中记录为独立实验。

### 断点续训

继续原来的总轮数设置，例如从 `last.pt` 恢复并训练到第 60 轮：

```powershell
python scripts/train.py `
  --images data/train/images `
  --masks data/train/masks `
  --split outputs/dataset_profile/split.json `
  --class-weights outputs/dataset_profile/class_weights.json `
  --output outputs/a00_b2_boundary `
  --epochs 60 `
  --batch-size 1 `
  --gradient-accumulation 4 `
  --crop-size 640 `
  --resume outputs/a00_b2_boundary/last.pt
```

恢复训练时应保持数据划分、batch、梯度累积、学习率和总轮数不变。

## 8. 测试集推理

只使用验证 mIoU 最好的 `best.pt`：

```powershell
python scripts/predict.py `
  --images data/test1/images `
  --checkpoint outputs/a00_b2_boundary/best.pt `
  --output outputs/a00_b2_boundary/test1_predictions
```

推理脚本从检查点内的模型配置重建网络，不需要再次下载 MiT-B2 权重。它对每张 `1024x1024` 图片执行一次完整的单尺度前向传播，不进行翻转 TTA 或多次结果平均。

## 9. 校验并生成提交 ZIP

```powershell
python tools/make_submission.py `
  --images data/test1/images `
  --predictions outputs/a00_b2_boundary/test1_predictions `
  --output submissions/a00_b2_boundary_test1.zip
```

打包前会逐文件验证：

- 预测数量和文件名与测试集完全一致。
- 尺寸与对应原图一致。
- PNG 模式严格为单通道 `L`，不是 Palette 或 RGB。
- 像素 ID 全部位于 `0-8`。
- ZIP 内直接存放 PNG，不额外套一层目录。

## 10. 初赛到复赛工作流

1. 跑数据体检与固定划分，保存三份 profile 文件。
2. 使用上述默认配置完成 A00；不要根据 11 张示例数据决定模型优劣。
3. 检查总 mIoU、类别 `1-8` IoU、训练/验证 loss 曲线，特别关注荒地、农田和车辆。
4. 每次实验只改变一个因素，例如 crop 640→512、B2→B1、关闭边界损失或修改类别权重。
5. 初赛测试集 1 只用于提交验证，不以榜单反复手工拟合标签。
6. 复赛测试集 2 发布后，沿用同一个训练集划分和实验规范，重新推理并按相同格式打包。
7. 复赛结束前冻结最终单模型配置、随机种子、依赖版本、训练命令和 Git 提交。
8. 半决赛所需 Docker、完整复现脚本和技术方案 PDF 在进入半决赛后按组委会最新通知补齐；本 README 当前不擅自假定尚未发布的细则。

## 11. 实验纪律

- 正式运行前在 `docs/EXPERIMENTS.md` 登记实验编号。
- 一个实验只验证一个主要改动。
- 记录 Git 提交、数据划分、随机种子、硬件、完整命令、逐类 IoU、耗时和结论。
- 不覆盖旧输出目录，不删除失败实验记录。
- 数据、权重、预测、提交包和账号信息不得提交 Git。

## 12. 自动化测试

```powershell
python -m unittest discover -s tests -v
```

当前测试覆盖数据异常检查、mIoU、Ignore 类、Dice、边界标签以及官方提交格式校验。
