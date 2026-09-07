# B00 DINOv3 单模型分割路线

状态：代码就绪，尚未正式训练或取得验证/官方成绩。

## 1. 受控实验目的

B00相对A03保留相同的官方训练数据、scene-guard代理划分、train-only类别权重、
legacy裁剪、数据增强、损失函数、60轮计划和随机种子；主要改动是把MiT-B2编码器及
其解码适配替换为DINOv3 ViT-S+/16及Token密集特征适配器。B00不从任何A线检查点
续训，不使用测试标签、伪标签、模型集成、TTA或预测平均。

选择依据来自既有实验：

- A03提供当前更保守的场景隔离代理验证口径。
- A02的targeted裁剪未改善官方测试成绩，B00继续使用legacy裁剪。
- A04把boundary weight从0.1降为0后，官方mIoU从A03的0.664582降至0.661388；
  因此B00保留0.1边界辅助损失，而不是直接删除边界监督。
- D03表明高可信重叠区域整体标签冲突率约0.4204%，边界像素冲突率约为内部的
  10.09倍；B00沿用label smoothing=0.05、Dice和现有3x3膨胀硬边界目标，不人工改标签。
  更软的容忍边界属于后续独立实验，不与首次换骨干混在一起。
- 测试集1描述统计比训练集略暗，继续保留亮度、对比度和饱和度抖动。

## 2. 单模型结构

```text
RGB 640x640 / 1024x1024
├── DINOv3 ViT-S+/16
│   ├── block 2 token feature  ┐
│   ├── block 5 token feature  │ reshape + 1x1 projection
│   ├── block 8 token feature  │ + bilinear upsample to 1/4
│   └── block 11 token feature ┘
└── two-layer RGB detail stem -> true 1/4 feature
        ↓
five-feature concatenation + GroupNorm/GELU fusion
        ↓
depthwise local refinement × (1 + predicted boundary probability)
        ↓
single 9-class segmentation head
```

DINOv3的四个中间层都是1/16 Token网格；RGB细节支路提供真正的1/4局部特征，避免
简单上采样过度损失车辆、道路和建筑边缘。边界头属于同一个端到端模型内部的辅助任务，
最终仍只有一个模型和一张类别预测图。

默认骨干：

- variant：`vits16plus`
- patch：16
- blocks：`2,5,8,11`（从0开始）
- embedding：384
- encoder parameters：28,697,472
- total parameters（decoder channels 192）：30,218,890
- 公开权重：`dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`
- 权重SHA-256：`4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea`
- 官方源码提交：`6876159a11b4df116f30f667f8c9888617df0751`

权重和官方DINOv3源码受Meta DINOv3 License约束，不进入本仓库；使用者需从官方来源
合法取得，并保留许可证。正式采用前仍需取得组委会对LVD-1689M权重的明确合规确认。

## 3. 服务器准备

推荐路径：

```text
/root/autodl-tmp/models/dinov3/                  # Meta官方源码
/root/autodl-tmp/models/dinov3-checkpoints/      # 公开权重
```

克隆并固定官方源码：

```bash
git clone https://github.com/facebookresearch/dinov3.git /root/autodl-tmp/models/dinov3
git -C /root/autodl-tmp/models/dinov3 checkout --detach 6876159a11b4df116f30f667f8c9888617df0751
```

若服务器无法直连GitHub，可只把读取地址替换为已验证可用的镜像；仍必须核验最终提交号。
权重通过SCP上传到`/root/autodl-tmp/models/dinov3-checkpoints/`，不要提交到Git。

在仓库根目录检查源码、权重、哈希和完整前向：

```bash
python tools/check_dinov3_setup.py --source /root/autodl-tmp/models/dinov3 --weights /root/autodl-tmp/models/dinov3-checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth --variant vits16plus --device cuda --size 64
```

必须输出`status: ok`、上述源码提交号、上述权重SHA-256、`logits_shape=[1,9,16,16]`。

## 4. 冒烟测试

冒烟只验证数据、前向、反向、验证和检查点重建，不能用于比较模型：

```bash
python scripts/train.py --architecture dinov3 --model /root/autodl-tmp/models/dinov3-checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth --dinov3-source /root/autodl-tmp/models/dinov3 --dinov3-variant vits16plus --dinov3-blocks 2,5,8,11 --images data/Examples/images --masks data/Examples/masks --output outputs/b00_dinov3_vits16plus_smoke --epochs 1 --batch-size 1 --gradient-accumulation 1 --crop-size 512 --encoder-lr 3e-5 --decoder-lr 1e-4 --decoder-channels 192 --boundary-weight 0.1 --workers 0 --device cuda --no-gradient-checkpointing
```

## 5. B00正式训练

仅在冒烟通过、GPU空闲且输出目录不存在后启动：

```bash
python scripts/train.py --architecture dinov3 --model /root/autodl-tmp/models/dinov3-checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth --dinov3-source /root/autodl-tmp/models/dinov3 --dinov3-variant vits16plus --dinov3-blocks 2,5,8,11 --images data/train/images --masks data/train/masks --split outputs/a03_scene_guard_inputs/candidate_split.json --class-weights outputs/a03_scene_guard_inputs/class_weights.json --output outputs/b00_dinov3_vits16plus_scene_guard --epochs 60 --batch-size 4 --gradient-accumulation 1 --crop-size 640 --encoder-lr 3e-5 --decoder-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05 --power 0.9 --val-fraction 0.1 --num-classes 9 --ignore-index 0 --decoder-channels 192 --dice-weight 0.3 --boundary-weight 0.1 --label-smoothing 0.05 --scale-min 0.75 --scale-max 1.5 --color-jitter 0.2 --rare-crop-probability 0.5 --crop-strategy legacy --seed 3407 --workers 8 --device cuda --no-gradient-checkpointing
```

正式云端运行必须放进`screen`并把stdout/stderr写入独立日志。仓库根目录可直接执行下面
这一条单行命令：

```bash
mkdir -p outputs/logs && screen -dmS b00 bash -lc 'cd /root/autodl-tmp/AIC-2026-johnmayer && python scripts/train.py --architecture dinov3 --model /root/autodl-tmp/models/dinov3-checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth --dinov3-source /root/autodl-tmp/models/dinov3 --dinov3-variant vits16plus --dinov3-blocks 2,5,8,11 --images data/train/images --masks data/train/masks --split outputs/a03_scene_guard_inputs/candidate_split.json --class-weights outputs/a03_scene_guard_inputs/class_weights.json --output outputs/b00_dinov3_vits16plus_scene_guard --epochs 60 --batch-size 4 --gradient-accumulation 1 --crop-size 640 --encoder-lr 3e-5 --decoder-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05 --power 0.9 --val-fraction 0.1 --num-classes 9 --ignore-index 0 --decoder-channels 192 --dice-weight 0.3 --boundary-weight 0.1 --label-smoothing 0.05 --scale-min 0.75 --scale-max 1.5 --color-jitter 0.2 --rare-crop-probability 0.5 --crop-strategy legacy --seed 3407 --workers 8 --device cuda --no-gradient-checkpointing 2>&1 | tee outputs/logs/b00_dinov3_vits16plus_scene_guard.log'
```

先以batch 4执行冒烟；如果OOM，只把batch降到2并把gradient accumulation升到2，保持
有效batch为4，记录为运行适配而不是算法增益。持续查看日志：

```bash
tail -f outputs/logs/b00_dinov3_vits16plus_scene_guard.log
```

断点恢复时必须重新给出与原运行相同的参数，并额外指定`--resume`。恢复过程从B00自己的
`last.pt`重建DINOv3结构并加载模型、优化器、调度器和AMP状态，不再读取初始公开权重；
但仍需要固定版本的DINOv3源码来构造骨干。不得从A线检查点恢复。

同一配置从`last.pt`恢复的单行命令：

```bash
screen -dmS b00_resume bash -lc 'cd /root/autodl-tmp/AIC-2026-johnmayer && python scripts/train.py --architecture dinov3 --model /root/autodl-tmp/models/dinov3-checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth --dinov3-source /root/autodl-tmp/models/dinov3 --dinov3-variant vits16plus --dinov3-blocks 2,5,8,11 --images data/train/images --masks data/train/masks --split outputs/a03_scene_guard_inputs/candidate_split.json --class-weights outputs/a03_scene_guard_inputs/class_weights.json --output outputs/b00_dinov3_vits16plus_scene_guard --epochs 60 --batch-size 4 --gradient-accumulation 1 --crop-size 640 --encoder-lr 3e-5 --decoder-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05 --power 0.9 --val-fraction 0.1 --num-classes 9 --ignore-index 0 --decoder-channels 192 --dice-weight 0.3 --boundary-weight 0.1 --label-smoothing 0.05 --scale-min 0.75 --scale-max 1.5 --color-jitter 0.2 --rare-crop-probability 0.5 --crop-strategy legacy --seed 3407 --workers 8 --device cuda --no-gradient-checkpointing --resume outputs/b00_dinov3_vits16plus_scene_guard/last.pt 2>&1 | tee -a outputs/logs/b00_dinov3_vits16plus_scene_guard.log'
```

## 6. 验证诊断、推理和提交

完整验证诊断：

```bash
python scripts/analyze_validation.py --images data/train/images --masks data/train/masks --split outputs/a03_scene_guard_inputs/candidate_split.json --checkpoint outputs/b00_dinov3_vits16plus_scene_guard/best.pt --output outputs/b00_dinov3_vits16plus_val_analysis --dinov3-source /root/autodl-tmp/models/dinov3 --device cuda
```

测试集1推理：

```bash
python scripts/predict.py --images data/test1/images --checkpoint outputs/b00_dinov3_vits16plus_scene_guard/best.pt --output outputs/b00_dinov3_vits16plus_scene_guard/test1_predictions --dinov3-source /root/autodl-tmp/models/dinov3 --device cuda
```

校验并打包：

```bash
python tools/make_submission.py --images data/test1/images --predictions outputs/b00_dinov3_vits16plus_scene_guard/test1_predictions --output submissions/b00_dinov3_vits16plus_scene_guard_test1.zip
```

只提交验证最佳的单个`best.pt`所产生的单尺度结果。B00不与A00或其他检查点平均。
