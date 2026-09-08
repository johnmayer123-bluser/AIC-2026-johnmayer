# C00 LoveDA 两阶段迁移（仅研究，禁止比赛提交）

状态：代码与测试就绪，尚未下载LoveDA或训练。

## 1. 边界与实验问题

C00回答一个研究问题：在相同DINOv3 ViT-S+/16结构下，先用类别接近的LoveDA学习
遥感地物，再只用AIC官方训练集微调，是否改善官方验证表现。

LoveDA是外部数据。本项目规则禁止额外训练数据，因此C00、其派生检查点和预测均不得
用于AIC排行榜或最终提交。可提交的对照仍是只使用官方数据的B01。

C00不读取AIC测试图，不下载LoveDA Test，不生成AIC测试预测。代码包含三层保护：

- LoveDA准备目录和每次训练输出都写入`DO_NOT_SUBMIT.txt`。
- 从research-only检查点初始化或恢复时必须继续传`--research-only`。
- `scripts/predict.py`默认拒绝research-only检查点。

## 2. 数据和类别

使用LoveDA官方Zenodo的Train和Val，许可证为CC BY-NC-SA 4.0且仅限学术、非商业用途。
官方数据页：<https://zenodo.org/records/5706578>；官方代码：
<https://github.com/Junjue-Wang/LoveDA>。

LoveDA标签与AIC的映射：

| LoveDA ID | LoveDA | C00含义 |
|---:|---|---|
| 0 | no-data | Ignore |
| 1 | background | Background |
| 2 | building | Building |
| 3 | road | Road |
| 4 | water | Water |
| 5 | barren | Barren |
| 6 | forest | Vegetation |
| 7 | agriculture | Agricultural |
| - | 无 | Vehicle（第二阶段重新初始化分类头） |

第一阶段使用8通道分类头（0到7）；第二阶段建立9通道分类头，只迁移形状兼容的编码器、
Token投影、RGB细节支路、融合、局部细化和边界分支。`classifier.1.weight/bias`重新随机
初始化，优化器、调度器、AMP scaler和epoch全部重置。

## 3. 下载与校验

服务器数据盘至少预留15GB：

```bash
df -h /root/autodl-tmp
```

```bash
mkdir -p /root/autodl-tmp/dataset_archives /root/autodl-tmp/datasets/loveda_raw
```

```bash
wget -c -O /root/autodl-tmp/dataset_archives/loveda_Train.zip 'https://zenodo.org/records/5706578/files/Train.zip?download=1'
```

```bash
wget -c -O /root/autodl-tmp/dataset_archives/loveda_Val.zip 'https://zenodo.org/records/5706578/files/Val.zip?download=1'
```

```bash
echo 'de2b196043ed9b4af1690b3f9a7d558f  /root/autodl-tmp/dataset_archives/loveda_Train.zip' | md5sum -c -
```

```bash
echo '84cae2577468ff0b5386758bb386d31d  /root/autodl-tmp/dataset_archives/loveda_Val.zip' | md5sum -c -
```

```bash
unzip -q /root/autodl-tmp/dataset_archives/loveda_Train.zip -d /root/autodl-tmp/datasets/loveda_raw
```

```bash
unzip -q /root/autodl-tmp/dataset_archives/loveda_Val.zip -d /root/autodl-tmp/datasets/loveda_raw
```

定位实际根目录；预期能看到Urban/Rural下的`images_png`和`masks_png`：

```bash
find /root/autodl-tmp/datasets/loveda_raw -maxdepth 4 -type d | sort
```

以下命令假设解压后为`loveda_raw/Train`和`loveda_raw/Val`。准备器校验尺寸、配对、标签
ID并创建零复制符号链接，不读取Test：

```bash
python tools/prepare_loveda.py --train-root /root/autodl-tmp/datasets/loveda_raw/Train --val-root /root/autodl-tmp/datasets/loveda_raw/Val --output data/research_loveda_DO_NOT_SUBMIT
```

```bash
python tools/check_dataset.py --images data/research_loveda_DO_NOT_SUBMIT/images --masks data/research_loveda_DO_NOT_SUBMIT/masks --output outputs/c00_loveda_dataset_check_DO_NOT_SUBMIT --expected-size 1024x1024 --class-ids 0,1,2,3,4,5,6,7 --ignore-ids 0
```

只用LoveDA Train统计8类训练权重：

```bash
python tools/train_only_weights.py --images data/research_loveda_DO_NOT_SUBMIT/images --masks data/research_loveda_DO_NOT_SUBMIT/masks --split data/research_loveda_DO_NOT_SUBMIT/split.json --output outputs/c00_loveda_weights_DO_NOT_SUBMIT --num-classes 8
```

## 4. 第一阶段：LoveDA适配

第一阶段固定30轮。它是外部数据适配，不与B00/B01验证成绩直接比较：

```bash
mkdir -p outputs/logs && screen -dmS c00_loveda bash -lc 'cd /root/autodl-tmp/AIC-2026-johnmayer && python -u scripts/train.py --architecture dinov3 --model /root/autodl-tmp/models/dinov3-checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth --dinov3-source /root/autodl-tmp/models/dinov3 --dinov3-variant vits16plus --dinov3-blocks 2,5,8,11 --images data/research_loveda_DO_NOT_SUBMIT/images --masks data/research_loveda_DO_NOT_SUBMIT/masks --split data/research_loveda_DO_NOT_SUBMIT/split.json --class-weights outputs/c00_loveda_weights_DO_NOT_SUBMIT/class_weights.json --output outputs/c00_loveda_pretrain_DO_NOT_SUBMIT --epochs 30 --batch-size 4 --gradient-accumulation 1 --crop-size 640 --encoder-lr 3e-5 --decoder-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05 --power 0.9 --num-classes 8 --ignore-index 0 --decoder-channels 192 --dice-weight 0.3 --boundary-weight 0.1 --label-smoothing 0.05 --scale-min 0.75 --scale-max 1.5 --color-jitter 0.2 --rare-crop-probability 0.5 --crop-strategy legacy --seed 3407 --workers 8 --device cuda --no-gradient-checkpointing --research-only 2>&1 | tee outputs/logs/c00_loveda_pretrain_DO_NOT_SUBMIT.log'
```

```bash
tail -f outputs/logs/c00_loveda_pretrain_DO_NOT_SUBMIT.log
```

## 5. 第二阶段：AIC官方数据研究性微调

第二阶段使用与B01相同的A00原split及对应train-only权重。唯一额外因素是LoveDA第一阶段
初始化。为保证结论可解释，应先完成只用官方数据的B01对照。

```bash
screen -dmS c00_aic bash -lc 'cd /root/autodl-tmp/AIC-2026-johnmayer && python -u scripts/train.py --architecture dinov3 --model /root/autodl-tmp/models/dinov3-checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth --dinov3-source /root/autodl-tmp/models/dinov3 --dinov3-variant vits16plus --dinov3-blocks 2,5,8,11 --images data/train/images --masks data/train/masks --split outputs/dataset_profile/split.json --class-weights outputs/a01_train_only_weights/class_weights.json --output outputs/c00_aic_finetune_DO_NOT_SUBMIT --epochs 60 --batch-size 4 --gradient-accumulation 1 --crop-size 640 --encoder-lr 3e-5 --decoder-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05 --power 0.9 --num-classes 9 --ignore-index 0 --decoder-channels 192 --dice-weight 0.3 --boundary-weight 0.1 --label-smoothing 0.05 --scale-min 0.75 --scale-max 1.5 --color-jitter 0.2 --rare-crop-probability 0.5 --crop-strategy legacy --seed 3407 --workers 8 --device cuda --no-gradient-checkpointing --init-checkpoint outputs/c00_loveda_pretrain_DO_NOT_SUBMIT/best.pt --research-only 2>&1 | tee outputs/logs/c00_aic_finetune_DO_NOT_SUBMIT.log'
```

启动日志必须出现初始化报告，且只允许以下两项重新初始化：

```text
classifier.1.bias
classifier.1.weight
```

```bash
tail -f outputs/logs/c00_aic_finetune_DO_NOT_SUBMIT.log
```

## 6. 仅做官方验证诊断

可以在A00的700张验证集上研究，不执行AIC测试预测、不打包提交：

```bash
python scripts/analyze_validation.py --images data/train/images --masks data/train/masks --split outputs/dataset_profile/split.json --checkpoint outputs/c00_aic_finetune_DO_NOT_SUBMIT/best.pt --output outputs/c00_aic_val_analysis_DO_NOT_SUBMIT --dinov3-source /root/autodl-tmp/models/dinov3 --device cuda
```

C00结果只能写成研究观察，不能报告为合规排行榜方法，也不能与B01官方成绩合并、平均或
集成。研究结束后保留配置、日志、哈希和验证结果；不得把数据、权重或预测提交Git。
