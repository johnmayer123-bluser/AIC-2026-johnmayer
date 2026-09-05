# A01 / A02：训练集权重与指定类别裁剪

## 当前已完成，尚未训练

2026-09-04：25项测试通过；原legacy裁剪框和随机数消耗顺序经回归测试保持一致；targeted全增强图像/标签对齐与验证不裁剪通过测试。没有运行新的正式训练，也没有推送Git。

- A01：仅将A00权重改为train-only，仍用legacy裁剪。
- A02：与A01使用相同权重，只把crop_strategy改为targeted。
- 两轮均从同一个公开MiT-B2预训练编码器开始，解码器按相同seed初始化，不从A00的best.pt/last.pt继续。
- 保留60epoch、batch4、累积1、crop640、workers8、学习率/增强/损失等A00配置。
- 模型结构、验证、预测代码未改。不得同时运行两轮占用同一张GPU。

## 代码及生成文件

| 文件 | 作用 |
|---|---|
| src/uavseg/data.py | 新增可切换targeted裁剪，默认legacy；训练图片抽样频率不变 |
| scripts/train.py | --crop-strategy选项写入config和检查点；resume禁止偷偷换裁剪/增强设置 |
| tools/train_only_weights.py | 只读取固定split中train标签的像素，复用原profile权重公式，不重建split |
| tools/audit_crops.py | 仅对训练图执行两种裁剪的分层配对检查，输出统计和对照图 |
| tests/test_targeted_crops.py | 9项新增测试：兼容性、边缘单像素、阈值、最佳回退、目标抽取、完整增强、resume、训练集隔离 |
| outputs/a01_train_only_weights/class_weights.json | 新训练权重，含train/val数量、split哈希、训练标签内容指纹 |
| outputs/a01_train_only_weights/train_mask_stats.json | 6296张训练标签的逐图类别像素统计，无验证图统计 |
| outputs/a02_crop_audit/summary.json | 裁剪检查汇总，不是模型分数 |
| outputs/a02_crop_audit/crops.csv | 每次裁剪的目标类别、框、次数、类别面积和阈值状态 |
| outputs/a02_crop_audit/examples/ | 6张预选对照图：原图/旧裁剪/新策略裁剪及其标签 |

新权重按ID0–8：

```text
[0, 0.5, 0.5201968839, 0.7428367594, 0.8859228255,
 1.6522928458, 0.5, 0.8345051634, 2.5300294808]
```

权重文件SHA256：f1dcb9480b906edca77d977064218a81e863c9fcfdc50ebb708994cd73bfcca6。
对应split SHA256：9c5ea923a5c526b842cc8ca4f15d81a776746c5cacea336f0b4b027e444d0055。

## 离线裁剪检查结果

训练集中：低占比裸地(0,5%)510张、其他裸地476张、无裸地5310张。
每组固定seed随机选64张，每图重复4次，每种策略768个crop；分层样本不代表自然训练分布。

| 低占比裸地组，256个crop/策略 | legacy | targeted |
|---|---:|---:|
| crop包含裸地的比例 | 55.47% | 65.63% |
| crop平均裸地面积占比 | 1.61% | 2.28% |
| crop平均背景面积占比 | 33.15% | 33.37% |

targeted在该组明确选中裸地91次，91次均保留裸地，其中87次达到0.5%阈值；其余4次保留8次尝试中裸地最多的裁剪。
其他裸地组命中率两者均95.70%；无裸地组没有凭空产生ID5。
目视检查0407、2957、0077三张训练对照图：图像与标签同步；普通分支下两种策略可能完全一样，这是设计保留的行为。
这些只是采样曝光结果，不是mIoU提升证据。Label.txt仅提供类别名称，粗标注的具体语义边界仍需结合官方定义核查，不擅自重标。

## 上传服务器（没有自动push）

旧实例是27158端口、connect.westd.seetacloud.com；若已换实例，请把以下命令中的端口和主机都改为当前值。
无卡模式可以完成上传；开始训练必须在GPU模式并使用原训练环境。

服务器先确认没有任务在训练，且相关文件没有需要保留的队友修改：

```bash
cd /root/autodl-tmp/AIC-2026-johnmayer
git diff -- scripts/train.py src/uavseg/data.py
cp -n scripts/train.py scripts/train.py.before_a01
cp -n src/uavseg/data.py src/uavseg/data.py.before_a01
mkdir -p outputs/a01_train_only_weights
```

若git diff显示意外修改，先停下协调，不能直接覆盖。以下在本机PowerShell执行：

```powershell
cd D:\AIC-2026\aicomp-uav-segmentation
scp -P 27158 scripts/train.py root@connect.westd.seetacloud.com:/root/autodl-tmp/AIC-2026-johnmayer/scripts/
scp -P 27158 src/uavseg/data.py root@connect.westd.seetacloud.com:/root/autodl-tmp/AIC-2026-johnmayer/src/uavseg/
scp -P 27158 tools/train_only_weights.py tools/audit_crops.py root@connect.westd.seetacloud.com:/root/autodl-tmp/AIC-2026-johnmayer/tools/
scp -P 27158 tests/test_targeted_crops.py root@connect.westd.seetacloud.com:/root/autodl-tmp/AIC-2026-johnmayer/tests/
scp -P 27158 outputs/a01_train_only_weights/class_weights.json root@connect.westd.seetacloud.com:/root/autodl-tmp/AIC-2026-johnmayer/outputs/a01_train_only_weights/
```

服务器确认：

```bash
cd /root/autodl-tmp/AIC-2026-johnmayer
python -m unittest discover -s tests -v
python scripts/train.py --help
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
sha256sum outputs/a01_train_only_weights/class_weights.json outputs/dataset_profile/split.json
```

检查hash与上面一致、CUDA为True。服务器若没有此前D00测试文件，测试总数可与本机不同，但所有已有测试都应通过。

## 先跑A01：只改类别权重

使用GPU实例、原Python训练环境；先确认输出目录不存在，不要覆盖旧实验。如果更换硬件/环境，记录差异。

```bash
cd /root/autodl-tmp/AIC-2026-johnmayer
test ! -e outputs/a01_b2_trainonly_legacy && echo "new output path OK"
mkdir -p outputs/logs
```

只有看到new output path OK再继续。下面只启动A01：

```bash
nohup python -u scripts/train.py \
  --images data/train/images \
  --masks data/train/masks \
  --split outputs/dataset_profile/split.json \
  --class-weights outputs/a01_train_only_weights/class_weights.json \
  --output outputs/a01_b2_trainonly_legacy \
  --model /root/autodl-tmp/models/mit-b2 --local-files-only \
  --epochs 60 --batch-size 4 --gradient-accumulation 1 --crop-size 640 \
  --encoder-lr 0.00003 --decoder-lr 0.0001 --weight-decay 0.01 \
  --warmup-ratio 0.05 --power 0.9 --decoder-channels 192 \
  --dice-weight 0.3 --boundary-weight 0.1 --label-smoothing 0.05 \
  --scale-min 0.75 --scale-max 1.5 --color-jitter 0.2 \
  --rare-crop-probability 0.5 --crop-strategy legacy \
  --seed 3407 --workers 8 \
  > outputs/logs/a01_b2_trainonly_legacy.log 2>&1 < /dev/null &
```

```bash
tail -n 30 outputs/logs/a01_b2_trainonly_legacy.log
```

确认samples为6296/700并正常进入第一轮。运行期间不要关机。
另保存代码和环境快照（本次代码尚未提交，因此只保存Git SHA不够）：

```bash
mkdir -p outputs/a01_repro
git rev-parse HEAD > outputs/a01_repro/git_commit.txt
git status --short > outputs/a01_repro/git_status.txt
git diff HEAD -- scripts/train.py src/uavseg/data.py > outputs/a01_repro/training_changes.patch
python -m pip freeze > outputs/a01_repro/pip_freeze.txt
sha256sum scripts/train.py src/uavseg/*.py outputs/dataset_profile/split.json outputs/a01_train_only_weights/class_weights.json > outputs/a01_repro/sha256.txt
```

## A01完成后，才运行A02

先核验A01指标与输出，保留全部资料。之后复制A01训练命令，只替换：

```text
--output outputs/a02_b2_trainonly_targeted
--crop-strategy targeted
日志路径 outputs/logs/a02_b2_trainonly_targeted.log
```

A02输出也必须是全新目录；不要加--resume，不要改epoch/batch/lr/其他增强。A02另存对应代码环境快照。
训练结束后对A01/A02各自best.pt执行analyze_validation.py，使用原split，分别写入新的验证分析目录。
比较整体mIoU、裸地precision/recall/IoU、小占比组表现、背景→裸地误报和其他类别分数。
相同seed的单次对照只是初步证据；若收益很小，需要后续重复实验排除随机波动。

## 从数据重新生成训练权重（可复现，不要覆盖现有输出）

```bash
python tools/train_only_weights.py --images data/train/images --masks data/train/masks --split outputs/dataset_profile/split.json --output outputs/a01_train_only_weights_recomputed
```

可使用新统计文件复跑裁剪检查，保持seed3407、per-group64、repeats4：

```bash
python tools/audit_crops.py --images data/train/images --masks data/train/masks --split outputs/dataset_profile/split.json --train-stats outputs/a01_train_only_weights_recomputed/train_mask_stats.json --output outputs/a02_crop_audit_server --per-group 64 --repeats 4
```
