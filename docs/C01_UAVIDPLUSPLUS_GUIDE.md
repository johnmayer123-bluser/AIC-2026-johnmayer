# C01：LoveDA → UAVid++ → AIC顺序迁移

状态：无卡准备代码完成，尚未启动UAVid++或AIC训练。

## 实验问题

C00已经给DINOv3增加LoveDA遥感地物先验。C01只插入一个UAVid++适配阶段，测试低空
斜视场景和更精细车辆标签能否继续改善最终AIC表现。C01不是把数据粗暴拼接：顺序迁移
避免2522张LoveDA图和200张UAVid++帧之间的采样比例成为额外变量。

参赛者报告组委会已明确允许外部有标注数据训练。务必保留带日期的原始回复；仓库中的
`approval_reference`只是审计索引，不能代替比赛方证据。UAVid++及原UAVid图像受
CC BY-NC-SA 4.0约束，仅用于符合其条款的非商业研究/比赛用途。

## 数据选择与映射

只使用以下两包：

- `UAVid_rgb_only.zip`：420张原始RGB帧。
- `UAVid++_labels.zip`：修订并扩展后的11类标签。

不要使用`UAVid+_labels.zip`或原始UAVid标签；三者对应同一批图像，混用会给同一像素
制造互相冲突的监督。

| UAVid++ | RGB | AIC目标 |
|---|---|---|
| Background Clutter | 0,0,0 | 1 Background |
| Building (Wall) | 128,0,0 | 2 Building |
| Road | 128,64,128 | 3 Road |
| Tree / Low Vegetation | 0,128,0 / 128,128,0 | 6 Vegetation |
| Dynamic / Static Car | 64,0,128 / 192,0,192 | 8 Vehicle |
| Human | 64,64,0 | 0 Ignore |
| Water | 0,0,255 | 4 Water |
| Sky | 128,255,255 | 0 Ignore |
| Roof | 70,70,70 | 2 Building |

Human和Sky没有稳妥的AIC等价类，因此不强行塞进Background。UAVid++没有Barren和
Agricultural，它们会在最终AIC阶段继续学习。

## 无卡实例：上传、校验与转换

以下命令均为单行。先在服务器执行：

```bash
mkdir -p /root/autodl-tmp/dataset_archives/uavidplusplus
```

然后在本机PowerShell执行：

```powershell
scp -P 41280 "D:\AIC-2026\UAVid++\UAVid_rgb_only.zip\UAVid_rgb_only.zip" "D:\AIC-2026\UAVid++\UAVid++_labels.zip" root@connect.westd.seetacloud.com:/root/autodl-tmp/dataset_archives/uavidplusplus/
```

回到服务器。同步包含C01工具的Git提交后执行：

```bash
cd /root/autodl-tmp/AIC-2026-johnmayer
sha256sum /root/autodl-tmp/dataset_archives/uavidplusplus/UAVid_rgb_only.zip /root/autodl-tmp/dataset_archives/uavidplusplus/UAVid++_labels.zip
python tools/prepare_uavidplusplus.py --rgb-archive /root/autodl-tmp/dataset_archives/uavidplusplus/UAVid_rgb_only.zip --labels-archive /root/autodl-tmp/dataset_archives/uavidplusplus/UAVid++_labels.zip --output data/c01_uavidplusplus --tile-size 1088 --workers 8 --expected-rgb-sha256 845AA0BE9370E18A787380A52862EB296E35D5BDB62ADDA5CA6AAE0CA0A4A887 --expected-labels-sha256 F92F2B31372F4B4BAAB986DD6B8B6D7DD72E0CE85CB53A1DA1243144CA7741D8 --approval-reference organizer-chat-2026-09-08
python tools/check_dataset.py --images data/c01_uavidplusplus/images --masks data/c01_uavidplusplus/masks --output outputs/c01_uavidplusplus_check --expected-size 1088x1088 --class-ids 0,1,2,3,4,5,6,7,8 --ignore-ids 0
python tools/train_only_weights.py --images data/c01_uavidplusplus/images --masks data/c01_uavidplusplus/masks --split data/c01_uavidplusplus/split.json --output outputs/c01_uavidplusplus_weights --num-classes 9
python -c "import json; from pathlib import Path; m=json.loads(Path('data/c01_uavidplusplus/metadata.json').read_text()); s=json.loads(Path('data/c01_uavidplusplus/split.json').read_text()); c=json.loads(Path('outputs/c01_uavidplusplus_check/summary.json').read_text()); print('frames=',m['frame_count']); print('tiles=',m['tile_count']); print('split=',len(s['train']),len(s['val'])); print('healthy=',c['healthy']); print('issues=',c['records_with_issues']); print('test_labels_opened=',m['test_labels_opened'])"
df -h /root/autodl-tmp
```

预期验收：`frames={'train': 200, 'val': 70}`、`tiles={'train': 1600, 'val': 560}`、
`split=1600 560`、`healthy=True`、`issues=0`、`test_labels_opened=False`。这里停止，不需要GPU。

转换器先打印并核验两个整包SHA256；随后只解码Train/Val成员，ZIP读取过程会校验这些
实际使用成员的CRC。它不额外扫描或解码Test内容，因此避免对5.9GB RGB包做一次无意义
的重复全包读取。

## 有GPU后才执行的两阶段训练

第一阶段从C00 LoveDA最佳8类检查点初始化UAVid++ 9类模型；必须看到只重置
`classifier.1.weight/bias`。建议先跑15轮，而不是照搬UAVid++论文40轮，因为本实验是
域适配而非从零训练。第二阶段从UAVid++最佳9类检查点初始化AIC，并完全复用C00的AIC
split、类别权重和60轮设置。具体启动命令在无卡验收后再执行，避免数据准备异常时误开训练。

最终比较必须同时报告C00/C01在同一AIC验证划分上的最佳mIoU和官方测试mIoU；不能用
UAVid++验证分数代替AIC指标，也不能把C01与B00做验证口径不一致的直接排名。

数据与方法依据：

- UAVid++数据卡：<https://huggingface.co/datasets/vivianchiciudean/uavidplusplus>
- UAVid++官方代码：<https://github.com/vivichiciudean/uavidplusplus-code>
- UAVid++项目页：<https://vivichiciudean.github.io/uavidplusplus/>
