# 第二步：冻结 A00，诊断验证错误

不重新训练，不下载预训练权重，不生成新 split，不修改原 best.pt。只用原模型推理固定的 700 张验证图。

## 已完成的检查

2026-09-04：新增独立脚本 scripts/analyze_validation.py，训练、模型、数据增强代码均未修改。
16 项单元测试通过，包含混淆矩阵方向、ignore 与背景区分、空分组、面积边界、微型模型端到端和防覆盖检查。
本机 RTX 5070 Laptop GPU 已用真实 A00 best.pt 和核验后的云端 split 跑通前 4 张：
输出 outputs/a00_val_analysis_local_smoke，status=complete，partial=true。
这不是完整验证，不作为改进分数，也不能据此验证裸地假设。完整 700 张及云端运行尚未执行。

## 推荐：直接在本机运行

本机已有完整数据和权重，并已验证 GPU 可用，不必为诊断等待云端 GPU。
在 Windows PowerShell 中运行（反引号必须为行末最后一个字符）：

```powershell
conda activate aic-uav
cd D:\AIC-2026\aicomp-uav-segmentation
python scripts/analyze_validation.py `
  --images "D:\AIC-2026\2026-低空图像语义分割赛道-训练集\train\train\images" `
  --masks "D:\AIC-2026\2026-低空图像语义分割赛道-训练集\train\train\masks" `
  --split "D:\AIC-2026\training_backups\a00_b2_boundary_pro6000\server_dataset_profile\split.json" `
  --checkpoint "D:\AIC-2026\training_backups\a00_b2_boundary_pro6000\best.pt" `
  --output outputs/a00_val_analysis `
  --top-k 12
```

不加 --limit 才是完整验证。默认 batch=1、原图推理、CUDA AMP，无 TTA，无随机增强。
完成后进度应为 700/700，run.json 应显示 status=complete、partial=false、evaluated_images=700。
脚本拒绝复用已存在的输出目录；若先前失败，用新名字如 outputs/a00_val_analysis_retry1，不删除旧资料。

## 可选：在服务器上运行

新脚本尚未推送 GitHub，不能只用 git pull 获得。先在本机 PowerShell 上传脚本。
以下地址和端口是旧实例信息；若换实例，必须替换为控制台当前 SSH 信息。

```powershell
scp -P 27158 "D:\AIC-2026\aicomp-uav-segmentation\scripts\analyze_validation.py" root@connect.westd.seetacloud.com:/root/autodl-tmp/AIC-2026-johnmayer/scripts/
```

上传可在无卡模式进行。运行前切到 GPU 模式并激活原训练 Python 环境；不要在无卡实例上直接跑默认 CUDA 命令。
该脚本复用原仓库 src/uavseg，无需新装依赖或 mmcv；best.pt 自带模型配置，不访问 Hugging Face。

在服务器 Bash 中：

```bash
cd /root/autodl-tmp/AIC-2026-johnmayer
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
ls -lh outputs/a00_b2_boundary_pro6000/best.pt outputs/dataset_profile/split.json
```

确认 CUDA 输出 True，再先跑 4 张检查：

```bash
python scripts/analyze_validation.py \
  --images data/train/images \
  --masks data/train/masks \
  --split outputs/dataset_profile/split.json \
  --checkpoint outputs/a00_b2_boundary_pro6000/best.pt \
  --output outputs/a00_val_analysis_smoke \
  --limit 4 --top-k 2
```

成功后完整验证，使用后台会话并记录日志，SSH 断开不终止：

```bash
nohup python -u scripts/analyze_validation.py \
  --images data/train/images \
  --masks data/train/masks \
  --split outputs/dataset_profile/split.json \
  --checkpoint outputs/a00_b2_boundary_pro6000/best.pt \
  --output outputs/a00_val_analysis \
  --top-k 12 > outputs/a00_val_analysis.log 2>&1 < /dev/null &
```

```bash
tail -n 20 outputs/a00_val_analysis.log
cat outputs/a00_val_analysis/run.json
```

run.json 最初为 running，成功才变为 complete。文件夹存在不代表成功。
重新执行完整命令前需改用新的输出目录和日志名称，防止覆盖日志。

完成后在本机 PowerShell 下载（若本机已运行本地路线，不要下载到同名已有目录）：

```powershell
scp -P 27158 -r root@connect.westd.seetacloud.com:/root/autodl-tmp/AIC-2026-johnmayer/outputs/a00_val_analysis "D:\AIC-2026\training_backups\a00_b2_boundary_pro6000\"
```

## 输出文件看什么

| 文件 | 用途 |
| --- | --- |
| run.json | 完成状态、是否仅抽样、数量、设备/版本、权重和 split 的 SHA256、源码 SHA256、与检查点记录的分差 |
| metrics.json | 全验证 mIoU、逐类指标、混淆矩阵、每个裸地面积组的详细统计、解释限制 |
| per_class.csv | 各类 precision、recall、IoU、真实/预测像素数；重点看裸地 ID5 |
| confusion_counts.csv | 行为真值，列为预测：每种真实类别被分到了哪里 |
| confusion_row_rates.csv | 每行除以该类真实像素数；裸地→背景为第5类行、第1类列，不是转置方向 |
| per_image.csv | 每图 mIoU、逐类 IoU、裸地面积、裸地召回和主要混淆比例 |
| barren_area_groups.csv | 裸地缺失、(0,5%)、[5%,20%)、[20%,50%)、[50%,100%] 分组，含图数和像素数 |
| selected_cases.json | 最差 mIoU 和最差裸地召回的图片名单，最多各12张 |
| cases/ | 原图、真值、预测、错误图四联对照，重复入选的图片只保存一份 |
| predictions/ | 所有已评估验证图的原尺寸单通道类别ID预测，非测试提交 |

CSV 比例为 0–1。CSV 空白/JSON null 表示无法定义，不是零分。
整体 mIoU 来自累计像素混淆矩阵，不等于 per_image.csv 的 mIoU 算术平均。
真值0忽略；真值非0而预测为0仍计入漏检。背景是ID1，不能和忽略ID0混淆。

## 如何判断下一步

1. 先确认完整验证 mIoU 是否接近原 0.7994679067716781。本地和服务器版本/硬件不同，可能有数值差异；明显偏差时先核对配置、输入数据、split 和权重，不能当成模型改进/退化。
2. 看裸地主要被预测为背景、农田还是其他类，同时检查 precision 和 recall，区分漏检与误报。
3. 看裸地面积越大时召回是否下降、裸地→背景比例是否上升。每组按像素累计，不是按图平均；组内图片少或场景不同会干扰解释，不能直接推出面积导致错误。
4. 用 cases 图和 per_image 定位图像，检查错误是集中边缘还是区域内部。对照图仅为显示缩小，指标使用原图；脚本尚未计算独立的边界/内部指标。
5. 根据证据设计下一项单因素实验；暂不同时改模型、采样、batch、epoch。验证诊断也不能证明测试集的逐类错误，因为没有测试真值。

给助手查看：run.json、metrics.json、barren_area_groups.csv、per_image.csv 和 cases 目录。
