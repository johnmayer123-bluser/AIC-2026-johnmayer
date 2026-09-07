# 团队交接指南

更新时间：2026-09-07

本文档是三名队员及各自 Codex 的共同交接入口。开始工作前先读本文、`README.md`、
`AGENTS.md` 和 `docs/EXPERIMENTS.md`。

## 当前状态（2026-09-07，以本节为准）

- 当前仓库：`johnmayer123-bluser/AIC-2026-johnmayer`；本地路径 `D:\AIC-2026\aicomp-uav-segmentation`。旧仓库与S00信息保留在后面的历史快照中，不代表当前仓库连接。
- A00正式实验已完成：BA-SegFormer-B2，6296训练/700验证，seed3407，crop640，batch4×累积1，60epoch。
- 最佳为第60轮，验证mIoU=0.7994679067716781；用户报告官方测试集1 mIoU=0.669362，相差13.0106个百分点。
- GPU日志为RTX PRO 6000 Blackwell Server Edition；PyTorch2.12.1+cu130，CUDA13.0。总参数24,749,546，全部可训练，AMP开启，梯度检查点实际关闭。
- 已核验本地best.pt/last.pt均为第60轮且模型参数一致；500张预测与提交ZIP逐字节一致、格式有效。
- 第48轮已达到79.8784%，最后12轮只再提升0.0684个百分点；下一轮优先检查泛化与验证独立性，不直接认定缺少训练轮数。
- 完整登记见 `docs/EXPERIMENTS.md`，逐文件/算法/数据处理复盘见 `docs/A00_REVIEW.md`。
- 云端复现资料已通过无卡模式补录并下载核验：三份profile JSON与服务器SHA256一致，与旧本地副本仅LF/CRLF换行不同，划分与数值完全相同；云端事后HEAD为a6668b5、工作区干净，Python3.12.3与完整pip freeze已备份。此为事后快照，不冒充训练开始时的代码/环境快照。
- A03 scene-guard代理划分基线已完成：最佳验证mIoU=0.7686515018，官方测试集1=0.664582。A04仅关闭边界辅助损失，最佳验证mIoU=0.7690574955但官方降至0.661388；A04被否定，后续恢复boundary weight 0.1。A00仍以0.669362保持官方最高。
- D03标签一致性审计：3551个高可信重叠对像素冲突率0.4204%；边界/内部冲突率2.7886%/0.2765%，边界单位像素风险约10.09倍。该结果支持研究容忍边界，不支持人工改标签或把全部冲突当真实错标。
- B00 DINOv3 ViT-S+/16单模型路线已在本地实现但尚未训练。采用block 2/5/8/11 Token密集特征、1/4 RGB细节支路和现有边界门控解码；训练/预测/诊断已兼容旧A线检查点。官方权重和源码不进Git，完整说明见`docs/B00_DINOV3_GUIDE.md`。

### 当前文件位置

- 云端仓库：`/root/autodl-tmp/AIC-2026-johnmayer`。
- 云端训练输出：`/root/autodl-tmp/AIC-2026-johnmayer/outputs/a00_b2_boundary_pro6000`。
- 云端预训练编码器：`/root/autodl-tmp/models/mit-b2`。
- 云端提交：`/root/autodl-tmp/AIC-2026-johnmayer/submissions/a00_b2_boundary_pro6000_test1.zip`。
- 本地完整训练备份：`D:\AIC-2026\training_backups\a00_b2_boundary_pro6000`。
- 本地提交：`D:\AIC-2026\submissions\a00_b2_boundary_pro6000_test1.zip`。
- 本地DINOv3公开权重：`D:\AIC-2026\DINOv权重文件\checkpoints\dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`（不进入Git）。

### 当前下一步

1. 保留A00为当前官方最佳和冻结回退方案；不覆盖A00-A04任何输出。
2. 先向组委会确认DINOv3 LVD-1689M公开学术预训练权重是否合规；未经确认只做内部候选实验。
3. 将本次B线代码提交并同步服务器，再按`docs/B00_DINOV3_GUIDE.md`固定Meta官方源码提交、上传ViT-S+权重并运行setup checker。
4. 先跑11图B00冒烟；只确认前向、反向、验证、检查点重建和显存，不用冒烟mIoU选模型。
5. 冒烟通过后，按A03相同scene-guard划分、train-only权重、legacy增强和损失正式训练B00。B00从公开DINOv3权重初始化，不续训任何A线模型。
6. B00最佳验证与A03同口径比较；再决定是否使用一次官方测试提交。任何结论都需同时记录验证、逐类、最后10轮稳定性和官方成绩。
7. 工作区`run_d03.ps1`和`tools/audit_label_consistency.py`为用户未跟踪文件，本次B线实现未修改或删除它们。

## 2026-09-03历史交接快照（以下不作为当前操作指令）

为保留S00和迁移前记录，以下文本按历史保存。其中仓库、硬件、路径和“下一步”均已由上面的当前状态替代。

### 当时的状态

- 私有 GitHub 仓库：`CheeseMirror-Gao/aicomp-uav-segmentation`
- 当前基线：单个 `SegFormer-B0`，公开初始权重 `nvidia/mit-b0`
- 基线代码提交：`e6c59a5 Add SegFormer B0 training baseline`
- 日志警告修复：`4bc0869 Avoid tensor conversion warning in training log`
- 数据体检：11 对样例图像/掩码均健康，尺寸为 `1024x1024`，标签 ID 为 `0-8`
- 云端环境：PyTorch 2.8.0、Python 3.12、CUDA 12.8、单卡 RTX 4090D 24GB 已验证
- 完整闭环已跑通：公开权重下载 → 微调 → 验证 mIoU → 保存检查点 → 生成预测掩码
- 官方正式数据已下载到本地压缩包：训练集6996对、测试集1共500张，尚未解压进仓库忽略目录
- 已按正式赛题重构候选方案：单个 BA-SegFormer-B2，公开MiT-B2编码器加本地边界感知解码器
- 新增正式数据解压、完整profile、协变量感知固定划分、类别权重、断点训练、预测校验和ZIP打包流程
- 本机为RTX 5070 8GB；旧的Python 3.7/PyTorch 1.6/CUDA 10.1环境不兼容，README要求新建Python 3.11 + CUDA 13.0 PyTorch环境
- 重构后的代码已通过8项CPU单元测试，但尚未在本机GPU上完成模型下载、显存冒烟或正式训练

## 已完成的冒烟实验

这不是正式成绩，只用于确认程序可以端到端运行。

```text
样例总数：11
训练/验证：9/2
模型：SegFormer-B0
训练轮数：2
batch size：2
crop size：512
学习率：0.00006
随机种子：3407
第1轮 mIoU：0.043504218693065026
第2轮 mIoU：0.07028324067868295
最佳 mIoU：0.07028324067868295
```

正式数据到手后应从公开 `nvidia/mit-b0` 权重重新训练，不要从这次 11 张样例的
`last.pt` 继续训练。

## 文件在哪里

GitHub 只保存代码和文档，不保存数据与权重。当前云端约定路径：

```text
/root/autodl-tmp/aicomp/repo/aicomp-uav-segmentation  # Git 仓库
/root/autodl-tmp/aicomp/datasets                       # 数据集
/root/autodl-tmp/aicomp/models                         # 公开预训练权重缓存
/root/autodl-tmp/aicomp/outputs/b0_smoke/best.pt       # 冒烟实验最佳权重
/root/autodl-tmp/aicomp/outputs/b0_smoke/last.pt       # 冒烟实验最后状态
/root/autodl-tmp/aicomp/outputs/b0_smoke/config.json   # 实验参数
/root/autodl-tmp/aicomp/outputs/b0_smoke/metrics.jsonl # 每轮指标
/root/autodl-tmp/aicomp/logs/b0_smoke.log              # 训练日志
```

这些云端文件是否仍存在，接手者必须通过 `ls` 验证，不能只依赖本文档。

## 新队员或新 Codex 如何接手

1. 接受 GitHub Collaborator 邀请并克隆仓库。
2. 打开仓库后，让 Codex 先读四个文件：`README.md`、`AGENTS.md`、本文和实验表。
3. 运行：

   ```bash
   git status
   git log --oneline -5
   python -m unittest discover -s tests -v
   ```

4. 不要立即重写训练代码；先说明准备验证的一个假设。
5. 需要云端操作时，通过私下渠道获取当前实例的 SSH 信息。密码和 Token 不得进入仓库或 Codex 提示词。
6. 登录云端后先检查：

   ```bash
   nvidia-smi
   df -h /root/autodl-tmp
   ls -lah /root/autodl-tmp/aicomp
   screen -ls
   ```

7. 群内声明已接管云端，确认无人正在训练后再操作。

## 日常协作流程

开始代码工作前：

```bash
git pull
git switch -c <姓名或缩写>-<任务名称>
```

修改后运行测试，提交并推送分支。合并前说明唯一改动、预期影响和验证结果。云端实验必须
使用一个全新的实验编号和输出目录，禁止覆盖已有结果。

开始训练前在 `docs/EXPERIMENTS.md` 预登记实验；结束后补充 mIoU、各类别 IoU、耗时和
结论。交接时更新本文档的“当前状态”和“下一步”。

## 训练会话交接

所有长训练都应在命名的 `screen` 会话中运行，例如：

```bash
screen -S a01
```

离开但不中断训练：按 `Ctrl+A`，松开后按 `D`。接手者查看和恢复：

```bash
screen -ls
screen -r a01
tail -n 50 /root/autodl-tmp/aicomp/logs/a01.log
```

继续训练使用对应实验的 `last.pt`；推理和提交候选使用验证 mIoU 最好的 `best.pt`。

## 关机与切换实例

按量实例关机后停止 GPU 计费，但不会预留 GPU。关机前必须：

1. 确认没有仍需运行的进程，或确认已保存 `last.pt`。
2. 保存 `best.pt`、`last.pt`、`config.json`、`metrics.jsonl` 和日志。
3. 将代码提交并推送 GitHub。
4. 更新本文档和实验表。
5. 将关键权重备份到本地或可靠网盘。

原实例无卡时，不要先删除原实例。使用 AutoDL 的克隆/迁移功能，优先同地区，并勾选复制
数据盘。新实例使用新的 SSH 地址、端口和密码。登录后执行：

```bash
nvidia-smi
df -h /root/autodl-tmp
ls -lah /root/autodl-tmp/aicomp
```

只有确认代码、数据、模型缓存、权重和日志完整后，才能释放旧实例。若未复制数据盘，必须
通过跨实例拷贝、AutoDL 网盘或本地备份恢复，不能假设 GitHub 包含这些文件。

## 当前下一步

1. 按README新建 `aic-uav` 环境并确认 `torch.cuda.is_available()` 为True。
2. 运行 `tools/prepare_dataset.py` 解压正式数据，再运行完整数据体检。
3. 运行 `tools/profile_dataset.py` 生成 `profile.json`、`split.json` 和 `class_weights.json`。
4. 使用A00完整参数先跑1轮显存与速度冒烟；若OOM，只把crop从640改为512并另记实验。
5. 冒烟成功后从公开 `nvidia/mit-b2` 开始60轮正式训练，保留逐类IoU和最佳检查点。
6. 用 `best.pt` 推理测试集1，经 `tools/make_submission.py` 强制校验后提交。
7. 最终提交只能使用一个模型，禁止模型集成、权重平均或预测平均。

