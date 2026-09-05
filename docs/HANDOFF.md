# 团队交接指南

更新时间：2026-09-05

本文档是三名队员及各自 Codex 的共同交接入口。开始工作前先读本文、`README.md`、
`AGENTS.md` 和 `docs/EXPERIMENTS.md`。

## 当前状态（2026-09-05，以本节为准）

- 当前仓库：`johnmayer123-bluser/AIC-2026-johnmayer`；本地路径 `D:\AIC-2026\aicomp-uav-segmentation`。旧仓库与S00信息保留在后面的历史快照中，不代表当前仓库连接。
- A00正式实验已完成：BA-SegFormer-B2，6296训练/700验证，seed3407，crop640，batch4×累积1，60epoch。
- 最佳为第60轮，验证mIoU=0.7994679067716781；用户报告官方测试集1 mIoU=0.669362，相差13.0106个百分点。
- GPU日志为RTX PRO 6000 Blackwell Server Edition；PyTorch2.12.1+cu130，CUDA13.0。总参数24,749,546，全部可训练，AMP开启，梯度检查点实际关闭。
- 已核验本地best.pt/last.pt均为第60轮且模型参数一致；500张预测与提交ZIP逐字节一致、格式有效。
- 第48轮已达到79.8784%，最后12轮只再提升0.0684个百分点；下一轮优先检查泛化与验证独立性，不直接认定缺少训练轮数。
- 完整登记见 `docs/EXPERIMENTS.md`，逐文件/算法/数据处理复盘见 `docs/A00_REVIEW.md`。
- 云端复现资料已通过无卡模式补录并下载核验：三份profile JSON与服务器SHA256一致，与旧本地副本仅LF/CRLF换行不同，划分与数值完全相同；云端事后HEAD为a6668b5、工作区干净，Python3.12.3与完整pip freeze已备份。此为事后快照，不冒充训练开始时的代码/环境快照。

### 当前文件位置

- 云端仓库：`/root/autodl-tmp/AIC-2026-johnmayer`。
- 云端训练输出：`/root/autodl-tmp/AIC-2026-johnmayer/outputs/a00_b2_boundary_pro6000`。
- 云端预训练编码器：`/root/autodl-tmp/models/mit-b2`。
- 云端提交：`/root/autodl-tmp/AIC-2026-johnmayer/submissions/a00_b2_boundary_pro6000_test1.zip`。
- 本地完整训练备份：`D:\AIC-2026\training_backups\a00_b2_boundary_pro6000`。
- 本地提交：`D:\AIC-2026\submissions\a00_b2_boundary_pro6000_test1.zip`。

### 当前下一步

1. 保留A00为冻结基线，不覆盖原输出，不因增加epoch/batch直接修改原检查点续训。
2. 云端dataset_profile、事后Git SHA和完整依赖已补齐，位于本地训练备份的server_dataset_profile与repro_info目录；官方评分回执仍待补充。后续使用核验后的云端split，避免重新生成划分。
3. D00完整验证诊断已完成：700张、mIoU=0.7994675430，与A00原验证基本一致。详见docs/EXPERIMENTS.md的D00完成记录；输出outputs/a00_val_analysis。裸地与背景双向混淆突出，低占比裸地漏检更严重；大于等于50%占比仅5张，不能推出面积因果。D01/D02场景重叠审计也已完成，结论见本节第10–12项；尚无测试逐类分数，不猜测测试掉分类别。
4. A00历史权重含验证标签统计；A01/A02已改用train-only权重，原split保持不变。若更换验证划分，先建立新的可比基线。
5. 算法、增强、采样、batch和epoch改动分别登记，禁止一次改动多个因素后声称某一模块有效。
6. 2026-09-04补充：A01/A02均已完成60轮并下载，本机700图诊断完成。A01最佳epoch60、79.90996%；A02最佳epoch54、80.13857%。A02裸地IoU/precision上升但recall下降，低占比组IoU27.00%→26.79%，不能声称解决低占比漏检。最后10轮均分A02略低于A01，收益稳定性未证实。用户已报告官方成绩：A00=0.669362、A01=0.668672、A02=0.667215（此前0.599012为误报，已更正）。A00仍为最高官方成绩；A01为train-only规范对照，targeted暂不作为默认改进，未启动A03。
7. 本机RGB审计已完成，见outputs/domain_audit：6296 train/700 val/500 test，跨组完全相同RGB哈希组为0；哈希近似筛出22张val的27对候选、11张test的16对候选，不能把候选数当作重复数。目视确认val 4808/1239/1990分别与train 4707/6757/3885有明显同场景重叠，现有划分不能称为场景独立；测试也有同场景案例test1_134与4684。该筛查会漏掉平移/旋转/相邻切片，尚未建立完整场景分组。
8. 外观描述：train/val/test平均亮度0.39890/0.39558/0.37991；test较暗，但差异约-0.167个train标准差，不能据此归因约13个百分点落差。下一步优先补做场景来源/几何重叠核查，提出新分组划分后另建基线；不删除图片、不更改官方标签、不直接覆盖原split。新增审计脚本与测试，完整27项测试通过。未推送Git。
9. 用户当前工作方式：本机可执行命令由助手执行，只将需要用户运行的服务器命令直接放在对话框；不另写操作指令文档。本次审计不需要服务器操作。
10. 2026-09-05场景审计扩展完成：赛题资料确认patch已重新排序且无来源映射，6996张PNG均无info/EXIF。SIFT+视觉词袋检索每图48候选，共核验295,036对，得到3756对strong和2058对review。原划分689对strong跨界；385张val与train直接strong相连，按strong连通分量为393/700张val与train相连。冻结A00在检出重叠/未检出重叠子集mIoU分别84.2296%/76.1391%，差8.0904个百分点；场景难度混杂，不能把差值当纯泄漏效应。
11. 保守草案outputs/scene_audit_expanded/candidate_split.json：6296/700，SHA256=4294be282fb8546ccde5d93694a618f9e45bfd815eb1e836f9a1b9669ff51f7a；当前检出的strong/review跨界均为0，评分类像素份额约9.35%–10.69%，但2570个单例仍是来源未知，不得宣称严格场景独立。草案train-only权重位于outputs/a03_scene_guard_weights，class_weights SHA256=06bdf521c9f5c4467e5241765d979e46a9f5c0f80603013d831efcb65d4032d6。
12. 下一步尚未实施：将草案split与新权重同步到服务器，先从公开MiT-B2初始化、按A01原配置重跑“新划分基线”，只改变split和由split必然派生的train-only权重。用新实验编号和新输出目录，不续训旧检查点。新旧验证分数不可直接作为算法增益比较；建立新基线后才在同一新划分上做单因素域泛化实验。用户要求本机命令由助手执行，服务器命令才发对话框。

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

