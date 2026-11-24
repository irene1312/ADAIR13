# AdaIR 复数 ISAR 去噪

本分支 `isar-complex2ch-sim2real` 在作者 AdaIR 上增加独立 ISAR 流程。原始 `train.py`、`test.py`、光学配置与 RGB 默认模型保持兼容。新流程不使用 Lightning、WandB、光学数据或 RGB 预训练权重。

参考实现、精确提交和许可证见 [参考来源](docs/REFERENCE_PROVENANCE.md)。
频率模块的已验证退化现象见 [频率诊断](docs/FREQUENCY_DIAGNOSTICS.md)。
本地实际验收记录见 [本地验证](docs/LOCAL_VALIDATION.md)。

## 环境

建议从仓库根目录运行命令。服务器项目路径为 `/root/autodl-tmp/ADAIR13`。
保留已工作的 PyTorch/CUDA 环境，不使用原 `env.yaml` 去覆盖 AutoDL 环境。
Python 3.10+、PyTorch 2.x；其他依赖：

```bash
cd /root/autodl-tmp/ADAIR13
python -m pip install -r requirements-isar.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

本地 Windows 已验证 Python 3.12.7 / torch 2.5.1+cpu / NumPy 1.26.4 /
SciPy 1.13.1 / h5py 3.11.0 / einops 0.8.1 / scikit-image 0.24.0 /
Matplotlib 3.9.2 / pytest 7.4.4。缺失的 einops 仅安装到本仓库 `.local_deps`，未替换全局 torch/CUDA。
`scripts/_bootstrap.py` 和测试配置会读取该目录；AutoDL 正常 pip 安装即可。

Windows PowerShell 示例（数据位置通过参数提供，代码不绑定该路径）：

```powershell
Set-Location D:\Research\ADAIR13
python scripts/preflight_isar.py --data-root 'D:\研一实验数据资料\data_45' --aircraft-root 'D:\研一实验数据资料\shice_air_Test' --newyak-root 'D:\研一实验数据资料\newyak'
python -m pytest -q
python scripts/diagnose_isar.py --device cpu --threads 4
```

## 网络与复数接口

- 正式配置：[configs/isar/default.json](configs/isar/default.json)。dim=48、blocks=[4,6,6,8]、refinement=4、heads=[1,2,4,8]、FFN=2.66、WithBias、bias=False、decoder=True。
- 输入输出均为 float32 `[B,2,H,W]`，通道依次 Re、Im。完整双通道模型 **28,771,432** 参数；原 RGB 模型 **28,784,824** 参数。
- 三个 FreModule 的 `conv` 和 `conv1` 都接收双通道。Transformer/MDTA/GDFN、AFLB、调制、跨注意力、上下采样、skip 和全局残差完整保留。
- 无最终激活或像素裁剪，允许正负实部虚部。源 MAT 与恢复复数图像为 complex128。
- 模型外按 8 的倍数向右/下 reflect padding，过小尺寸用 replicate；统计量先在未 padding 的输入上计算，输出裁回原尺寸。没有图像 resize。FreModule **内部**作者原有 bilinear 特征尺度匹配和 FFT/IFFT/abs 保留。
- 不修复原整数频率掩膜；这属于后续独立实验，而非本次忠实复现。

## 数据与固定 manifest

```text
/root/autodl-tmp/data_45/           # 保持 IDF manifest 中卫星相对路径
/root/autodl-tmp/datasets/          # 递归查找 An26_Test.mat、Yak_Test.mat、JZ_Test.mat
/root/autodl-tmp/newyak/newyak/      # yake1.mat、yake2.mat
```

只接受原 `manifests/isar_sim.json`，不自动发现替代文件、不重划分、不忽略异常。
MATLAB 帧索引从 1 开始：

| 集合 | 目标 | 每目标帧 | 总数 |
|---|---|---|---:|
| train | Aqua/HST/Beidou | 21–240 | 660 |
| val | Aqua/HST/Beidou | 261–300 | 120 |
| seen test | Aqua/HST/Beidou | 321–360 | 120 |
| unseen test | CALIPSO/WV2 | 1–360 | 720 |

guard 和 manifest 内其他卫星不参与这些集合。实测数据不参与训练、验证、训练统计或最佳模型选择。

**实际 IDF manifest 和本地卫星文件均为 256×256×360 complex double**。
训练裁取 128×128；验证与测试使用原始完整空间尺寸。若服务器仅有 128×128 版本，预检会报错；不要为绕过报错修改 manifest 或重算替代统计量，需要先核对数据来源。
支持 scipy 常规 MAT、MATLAB v7.3 HDF5 逆维度读取及 real/imag compound；NewYak 读取 `data_test.im_ori`，也支持 HDF5 struct 引用。
飞机只读 Test MAT 的全部帧，不读取 Train MAT。正式完整飞机帧数由 Test 文件决定，本地为 An26=45、Yak=75、JZ=68。NewYak 每文件一张。

服务器完整预检（卫星集合完整逐帧校验，计算 660 张干净训练图统计量并比对 manifest，记录当前文件 SHA256）：

```bash
python scripts/preflight_isar.py --data-root /root/autodl-tmp/data_45 --aircraft-root /root/autodl-tmp/datasets --newyak-root /root/autodl-tmp/newyak/newyak
```

manifest 没有历史文件字节哈希，因此历史内容一致性的可核查依据是路径、目标、帧数、形状、类型和训练统计量；本次预检另保存当前数据 SHA256，断点续训据此拒绝文件内容变化。

## 训练、加噪与归一化

正式默认：seed=10、noise_seed=20261027、patch=128、effective batch=8、steps=10000、AdamW、固定 lr=1e-4、weight_decay=1e-4、grad_clip=0.1、Re/Im L1、amplitude_l1_weight=0、每 1000 步验证/保存、每 10 步日志、验证 SNR=[-5,0,5]。

每步使用 `default_rng(SeedSequence([seed,step]))`，先在排序目标中抽一个目标池，再为整个有效 batch 依次抽帧、裁剪、rot90、可选上下翻转、均匀抽 SNR∈[-5,5]、抽实部和虚部高斯噪声。顺序与 IDF `training_batch` 完全一致。没有额外 epoch 扩增预算。

```text
P = mean(abs(clean)**2)
sigma = sqrt(P / (2 * 10**(SNR/10)))
noise = sigma * (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))
noisy = clean + noise
```

不把实际噪声二次缩放到精确目标 SNR。测试 seed 沿用 IDF SHA256 的 `stable_seed(noise_seed,target,frame,SNR)`，与模型、manifest 顺序和子集无关。

训练/训练期间验证始终使用干净训练集固定统计量：

```text
mean_real = -1.079110754821375e-07
mean_imag =  1.3169225919155399e-07
shared_std = 0.012623665528847389
amplitude_scale = 1.0000000000000002
pixels_per_channel = 43253760
```

这套统计量写入配置和 checkpoint。没有使用实测飞机、未见卫星或测试数据估计它们。

**所有正式测试默认 `--normalization per_image`**：先加噪（raw 不加噪），再从这一张实际未 padding 输入以 float64 计算独立 Re/Im 均值和共享总体标准差。

```text
mu_r = mean(real(z)); mu_i = mean(imag(z))
std = sqrt(sum((real(z)-mu_r)**2 + (imag(z)-mu_i)**2)/(2*z.size))
x = [(real(z)-mu_r)/std, (imag(z)-mu_i)/std]
restored = (output[0]*std+mu_r) + 1j*(output[1]*std+mu_i)
```

严格保留 IDF encode 边界：原始 Re/Im 先转换 float32，再以 float64 做上述减均值/除标准差，最后 float32 进入网络；逐图统计自身始终从原 complex128 输入计算。因此往返误差包含两次 float32 舍入。反归一化用输入的同一组参数恢复 complex128。
禁止 clean/reference 拟合带噪统计量、batch 拟合、RMS/Min-Max 替代和零方差兜底。非有限值、零方差、源文件非 complex128 明确报错。
可用 `--normalization fixed_train` 做独立对照，不改变 checkpoint 训练统计或指标 amplitude_scale。

## AutoDL 从头正式训练与续训

```bash
cd /root/autodl-tmp/ADAIR13
python scripts/train_isar.py --config configs/isar/default.json --data-root /root/autodl-tmp/data_45 --device cuda --threads 4 --label adair_full
```

RTX 4090D 显存不足时，从一开始选择 microbatch=1 或 2，仍每步采样原来的 8 张：

```bash
python scripts/train_isar.py --config configs/isar/default.json --data-root /root/autodl-tmp/data_45 --device cuda --micro-batch-size 1 --label adair_full_accum8
```

每个微批 loss 按样本数/8 加权，累积完成后只裁剪一次、optimizer.step 一次；包括不能整除的最后微批。网络没有依赖 batch 统计的 BatchNorm。分块浮点归约可能有微小数值差异，不承诺 GPU bitwise 一致。

把下例路径替换成实际训练输出目录：

```bash
python scripts/train_isar.py --config configs/isar/default.json --data-root /root/autodl-tmp/data_45 --device cuda --micro-batch-size 1 --resume experiments/isar/train_actual_run/latest.json --label adair_resume
```

`steps=10000` 是累计 optimizer step 上限，不是续训额外 10000 步。
每次训练/续训创建新目录。所有 `step_XXXXXX.pt` 保留，latest/best/final.json 是显式 checkpoint 指针；只有当前活动 run 的 latest/best 指针随训练更新，不覆盖旧 run。
checkpoint 含完整模型、AdamW、step、best_score、best_checkpoint、完整配置、训练统计、manifest 哈希、训练/验证源文件哈希及 Python/NumPy/Torch/CUDA RNG。
最佳权重按全部 120×3 验证输入逐图 restored_psnr 平均选择。最终步总是验证并保存。
续训验证数值配置及数据身份；只允许改变总步数（Smoke）、日志/保存间隔、微批大小和输出路径等非协议选项。
跨机器迁移请保留历史最佳 step 文件；优先使用同目录相对指针，续训可在 checkpoint 同目录按原文件名定位历史 best。
只加载显式指定的可信本地 checkpoint，不搜索或下载其他权重。

可选 `--preload-train --preload-mib 1536` 只读预加载，增加 RAM 使用但不改变抽样次序；默认关闭。
训练逐步记录耗时、裁剪前梯度范数、CUDA 峰值 allocated；completed.json 另含 reserved 峰值。当前 CPU 验证不提供虚构的 4090D 显存结果。

## 六组正式测试

把变量设为本次正式训练的 `best.json`：

```bash
CKPT=/root/autodl-tmp/ADAIR13/experiments/isar/REPLACE_WITH_ACTUAL_RUN/best.json

python scripts/test_isar.py --suite satellite-seen --checkpoint "$CKPT" --data-root /root/autodl-tmp/data_45 --device cuda
python scripts/test_isar.py --suite satellite-unseen --checkpoint "$CKPT" --data-root /root/autodl-tmp/data_45 --device cuda
python scripts/test_isar.py --suite aircraft-raw --checkpoint "$CKPT" --data-root /root/autodl-tmp/datasets --device cuda
python scripts/test_isar.py --suite aircraft-noisy --checkpoint "$CKPT" --data-root /root/autodl-tmp/datasets --device cuda
python scripts/test_isar.py --suite newyak-raw --checkpoint "$CKPT" --data-root /root/autodl-tmp/newyak/newyak --device cuda
python scripts/test_isar.py --suite newyak-noisy --checkpoint "$CKPT" --data-root /root/autodl-tmp/newyak/newyak --device cuda
```

一次执行六组（与上述相同默认 per_image、SNR=-5/0/5、完整集合）：

```bash
python scripts/test_all_isar.py --checkpoint "$CKPT" --device cuda --satellite-root /root/autodl-tmp/data_45 --aircraft-root /root/autodl-tmp/datasets --newyak-root /root/autodl-tmp/newyak/newyak --save-complex all --export-mat
```

正式卫星 seen=120×3=360 输入，unseen=720×3=2160 输入。
飞机 raw=全部 Test 帧、不添加噪声；noisy=全部 Test 帧×3 SNR。NewYak raw=2 输入、noisy=6 输入。
raw 实测没有可靠无噪真值，实测所有实验都不计算 PSNR/SSIM/NMSE，只记录能量比、输入/输出峰值和 RMS。能量比不是精度指标，也不以接近 1 为最优准则。

卫星 PSNR/SSIM 使用物理复数幅度除固定训练 amplitude_scale，**仅指标**裁剪到 [0,1]。SSIM data_range=1、win_size=7、gaussian_weights=False、use_sample_covariance=True。
NMSE 使用反归一化后的完整复数值，无裁剪。`*_nmse_db` 是逐图 dB 的算术平均；`*_nmse_db_of_mean` 是平均 NMSE 后转 dB。
按目标/SNR 和 ALL 汇总。ALL 是全部逐图均值，因此不同目标帧数不同时按帧数加权。

## 输出与图像

每次测试独立创建 `results/isar/<suite>_<UTC时间>_<随机后缀>/`，不清理历史结果：

```text
run.json / completed.json / SUMMARY.md
per_image.csv / summary.csv / results.json
normalization_stats.jsonl
images/training_scale/*.png
images/input_peak_scale/*.png
curves/{psnr,ssim,nmse,nmse_db}_snr.png     # 卫星
curves/{energy_ratio,input_rms,restored_rms}_snr.png # 实测辅助诊断
complex/*.npz                           # 默认 all
complex/*.mat                           # --export-mat
```

训练目录另含 `config.json`、`train.csv`、`validation.csv`、loss-step/验证曲线、完整 step checkpoint、最佳/最新/最终指针，以及每次验证的逐图指标。

逐图归一化日志包含实际输入 SHA256、原图 SHA256、float32 编码 SHA256、源文件与帧、SNR/噪声 seed、输入/输出 dtype、原尺寸/padding 尺寸、mean_real/mean_imag/shared_std、训练 amplitude_scale 及 input peak。checkpoint SHA256、完整配置与数据身份记录在 run.json。

每目标按 `(mat_path,frame)` 排序后 linspace 等间隔取 3 张；正式 seen 帧为 321/340/360，unseen 为 1/180/360，An26=1/23/45、Yak=1/38/75、JZ=1/34/68，NewYak 各一张。
每选定帧/SNR 保存单图和共享色标对比图，hot/nearest，均为反归一化物理幅度。两种显示：固定训练 amplitude_scale、该实际输入 peak（raw 用 raw peak，人工噪声用 noisy peak）。同组所有子图用相同范围。图像不作为推理输入，也不改变实际复数输出。

NPZ/MAT 保存 reference（卫星）或 original_measured（实测）、实际 input、restored complex128，以及 restored_real、restored_imag、restored_phase。保存策略 `--save-complex all|selected|none`，默认 all；MAT 是额外导出。

## Smoke 与验收

轻量配置只用于功能检查，不得混入正式复现。任何非默认数值配置或限制验证/测试集都必须显式 `--smoke`，生成 SMOKE 目录、元数据和图标题。

```bash
python -m pytest -q
python scripts/diagnose_isar.py --device cuda
python scripts/train_isar.py --config configs/isar/smoke.json --data-root /root/autodl-tmp/data_45 --device cuda --smoke --label smoke
python scripts/test_all_isar.py --checkpoint /ABSOLUTE/SMOKE_RUN/best.json --device cuda --smoke --limit 3 --output-root results/smoke
```

没有真实数据时：

```bash
python scripts/make_smoke_isar.py --output-root local_validation
# 将打印的绝对目录替换到以下三个位置；synthetic config 自带对应 manifest/data_root。
python scripts/train_isar.py --config /ABSOLUTE/SYNTHETIC_SMOKE_DATA/config.json --smoke
python scripts/test_all_isar.py --checkpoint /ABSOLUTE/SYNTHETIC_TRAIN_RUN/best.json --manifest /ABSOLUTE/SYNTHETIC_SMOKE_DATA/manifest.json --satellite-root /ABSOLUTE/SYNTHETIC_SMOKE_DATA/satellite --aircraft-root /ABSOLUTE/SYNTHETIC_SMOKE_DATA/aircraft --newyak-root /ABSOLUTE/SYNTHETIC_SMOKE_DATA/newyak --smoke --limit 3
```

pytest 使用保留的独立 `local_validation/pytest_*` 目录，不自动删除 fixtures。
`.local_deps`、local_validation、experiments、results、MAT、NPZ、权重均被 Git 忽略，不上传数据或大型结果。
所有代码均不提供自动删除、清理或结果保留裁剪操作。
