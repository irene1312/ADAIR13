# 本地实际验证记录

日期：2026-11-24。工作目录 `D:/Research/ADAIR13`；分支 `isar-complex2ch-sim2real`；
起点 `ccb8b98e49614e07badd0641e5163fa7635c2f02`。
本记录只描述实际执行，不把 Smoke 数值当作正式模型结果。
机器可读证据摘要见 [validation_evidence.json](validation_evidence.json)。

## 已执行的真实数据预检

`local_validation/preflight_20261124T083539Z_c37e94b0/preflight.json`：

- 逐帧检查 train=660、val=120、test_seen=120、test_unseen=720；卫星文件均为 256×256×360 complex128。
- 重新计算完整 660 张训练图的均值、共享标准差、训练幅度最大值，与原 IDF manifest **逐数值一致**。
- 当前五个卫星 MAT 的 SHA256、大小、路径均已记录。未将其他卫星、guard 帧或实测数据加入这些集合。
- 飞机 Test MAT 全部实际读取检查：An26 45 帧、Yak 75 帧、JZ 68 帧，共 188 帧；Train MAT 未作为测试读取。
- 实际读取 Yake1/Yake2 的 data_test.im_ori，均为 128×128 complex128。

## 网络实测

完整双通道模型参数 28,771,432；完整 RGB 参数 28,784,824。

| 项目 | 实际结果 |
|---|---|
| 完整模型 128×128 前向 | 输出同形状、有限值；一次记录约 3.16 秒 |
| 完整模型 128×128 反向/更新 | 连续两次 L1 backward/AdamW；约 5.08 / 4.90 秒 |
| 完整模型 256×256 推理 | 输出同形状、有限值；约 5.97 秒 |
| 完整模型 133×197 推理 | pad 136×200 后裁回 133×197；约 2.36 秒 |
| 完整 RGB 模型 | 实测 `[1,3,32,40]` 前向正确；另用小配置对照初始提交源码，输出 bitwise 一致 |
| 频率诊断 | 原始低/高频、注意力后低频和梯度均有实际 hook/反向证据，详见独立报告 |

时间来自一次 CPU 运行，仅作本地工程参考，不代表 4090D 性能。512/1024 只做对应 decoder 特征尺寸的 FFT 探测，没有冒充完整大图网络前向。

## 完整模型真实数据一步 Smoke

命令：

```powershell
python scripts/train_isar.py --config configs/isar/default.json --data-root 'D:\研一实验数据资料\data_45' --output-root local_validation --smoke --steps 1 --micro-batch-size 1 --val-limit 1 --label full_model_effective_batch8
```

目录：`local_validation/SMOKE_full_model_effective_batch8_20261124T084331Z_4e6f5127`。
使用**完整原始规模模型**、真实训练数据、effective batch=8、microbatch=1、patch=128；实际 optimizer step 为 **32.609 秒左右**。
只在一个真实验证帧的 -5/0/5 dB 上做功能验证，生成 checkpoint、训练日志、验证日志和曲线。
该 run 不是正式 10000 步训练；其验证数值不作为科学结论或模型性能报告。

## 轻量真实数据训练/续训

使用明确独立的 `configs/isar/smoke.json`：dim=8、各层 1 block、refinement=1。
其余抽样/噪声/统计/optimizer 计算流程与正式代码共用。

- 两步训练：`local_validation/SMOKE_real_data_pipeline_20261124T083557Z_de7da6f5`。
- 从 latest 恢复至第 3 步：`local_validation/SMOKE_resume_real_data_20261124T083909Z_961b041b`。
- 两步的 effective batch=8、microbatch=2，optimizer step 约 1.69 / 1.36 秒。
- 每步在 3 个验证帧×3 SNR 上验证，固定训练统计量；best/latest/final 均有有效 full checkpoint 指针。

## 六组真实数据 Smoke 测试与产物审计

脚本实际运行：

```powershell
python scripts/test_all_isar.py --checkpoint local_validation/SMOKE_real_data_pipeline_20261124T083557Z_de7da6f5/best.json --satellite-root 'D:\研一实验数据资料\data_45' --aircraft-root 'D:\研一实验数据资料\shice_air_Test' --newyak-root 'D:\研一实验数据资料\newyak' --output-root local_validation/six_suite_real_smoke --smoke --limit 3 --export-mat
```

| suite | 原帧数 | 实际输入数 | PNG | NPZ | MAT |
|---|---:|---:|---:|---:|---:|
| satellite-seen | 9（正式集合 120） | 27 | 220 | 27 | 27 |
| satellite-unseen | 6（正式集合 720） | 18 | 148 | 18 | 18 |
| aircraft-raw | 9（完整 Test 集合 188） | 9 | 57 | 9 | 9 |
| aircraft-noisy | 9（完整 Test 集合 188） | 27 | 219 | 27 | 27 |
| newyak-raw | 2（完整两张） | 2 | 15 | 2 | 2 |
| newyak-noisy | 2（完整两张） | 6 | 51 | 6 | 6 |
| 合计 | | **89** | **710** | **89** | **89** |

这些输出来自轻量 Smoke 权重，标题/目录/run.json 均有标记；**不是正式全量卫星或飞机测试结果**。
默认正式命令不带 `--limit`，逐帧遍历完整集合。

输出审计命令：

```powershell
python scripts/audit_isar_outputs.py --results-root local_validation/six_suite_real_smoke
```

全部 89 份保存结果通过：从 NPZ 原图重放 IDF stable_seed/加噪并逐数组比较；从实际输入重算逐图归一化和编码 SHA256；检查反归一化 complex128、尺寸、实部/虚部/相位以及 NPZ/MAT 一致性；重算卫星指标。
raw 输入与 original 数组完全一致、noise_seed=None；所有实测结果无 PSNR/SSIM/NMSE 字段。
审计证据：`local_validation/output_audit_20261124T084557Z_cf01ac4e/audit.json`。

人工实际打开并检查以下 PNG，标题无裁切、同组共用单个色标、标签与实际任务一致：

- `.../SMOKE_satellite-seen_20261124T083851Z_0403e590/images/training_scale/Aqua_f0321_snr-5_comparison.png`
- `.../SMOKE_aircraft-raw_20261124T084032Z_63d896de/images/input_peak_scale/An26_f0001_snrraw_comparison.png`
- `.../SMOKE_newyak-noisy_20261124T084138Z_e814ec18/images/input_peak_scale/Yake1_f0001_snr-5_comparison.png`

上述 `...` 表示 `local_validation/six_suite_real_smoke`。训练 scale 和 input peak scale 的全部 PNG 均已生成。

## 自动测试

最终实际执行：`python -m pytest -q --junitxml=local_validation/pytest_final_20261124.xml`。
**20 passed in 12.27s**（本机该次运行；包含参数化后的测试项）。

覆盖：

- 原 manifest 精确帧区间、泄漏与重复检查；实际形状/统计不一致明确报错。
- 常规 MAT/HDF5 非方形哨兵数组的轴顺序、1-based 帧号、complex128/双通道顺序；拒绝 real/complex64 源数据。
- 与指定 IDF 源码对照的 noise、stable_seed、固定训练统计、3 个不同 step 的完整 batch 抽样/增强/噪声数组。
- 与 IDF 逐图统计、encode/decode、幅度 PSNR/SSIM/复数 NMSE 的精确对照，往返 float32 误差；每张图独立拟合，拒绝零方差和非有限值。
- RGB 对照初始作者源码、双通道初始化、外部 padding、全局残差和符号保留。
- 原频率空掩膜、非零跨注意力，以及非空低频掩膜时 rate_conv 仍断梯度。
- microbatch 不整除时梯度累积与完整有效 batch 的数值对照。
- 六组入口默认 per_image；raw 路径 monkeypatch 禁止调用 add_noise；Test-only 发现不读 Train。
- complex128 NPZ/MAT 的实部/虚部/相位保存；共享显示尺度；唯一运行目录。
- 小型 Synthetic MAT 完整训练/验证、checkpoint、续训；**分两段训练与连续两步的所有模型权重及 AdamW 状态 bitwise 相同**。Synthetic fixtures 保留于独立 pytest 目录，不删除。

## 尚需在 AutoDL 完成

1. 使用正式配置从头训练 10000 步、全部 120×3 验证输入选 best。
2. best 权重执行正式 seen 360、unseen 2160、飞机全部 raw/人工加噪、NewYak 全部输入。
3. 在实际 RTX 4090D/CUDA 环境测量 batch=8 或梯度累积的显存、吞吐及确定性提示。当前没有 CUDA，显存字段为 null。

本次未报告正式 PSNR/SSIM 提升，不把不充分训练的轻量恢复图用于效果结论。频率算法问题已诊断并按要求保留，是否改进需要另立实验。
IDF 参考工作树仍干净；IDF 主目录原有 `framework-v2/`、`presentation-v1/` 未跟踪目录状态未改变。
