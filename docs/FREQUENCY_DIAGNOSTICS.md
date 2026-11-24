# AdaIR 原始频率掩膜与梯度诊断

日期：2026-11-24。本报告区分网络内部的实值特征频域处理和最终复数 ISAR 输出。
没有修改作者 `FreModule.fft()`、频率分割规则或初始化方式。

## 实际执行依据

复现命令：`python scripts/diagnose_isar.py --device cpu --threads 4`。
完整原始规模双通道 AdaIR，28,771,432 参数，float32；输入随机 seed=10。
实际记录位于：

`local_validation/full_model_diagnostics_20261124T084035Z_a9e22e92/diagnostics.json`

执行了完整模型 128×128、256×256、133×197 前向；非整除尺寸由外部 padding 成 136×200。
另外按 512/1024 输入对应的 decoder 特征尺寸执行三个 FFT 分割器；后两项**不是完整大图模型推理**。
128×128 完整模型执行两次 L1 反向，第二次在第一次 AdamW 更新之后。
hook 记录 conv1 特征尺寸、rate_conv sigmoid、原整数半径、原始 FFT 高频/低频输出范数和 `channel_cross_h` 输出。
另外单独测试特征尺寸 512×512 的 FreModule.fft 前向/反向，以排除掩膜为空和 para1=0 对诊断的干扰。

## 空间尺寸与掩膜

`FreModule.forward(inp_img,y)` 首先将原图插值到 y 的空间大小，再经 conv1 提取频域特征。
三个 y 分别位于 H/8、H/4、H/2，通道分别为 384、192、96。

| 原输入/探测范围 | fre1 特征 | fre2 特征 | fre3 特征 | 三个低频掩膜非零比例 |
|---|---|---|---|---|
| 128×128 完整前向 | 16×16 | 32×32 | 64×64 | 0 / 0 / 0 |
| 256×256 完整前向 | 32×32 | 64×64 | 128×128 | 0 / 0 / 0 |
| 133×197 → pad 136×200 完整前向 | 17×25 | 34×50 | 68×100 | 0 / 0 / 0 |
| 512×512 对应 FFT 尺寸探测 | 64×64 | 128×128 | 256×256 | 0 / 0 / 0（本次初始化/输入） |
| 1024×1024 对应 FFT 尺寸探测 | 128×128 | 256×256 | 512×512 | 0 / 0 / 0.000030517578125 |

高频比例是 1−低频比例。因此常用 128/256 完整输入下，实际观测为高频掩膜全部为 1、低频掩膜全部为 0。
512 对应 fre3 测得半径 `[0,1]`，其中一维为空仍使整个低频矩形为空。
1024 对应 fre3 测得 `[2,1]`，有 8/262144 个低频位置非零。
不能据此把所有大尺寸输入的低频分支都断言为全零；结果依赖阈值及两维的整数半径。

原因是原代码：

```python
h_ = (h // 128 * threshold[i,0,:,:]).int()
w_ = (w // 128 * threshold[i,1,:,:]).int()
mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1
```

当特征 h<128，h//128=0；当 h=128 且 sigmoid 阈值未饱和到 1，正数截断仍为 0。
任一方向半径为 0，切片宽度就为 0。训练 patch=128 的三个 AFLB 因此均遇到退化。

## 原始低频与跨注意力必须区分

128×128 完整模型中，三个 `fft()` 原始 low 范数均为 0。
随后 `low_feature = channel_cross_h(low_feature, y)` 的输出范数分别为：
**1.0669621、1.8088018、4.1402979**，确实非零。
256×256 时分别为 **2.1702516、3.6223836、8.3092098**。

该跨注意力的 Q 来自原始 low，K/V 来自非零 decoder 特征 y。
bias=False 下 Q=0，attention logits=0，softmax 仍给出均匀权重，进而聚合非零 V。
因此“没有独立提取的低频输入”成立，“整个后续低频特征必然全零”不成立。
函数名也容易误导：原代码 `channel_cross_l` 处理 high，而 `channel_cross_h` 处理 low；本次保持原命名和调用关系。

## 实测反向梯度

对三个 AFLB 一共 93 个 parameter tensors 分类：

| 阶段 | grad=None | grad 全零 | grad 非零 |
|---|---:|---:|---:|
| 初始化 para1=0 时第一次反向 | 15 | 72 | 6 |
| 一次 AdamW 更新后的第二次反向 | 15 | 9 | 69 |

每个模块的 `para1` 初始为 0，`para2` 为 1，返回 `out*para1 + y*para2`。
所以第一次所有实际走过的频率分支梯度都为零，但 para1/para2 有梯度；这不能简单认定为所有频率参数永久无效。
更新 para1 后 conv1、跨注意力大部分参数、FreRefine 参数获得非零梯度。

持续无梯度的每模块 5 个 tensors：

- `conv.weight`：作者 forward/fft 没有使用这个层。
- `score_gen.weight/bias`：作者 forward/fft 没有使用这两个参数。
- `rate_conv.0.weight`、`rate_conv.2.weight`：forward 执行，但结果只经 `.int()` 进入离散切片边界，无法从 loss 回传。

第二次每模块还存在 3 个全零梯度 tensors：
`channel_cross_h.temperature`、`channel_cross_h.q.weight`、`channel_cross_h.q_dwconv.weight`。
这是空低频输入/Q=0 的结果。该分支 kv/project_out 参数仍可有非零梯度。
这里按完整参数 tensor 分类，合并 kv tensor 的非零梯度不表示其中每一个元素都非零。

单独 512×512 特征 FFT 探测的原始 low 范数 **7.0948472**（非空），conv1 梯度范数 **0.2002935**，
而 rate_conv 两层仍均为 grad=None。说明 rate_conv 的断梯度不只源于小尺寸空掩膜。

## 影响、保留策略和独立改进方向

当前复现训练仍可以通过全局残差、Transformer 和 gate 更新优化；但不能声称原 rate_conv 在通过 loss 学习频率边界。
常见 ISAR 尺寸中两分支没有形成原本期望的互补低/高频提取；低频跨注意力仍可能从 decoder 的 V 获取内容。
本次全部保留，作为忠实原始基线，相关未使用权重也保留在参数量与 checkpoint 中。

未来可单独设计对照实验：检查按相对尺寸定义的最小非空整数窗口；采用可微 soft mask；或明确研究离散选择梯度估计。
它们分别改变边界含义、频率分割算法或梯度定义，需要单独分支、单独配置与重新训练，不能混入当前复现结果。
不建议只把空掩膜强制设一格后，就宣称阈值网络已可学习；整数索引的梯度问题仍在。

## 后续向 IDF 拆分移植的接口

| 模块 | 输入→输出 | 注意事项 |
|---|---|---|
| `FreModule` / AFLB | `raw [B,2,H0,W0]`, `y [B,C,H,W]` → `[B,C,H,W]` | `in_dim=2`；内部匹配 y 尺寸；C 能满足 C//16≥1，C 整除 heads；保留/单独研究频率边界与 para1=0 门控 |
| `FreRefine` / FMoM | low、high 两个 `[B,C,H,W]` → `[B,C,H,W]` | 空间门、通道门、1×1投影；不能将特征幅度误认为最终 ISAR 幅度 |
| `Chanel_Cross_Attention` | x(Q)、y(K/V) 同形状 `[B,C,H,W]` → 同形状 | 保留 Q/K/V 的来源及通道注意力方向；零 Q 仍有均匀聚合 V |
| `TransformerBlock` | `[B,C,H,W]` → 同形状 | WithBias LayerNorm、MDTA、GDFN、两个残差；C 整除 heads |
| Patch/上下采样/skip | 图像→层次特征→图像 | 属于整网骨架；不宜直接当作 IDF 的单个迭代更新替换件 |

未来整合应由 IDF 原复数 encode/decode 管理输入输出；本次没有修改 IDF，也没有实施移植。
