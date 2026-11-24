# 参考代码来源

本次只读参考了 `D:/Research/IDF/presentation-v1`。该工作树分支为
`experiment/scgn-per-image-normalization`，HEAD 为
`f5ca1811af03c4f7a3c37a5f13040a6b50d8eca6`。没有修改该项目。

对应上游：[irene1312/IDF 指定提交](https://github.com/irene1312/IDF/tree/f5ca1811af03c4f7a3c37a5f13040a6b50d8eca6)。

本项目从该提交通过 `git show` 原样提取 `manifests/isar_sim.json`。
换行统一为 LF 后 SHA256：
`a1f01317ad40d1895fe4393b1cf75aa44ca01de7c95d5057ab9589f6a0dbaa2c`。
原 manifest 中额外目标和 guard/unused 记录完整保留，但不进入本次训练/验证/测试集合。

以下文件以该提交为基础：

| 本项目文件 | 参考与调整 |
|---|---|
| `isar/data.py` | IDF 同名文件；增加源 MAT complex128 强校验 |
| `isar/noise.py` | IDF 同名文件，原样保留 |
| `isar/metrics.py` | IDF 同名文件，原样保留 |
| `isar/newyak.py` | IDF 同名文件，补充 HDF5 struct/reference 支持 |
| `isar/normalization.py` | IDF `test_normalization.input_stats` 和 `experiment.encode/decode` 的公式/计算次序；默认改为 per_image，增加异常校验 |
| `isar/experiment.py::training_batch` | IDF 同名函数的采样、裁剪、增强、SNR 与高斯随机数顺序；仅保留 uniform 模式 |
| `isar/evaluation.py::aggregate` | 延用逐图算术平均，同时区分平均 NMSE(dB) 和 dB(平均 NMSE) |
| `isar/visualization.py` | 延用按排序帧索引 linspace 取 3 张、hot/nearest/共享物理幅度色标 |

IDF 的 Apache-2.0 许可证全文保存于 [IDF_LICENSE.txt](IDF_LICENSE.txt)。
没有复制 IDF 网络、迭代更新、LCM 或其他创新模块。

AdaIR 网络来自本项目初始提交 `ccb8b98e49614e07badd0641e5163fa7635c2f02`。
`net/model.py` 唯一结构适配是将 `inp_channels` 传给三个 FreModule。
Patch Embedding 和输出卷积原本已经参数化，ISAR 配置直接使用 2。

本地回归测试直接在内存中执行指定 IDF 源码作对照，未导入/写入 IDF 的 pycache。
噪声、种子、训练采样、归一化、反归一化和指标已逐数组或逐数值比较。
不存在指定本地参考工作树的环境中，这些参考对照测试明确 skip，其余测试仍可执行。
