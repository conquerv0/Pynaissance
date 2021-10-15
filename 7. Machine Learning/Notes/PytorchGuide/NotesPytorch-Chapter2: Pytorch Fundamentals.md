## 1. Tensor

### 1. 基本定义： 

> 张量是基于向量和矩阵的推广，比如我们可以将标量视为零阶张量，矢量可以视为一阶张量，矩阵就是二阶张量。

- 0维张量/**标量** 标量是一个数字
- 1维张量/**向量** 1维张量称为“向量”。
- 2维张量 2维张量称为**矩阵**
- 3维张量 公用数据存储在张量 时间序列数据 股价 文本数据 彩色图片(**RGB**)
- 4维=图像，5维=视频

例如一个图像可以如下表达：

```
(width, height, channel) = 30
```

### 2. Tensor的基本性质和操作

```python
from __future__ import print_function
import torch

# 随机初始化矩阵
x = torch.rand(4, 3)
print(x)

# 构筑全为0的矩阵
x = torch.zeros(4, 3, dtype=torch.long)
print(x)

tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])

# 获取维度信息
print(x.size())
print(x.shape)
```

- 一些常见函数: 

  | 函数                                  | 功能                   |
  | ------------------------------------- | ---------------------- |
  | Tensor(**sizes*)                      | 基础构造函数           |
  | tensor(*data*)                        | 类似于np.array         |
  | ones(**sizes*)                        | 全1                    |
  | zeros(**sizes*)                       | 全0                    |
  | eye(**sizes*)                         | 对角为1，其余为0       |
  | arange(*s,e,step*)                    | 从s到e，步长为step     |
  | linspace(*s,e,steps*)                 | 从s到e，均匀分成step份 |
  | rand/randn(**sizes*)                  |                        |
  | normal(*mean,std*)/uniform(*from,to*) | 正态分布/均匀分布      |
  | randperm(*m*)                         | 随机排列               |

- 加法操作

```
```

### 3. 自动求导Autograd

> PyTorch 中，所有神经网络的核心是 `autograd `包。autograd包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义 ( define-by-run ）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的。

```python
import torch
# 创建张量，并设置requires_grad=True来追踪计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

```python
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
```

在这个张量上进行计算:

```python
y = x**2
print(y)

z = y * y * 3
out = z.mean()
print(z, out)
```

```python
# y是计算结果，所以具有grad_fn属性。
tensor([[1., 1.],
        [1., 1.]], grad_fn=<PowBackward0>)

# .requires_grad_()
tensor([[3., 3.],
        [3., 3.]], grad_fn=<MulBackward0>) tensor(3., grad_fn=<MeanBackward0>)
```

- 梯度:

进行反向传播

```python
out.backward()
# 输出导数 d(out) / dx
print(x.grad)
```

Note：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。

```python
# 再来反向传播⼀一次，注意grad是累加的 2 out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
```

### 4. 并行计算

> Cuda与并行计算：`CUDA`是我们使用GPU的提供商——NVIDIA提供的GPU并行计算框架。对于GPU本身的编程，使用的是`CUDA`语言来实现的。在PyTorch中，当我们使用了 `cuda()` 时，其功能是让我们的模型或者数据迁移到GPU当中，通过GPU开始计算。

**做并行的方法**

1. **网络结构分布到不同的设备中(Network partitioning)**

> 将模型部分拆分，把不同部分放入GPU做不同任务的计算。但是GPU之间的传输通信是个挑战，所以此方法逐渐淘汰。

![模型并行.png](https://github.com/datawhalechina/thorough-pytorch/raw/main/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20PyTorch%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/figures/%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C.png)

2. **同一层的任务分布到不同数据中**(**Layer-wise partitioning**)

> 将同层模型拆分，让不同GPU训练同一层模型的不同任务部分。可以保证组件传输，但是需要大量训练。

![拆分.png](https://github.com/datawhalechina/thorough-pytorch/raw/main/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20PyTorch%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/figures/%E6%8B%86%E5%88%86.png)

3. **不同的数据分布到不同的设备中，执行相同的任务(Data parallelism)**

> 不拆分模型，而是拆分数据。同一个模型在不同gpu中训练其中一部分数据，最后将输出数据汇总反传。

![数据并行.png](https://github.com/datawhalechina/thorough-pytorch/raw/main/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20PyTorch%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/figures/%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C.png)

