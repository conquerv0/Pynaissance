## 1. Pytorch Fundamentals

### 1.1 Introduction

- Pytorch是一个基于Lua编写的Torch库上的深度学习库，针对机器学习工程提供了从数据处理，数据包装，神经网络模型层等完整模块。
- Pytorch主要优势在于其简洁和更符合直觉的逻辑设计。

### 1.2 Installation and Configuration

- Pytorch安装基本是Anaconda/miniconda + Pytorch + Pycharm的工具包

  **1. Anaconda**

- 安装anaconda: [Anaconda | Individual Edition](https://www.anaconda.com/products/individual)

- 配置虚拟环境：打开anaconda后：

~~~
conda env list
~~~

- 创建虚拟环境： 

~~~
conda create -n 虚拟环境名称 python=版本名称
~~~

- 删除虚拟环境命令:

~~~
conda remove -n 名称 -all
~~~

- 激活环境命令: 

~~~
conda activate 名称
~~~

**2. Pytorch**

-  离线下载pytorch
- 首先下载对应版本的Pytorch和torchvisions： https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
- 打开~anaconda prompt

~~~
cd 刚下载的压缩包的路径
conda install --offline pytorch压缩包的全称（后缀都不能忘记）
conda install --offline torchvision压缩包的全称（后缀都不能忘记）
~~~

- 检验安装

~~~ Python
import torch
torch.cuda.is_available()
~~~

### 1.3 Related Resources(Guide, Official Doc)

1. [Awesome-pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list)

2. [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)

3. [Pytorch-handbook](https://github.com/zergtant/pytorch-handbook)

4. [PyTorch官方社区](https://discuss.pytorch.org/)

5. [Awesome-pytorch-list-CN-version](https://github.com/xavier-zy/Awesome-pytorch-list-CNVersion)

   
