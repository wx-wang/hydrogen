---
layout: posts
title: 神经网络BP法推导
date: 2020-03-25 19:50:19
tags: 机器学习
categories: [学习笔记, 计算机]
mathjax: true
---

## 参考文献：

1. 吴恩达机器学习；
2. 周志华《机器学习》
3. [神经网络，BP算法的理解与推导](https://zhuanlan.zhihu.com/p/45190898)
4. [矩阵向量求导链式法则](https://www.cnblogs.com/pinard/p/10825264.html)
5. [MarkDown 插入数学公式实验大集合](https://juejin.im/post/5a6721bd518825733201c4a2)
6. [Latex的公式输入](https://www.jianshu.com/p/05987743d27c)
7. [Markdown下LaTeX公式、编号、对齐](https://www.zybuluo.com/fyywy520/note/82980)

## 神经网络算法介绍

神经网络包括：**输入层，隐藏层，输出层**。取一个隐含层$i$，包括输入值$\vec{a}$，权值矩阵$\bf{W}$，输出值$\vec{o}$。关于输入值与输出值，有这样一个关系：
$$a_i^j = \sum_{i=1}^{i=p}{w_{ki}^{j-1}o_k^{j-1}} \tag{1}$$
$$o_i^j = f(a_i^j) \tag{2}$$

其中，$f(x)$为sigmod函数 ：$f(x)=\frac{1}{1+e^{-x}}$。其中，上角标j代表第j层神经网络，下角标i代表第i个神经元。以向量和矩阵形式表达，则可得到：
$$\vec{a^j}=W^{j-1}\vec{o^{j-1}} \tag{3}$$
其中：
$$W^{j} = \begin{bmatrix} w_{11} & w_{21} & \cdots & w_{p1}\\
\cdots & \cdots \\
 w_{1q} & w_{2q} & \cdots & w_{pq}\end{bmatrix}^{j-1}_{q\times p},
 \vec{a^j}=\begin{Bmatrix}
a_1\\a_2\\ \cdots \\a_p
\end{Bmatrix}_{p\times1}^{j}$$

$$\vec{o^j}=\begin{Bmatrix}
o_1\\o_2\\ \cdots \\o_p
\end{Bmatrix}_{p\times1}^{j}$$

在输入$x$后，通过给定的权值，每一层都可经过公式（1）和公式（2）神经网络正向反馈，可得到输出值$\hat y$。

## BP反向传播算法推导
 
为了让神经网络算法能够较好拟合出所需要的结果，需要对参数W进行训练。对于复杂的神经网络，可采用BP算法进行反向传播推导。而在训练时，对于参数的训练，可采用梯度下降法进行训练：

$$w = w - \lambda \frac{\partial C}{\partial w} \tag{4}$$

其中，$\lambda$为学习率，C为损失函数(cost function)。对于神经网络，定义损失函数为：

$$C = \frac{1}{2}||\hat y - y||_2 \tag{5}$$

进行训练参数推导。假设神经网络共有L层，对于最后一层，有：

$$
\Delta{w} = -\lambda \frac{\partial C}{\partial w}
= -\lambda \frac{\partial C}{\partial a^L} \frac{\partial a^L}{\partial w}
= -\lambda \frac{\partial C}{\partial a^L} (o^{L-1})^T
= -\lambda\epsilon^L(o^{L-1})^T
$$
其中：
$$
\epsilon^L = \begin{Bmatrix}
\epsilon_1^L\\
\epsilon_2^L\\ 
\cdots \\
\epsilon_p^L
\end{Bmatrix}_{p\times1}^{j}
$$

令误差$\epsilon^L=\frac{\partial C}{\partial a^L}$。
误差的求解方法如下：

$$
\frac{\partial C}{\partial a^L}=
\frac{\partial C}{\partial \hat y}\frac{\partial \hat y}{\partial a^L}
=(\hat y-y)\hat y^T(1-\hat y)
$$


为了求隐藏层中的权重系数W，不妨采用数学归纳法，当已知$\epsilon^i$时，求$\epsilon^{i-1}$：

$$
\epsilon^{i-1}=\frac{\partial C}{\partial a^{L-1}}
=\frac{\partial C}{\partial o^{L-1}}\frac{\partial o^{L-1}}{\partial a^{L-1}}
=\frac{\partial C}{\partial a^{L}}\frac{\partial a^{L}}{\partial o^{L-1}}\frac{\partial o^{L-1}}{\partial a^{L-1}}
=\epsilon^{i}W^Lf'(a^{L-1})
$$

考虑到矩阵偏导与标量求偏导有所不同（参考[4](https://www.cnblogs.com/pinard/p/10825264.html)），误差可写为：

$$
\epsilon^{i-1}=(\epsilon^{i})^TW\times f'(a^{L-1})
$$
(推导：
设A+1层有n个神经元，A层有m个神经元
$$
\epsilon_i^{A}=\frac{\partial C}{\partial a_i^{A}}
=\frac{\partial C}{\partial o_i^{A}}\frac{\partial o_i^{A}}{\partial a_i^{A}}
=\frac{\partial C}{\partial a^{A+1}}\frac{\partial a^{A+1}}{\partial o_i^{A}}\frac{\partial o_i^{A}}{\partial a_i^{A}}
=\sum_{j=1}^{j=n}{(\frac{\partial C}{\partial a_i^{A+1}}\frac{\partial a_i^{A+1}}{\partial o_i^{A}})}\frac{\partial o_i^{A}}{\partial a_i^{A}}
$$
利用向量形式表示得：
$$
\epsilon_i^{A}=(\epsilon^{A+1})^T(W_i)_{n\times1}\times\frac{\partial o_i^{A}}{\partial a_i^{A}}
$$
则：
$$
\epsilon^{A}=((W)_{n\times m})^T\epsilon^{A+1}\times\frac{\partial o_i^{A}}{\partial a_i^{A}}
$$

)

由此进行迭代，最终可求得训练参数值。
