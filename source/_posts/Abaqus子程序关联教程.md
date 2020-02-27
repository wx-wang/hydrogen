---
layout: posts
title: Abaqus子程序关联教程
date: 2020-02-27 19:40:02
tags: [教程, Abaqus]
categories: 教程
---

## 写在开头

本教程搜集于2018年，当时为了装abaqus子程序可谓是煞费苦心。故记录在此，以飨自己。<br>

## 限制条件

目前可**成功安装6.14和2016版本**。其他版本的欢迎尝试。教程参考了：[参考1](https://wenku.baidu.com/view/5db671e74a7302768f99392e.html)，[参考2](https://wenku.baidu.com/view/523f5ae1bdeb19e8b8f67c1cfad6195f302be800.html)<br>

## Start

ABAQUS要是想调用子程序，需要以下几个软件进行关联：**Microsoft Visual Studio 12.0、Intel Visual Fortran Composer XE 2013**。具体安装过程不多说，可百度。需要注意顺序：**先Visual Studio，再Fortran**。<br>

如何关联？只需要**这两行代码：**

```
@call "H:\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" X64
@call "H:\Program Files (x86)\Intel\Composer XE 2013 SP1\bin\ipsxe-comp-vars.bat" intel64 vs2012
// 两行代码对应的位置分别是vcvarsall.bat、ipsxe-comp-vars.bat两个文件的地址
```

对于ABAQUS2016，找到launcher.bat文件进行编辑，将上述两句插入到最开头即可（**bat文件位置示例：**`H:\SIMULIA\CAE\2016\win_b64\resources\install\cae`）。<br>

之后，运行Abaqus Verification，就成功关联。<br>

享受你的科研之旅吧！🙂





