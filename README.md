## 3DCNN预测有机无机杂化钙钛矿吸附能

### Abstract

我们首先使用MS对1049种有机无机钙钛矿建立了吸附模型并得到了对应的吸附能，再把钙钛矿的吸附模型转换为标准的32x32x32的体数据这样它们就可以被输入到3DCNN当中，之后建立了一个采用Coordinate Attention的3DCNN网络来预测它们的吸附能，Coordinate Attention机制能显著增加3DCNN网络的性能。同时我们还对模型进行了可解释性分析，比较了不同的卤素原子对吸附能影响的大小，结论为F>I>Br>Cl。

###  Installation
我们提供了一个环境文件：``requirements.txt``
```
pip install -r requirements.txt
```
请先在``options.py``中修改自己电脑的文件路径，再运行``train.py``文件即可训练模型

本实验采用了自制数据集，如果想通过同样的方式制作数据集我们把函数放在了``data_pre-processing.py``当中。

### Model Structure
 ![img] (image/png.png)

  
