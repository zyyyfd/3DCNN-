# 3DCNN预测有机无机杂化钙钛矿吸附能

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

 ![img](https://img-blog.csdnimg.cn/6544577d7203483394296cc7cd3612e6.jpeg)
 
### Data processing

首先我们将ms计算出来的xsd文件转为了cif文件，再对cif文件做了点云体素化处理，并把其中的12中原子做了独热编码，将cif文件转换为了13通道的体数据，最后将数据的尺寸统一缩放为13*32*32*32大小

cif点云体素化之后的数据图
![img](https://img-blog.csdnimg.cn/aa98086f52a544a68d625a870c7dfb2e.jpeg)

尺寸缩放完毕的数据图
![img](https://img-blog.csdnimg.cn/326e123a1ad747dfb843dbadfb572fc1.jpeg)

### Experimental result

对所有转换过的晶体结构数据的F，Cl，Br，I做了GuidedBackpropg，求出了它们的mean_attributions。
F:5.9999e-10    Cl:2.5186e-10    Br:6.0558e-10   I:7.6454e-10

同时还求出了模型对数据每一个像素的attributions,可以反应出模型更关注数据的哪一部分。
下面是它的热力图
![img](https://img-blog.csdnimg.cn/e0ad285bcac74a3d934a9343a6c81690.jpeg)
 
 

  
