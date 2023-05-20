
from captum.attr import GuidedBackprop
import torch
from model import CNN3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def element_guided(args):
    
    model_path=args.model_path
    path_npy2=args.path_npy2
    color_channel=args.color_channel

    guided= 0
    device = torch.device("cuda:0")
    model = CNN3D().to(device)
    model.load_state_dict(torch.load(model_path))
    model.train()
    input_list=os.listdir(path_npy2)
    for i in range(0,len(input_list)):
        input_path=path_npy2+'/'+input_list[i]
        x=np.load(input_path)
        x=torch.tensor(x,dtype=torch.float32).cuda(0)
        x=x.permute(3, 0, 1, 2)
        x=torch.unsqueeze(x, 0)

        input_tensor = torch.zeros_like(x)
        input_tensor[:, color_channel, :, :, :]=x[:, color_channel, :, :, :]
        x.requires_grad_()

        guided_backprop = GuidedBackprop(model)
        attributions = guided_backprop.attribute(inputs=x)
        attributions = attributions.squeeze().cpu().detach()
    
        abs_attributions = torch.abs(attributions)
        abs_attributions=torch.sum(abs_attributions)
    guided=guided+abs_attributions
    
    return guided


def crystal_visualization(args):
    data =np.load(args.crystal_visualization)

    # 调整形状以适应 Matplotlib 的绘图需求
    data = np.transpose(data, (3, 0, 1, 2))
    # 创建 3D 图形对象
    fig = plt.figure(dpi=1000)
    ax = fig.gca(projection='3d')

    cmaps = ['Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'YlOrBr', 'BuPu',
         'YlGnBu', 'RdPu', 'Greys', 'YlOrRd', 'OrRd', 'PuRd']

    # 绘制每个通道的 3D 图片
    for channel in range(13):
        x, y, z = np.nonzero(data[channel] >0)
        ax.scatter(x, y, z, c=data[channel][x, y, z], cmap=cmaps[channel], alpha=0.6,vmin=0.0, vmax=5.0,s = 10)  # 使用颜色表示梯度值

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()

