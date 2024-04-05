from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import argparse
import os
from captum.attr import GradientShap

def single_SHAP(args):
    from captum.attr import GradientShap
    device=args.device
    torch.manual_seed(2)
    batch_size = args.batch_size
    model_path = "models/model24.pth"
    device = torch.device("cuda:0")
    model = MMOE_SHAP().to(device)
    model.load_state_dict(torch.load(model_path))
    gs = GradientShap(model, multiply_by_inputs=True)
    attrs = np.zeros((1, 13, 32, 32, 32))
    baseline = np.zeros((1, 13, 32, 32, 32))
    baseline = torch.from_numpy(baseline)
    baseline = baseline.to(torch.float32)
    baseline.to(device)
    epoch_gs = 60

    cry = np.load("data/npys2/9422.npy")
    cry = torch.tensor(cry, dtype=torch.float32).cuda(0)
    cry = cry.permute(3, 0, 1, 2)
    cry = cry.reshape((1, 13, 32, 32, 32))
    #label2 = np.load("data/labels2/977.npy")
    #label2 = torch.tensor(label2, dtype=torch.int8).cuda(0)
    cry = cry.to(device)
    label1 = label1.to(device)
    cry.requires_grad_()

    for i in range(0,epoch_gs):
        attr = gs.attribute(cry,n_samples=1,baselines=baseline.to(device))
        attr = attr.detach().to("cpu")
        attrs =attrs+ attr.numpy()

    
def image_3d(args):
    data = attrs
    data=data.reshape((13,32,32,32))
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(5, 5), dpi=1200)
    ax = fig.gca(projection="3d")

    cmap = [
        "w",
        "lightgray",
        "g",
        "skyblue",
        "red",
        "lemonchiffon",
        "w",
        "w",
        "w",
        "sandybrown",
        "w",
        "w",
        "black",
    ]

    for channel in range(13):
        x, y, z = np.nonzero(data[channel] > 0)
        z_n=23
        x1=[]
        y1=[]
        z1=[]
        x2=[]
        y2=[]
        z2=[]

        for i in range(0,len(x)):
            if z[i]>z_n:
                x1.append(x[i])
                y1.append(y[i])
                z1.append(z[i])
            else:
                x2.append(x[i])
                y2.append(y[i])
                z2.append(z[i])

        ax.scatter(
            x1,
            y1,
            z1,
            c=cmap[channel],
            alpha=0.4,
            s=data[channel][x1, y1, z1]/60,
            marker="^",
        ) 

        ax.scatter(
            x2,
            y2,
            z2,
            c=cmap[channel],
            alpha=0.6,
            s=data[channel][x2, y2, z2]/70,
        )  

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("1.png", dpi=1200)

    plt.show()

def SHAP_satistic(args)
    batch_size = args.batch_size
    mydataset = MyDataset(args.path_npy2, args.path_label1, args.path_label2)
    gs_dataloader = torch.utils.data.DataLoader(mydataset, batch_size=1, num_workers=0)
    mydataset = MyDataset(args.path_npy2, args.path_label1, args.path_label2)
    model_path = "autodl-tmp/model7.pth"
    device = torch.device("cuda:0")
    model = MMOE().to(device)
    model.load_state_dict(torch.load(model_path))
    gs = GradientShap(model, multiply_by_inputs=True)
    attrs = np.zeros((1, 13, 32, 32, 32))
    baseline = np.zeros((1, 13, 32, 32, 32))
    baseline = torch.from_numpy(baseline)
    baseline = baseline.to(torch.float32)
    baseline.to(device)
    epoch_gs = 1

    list_gs_f = []
    list_gs_cl = []
    list_gs_br = []
    list_gs_i = []

    f_channel = 6
    cl_channel = 7
    br_channel = 8
    i_channel = 9
    for i in range(0, epoch_gs):

        for cry, label1, label2 in gs_dataloader:
            cry = cry.to(device)
            label1 = label1.to(device)

            if (
                (label2 == 0)
                or (label2 == 1)
                or (label2 == 2)
                or (label2 == 3)
                or (label2 == 8)
                or (label2 == 9)
                or (label2 < 0)
            ):
                continue
            # if (label2==7)or(label2==5)or(label2==6):
            # continue
            cry.requires_grad_()
            attr, delta = gs.attribute(
                cry,
                n_samples=1,
                stdevs=0.1,
                return_convergence_delta=True,
                baselines=baseline.to(device),
            )
            attr = attr.detach()
            attr = attr.to("cpu").numpy()

            cry = cry.cpu().detach().numpy()
            if np.sum(cry[:, f_channel, :, :, :]) != 0:
                gs_f = attr[:, f_channel, :, :, :]
                list_gs_f.append(np.sum(gs_f))
            if np.sum(cry[:, cl_channel, :, :, :]) != 0:
                gs_cl = attr[:, cl_channel, :, :, :]
                list_gs_cl.append(np.sum(gs_cl))
            if np.sum(cry[:, br_channel, :, :, :]) != 0:
                gs_br = attr[:, br_channel, :, :, :]
                list_gs_br.append(np.sum(gs_br))
            if np.sum(cry[:, i_channel, :, :, :]) != 0:
                gs_i = attr[:, i_channel, :, :, :]
                list_gs_i.append(np.sum(gs_i))