import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
import os
import argparse
import warnings

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, c5, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.p1_1 = nn.Conv3d(in_channels, c1, stride=1, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(c1)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.p2_1 = nn.Conv3d(in_channels, c2[0], stride=1, kernel_size=1)
        self.p2_2 = nn.Conv3d(c2[0], c2[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(c2[1])
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.p3_1 = nn.Conv3d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv3d(c3[0], c3[1], kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm3d(c3[1])
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.p4_1 = nn.Conv3d(in_channels, c4[0], kernel_size=1)
        self.p4_2 = nn.Conv3d(c4[0], c4[1], kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm3d(c4[1])
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.p5_1 = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        self.p5_2 = nn.Conv3d(in_channels, c5, kernel_size=1)
        self.bn5 = nn.BatchNorm3d(c5)
        self.relu5 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        p1 = self.p1_1(x)
        p1 = self.bn1(p1)
        p1 = self.relu1(p1)

        p2 = self.p2_1(x)
        p2 = self.p2_2(p2)
        p2 = self.bn2(p2)
        p2 = self.relu2(p2)

        p3 = self.p3_1(x)
        p3 = self.p3_2(p3)
        p3 = self.bn3(p3)
        p3 = self.relu3(p3)

        p4 = self.p4_1(x)
        p4 = self.p4_2(p4)
        p4 = self.bn4(p4)
        p4 = self.relu4(p4)

        p5 = self.p5_1(x)
        p5 = self.p5_2(p5)
        p5 = self.bn5(p5)
        p5 = self.relu5(p5)

        p = torch.cat((p1, p2), dim=1)
        p = torch.cat((p, p3), dim=1)
        p = torch.cat((p, p4), dim=1)
        p = torch.cat((p, p5), dim=1)

        return p

class CABlock(nn.Module):
    def __init__(self, channel, h, w, l, reduction=16):
        super(CABlock, self).__init__()

        self.h = h
        self.w = w
        self.l = l

        self.avg_pool_x = nn.AdaptiveAvgPool3d((h, 1, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool3d((1, w, 1))
        self.avg_pool_l = nn.AdaptiveAvgPool3d((1, 1, l))

        self.conv_1x1x1 = nn.Conv3d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm3d(channel // reduction)
        self.F_h = nn.Conv3d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.F_w = nn.Conv3d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.F_l = nn.Conv3d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid_l = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 4, 3, 2)

        x_w = self.avg_pool_y(x).permute(0, 1, 4, 2, 3)

        x_l = self.avg_pool_l(x)

        x_cat_conv_relu = self.relu(self.conv_1x1x1(torch.cat((x_h, x_w, x_l), 4)))

        x_cat_conv_split_h, x_cat_conv_split_w, x_cat_conv_split_l = (
            x_cat_conv_relu.split([self.h, self.w, self.l], 4)
        )  

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 4, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w.permute(0, 1, 2, 4, 3)))
        s_l = self.sigmoid_l(self.F_l(x_cat_conv_split_l))

        out = x * s_h.expand_as(x) * s_w.expand_as(x) * s_l.expand_as(x)
        return out

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv3d(13, 256, kernel_size=2, stride=2),
            CABlock(256, 16, 16, 16),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

        self.b2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=2, stride=2),
            CABlock(256, 4, 4, 4),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

        self.b3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        return x

class MMOE(nn.Module):
    def __init__(self):
        super(MMOE, self).__init__()

        for i in range(10):
            setattr(self, "expert_layer" + str(i + 1), Expert())
        self.expert_layers = [
            getattr(self, "expert_layer" + str(i + 1)) for i in range(10)
        ]

        self.gate1_1 = nn.Sequential(
            nn.Conv3d(13, 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 8, 10),
            nn.LeakyReLU(inplace=True),
        )

        self.gate1_2 = nn.Sequential(
            nn.Linear(20, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 10),
            nn.Softmax(dim=1),
        )

        self.gate2 = nn.Sequential(
            nn.Conv3d(13, 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 8, 10),
            nn.Softmax(dim=1),
        )

        self.towers2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 10),
            nn.LeakyReLU(inplace=True),
        )

        self.towers1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        gate2 = self.gate2(x)
        E_nets = [expert(x) for expert in self.expert_layers]
        gate2 = gate2.view(10)
        for i in range(0, 10):
            E_net2 = E_nets[i]
            E_net2 = E_net2.view(64)
            E_net2 = gate2[i] * E_net2
            E_net2 = E_net2.unsqueeze(0)
            if i == 0:
                towers2 = E_net2
            if i >= 1:
                towers2 = torch.cat([towers2, E_net2], dim=1)

        out2 = self.towers2(towers2)

        gate1_1 = self.gate1_1(x)
        gate1_1 = torch.cat([gate1_1, out2], dim=1)
        gate1_1 = gate1_1.unsqueeze(0)
        gate1 = self.gate1_2(gate1_1)
        gate1 = gate1.view(10)
        for i in range(0, 10):
            E_net1 = E_nets[i]
            E_net1 = E_net1.view(64)
            E_net1 = gate1[i] * E_net1
            E_net1 = E_net1.unsqueeze(0)
            if i == 0:
                towers1 = E_net1
            if i >= 1:
                towers1 = torch.cat([towers1, E_net1], dim=1)

        out1 = self.towers1(towers1)

        return out1, out2
    
class MMOE_SHAP(nn.Module):
    def __init__(self):
        super(MMOE_SHAP, self).__init__()

        for i in range(10):
            setattr(self, "expert_layer" + str(i + 1), Expert())
        self.expert_layers = [
            getattr(self, "expert_layer" + str(i + 1)) for i in range(10)
        ]

        self.gate1_1 = nn.Sequential(
            nn.Conv3d(13, 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 8, 10),
            nn.LeakyReLU(inplace=True),
        )

        self.gate1_2 = nn.Sequential(
            nn.Linear(20, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 10),
            nn.Softmax(dim=1),
        )

        self.gate2 = nn.Sequential(
            nn.Conv3d(13, 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 8, 10),
            nn.Softmax(dim=1),
        )

        self.towers2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 10),
            nn.LeakyReLU(inplace=True),
        )

        self.towers1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        gate2 = self.gate2(x)
        E_nets = [expert(x) for expert in self.expert_layers]
        gate2 = gate2.view(10)
        for i in range(0, 10):
            E_net2 = E_nets[i]
            E_net2 = E_net2.view(64)
            E_net2 = gate2[i] * E_net2
            E_net2 = E_net2.unsqueeze(0)
            if i == 0:
                towers2 = E_net2
            if i >= 1:
                towers2 = torch.cat([towers2, E_net2], dim=1)

        out2 = self.towers2(towers2)

        gate1_1 = self.gate1_1(x)
        gate1_1 = torch.cat([gate1_1, out2], dim=1)
        gate1_1 = gate1_1.unsqueeze(0)
        gate1 = self.gate1_2(gate1_1)
        gate1 = gate1.view(10)
        for i in range(0, 10):
            E_net1 = E_nets[i]
            E_net1 = E_net1.view(64)
            E_net1 = gate1[i] * E_net1
            E_net1 = E_net1.unsqueeze(0)
            if i == 0:
                towers1 = E_net1
            if i >= 1:
                towers1 = torch.cat([towers1, E_net1], dim=1)

        out1 = self.towers1(towers1)

        return out1


class CCMMOE_Dataset(Dataset):
    def __init__(self,args, cry_root_dir, label1_root_dir, label2_root_dir):
        self.args=args
        self.cry_root_dir = cry_root_dir
        self.cry_path_name = os.listdir(self.cry_root_dir)

        self.cry_path_name = sorted(
            self.cry_path_name, key=lambda x: int(x.split(".")[0])
        )
        for i in range(0, len(self.cry_path_name)):
            self.cry_path_name[i] = self.cry_root_dir + "/" + self.cry_path_name[i]

        self.label1_root_dir = label1_root_dir
        self.label1_path_name = os.listdir(self.label1_root_dir)
        # self.label1_path_name.remove('.ipynb_checkpoints')
        self.label1_path_name = sorted(
            self.label1_path_name, key=lambda x: int(x.split(".")[0])
        )
        for i in range(0, len(self.label1_path_name)):
            self.label1_path_name[i] = (
                self.label1_root_dir + "/" + self.label1_path_name[i]
            )

        self.label2_root_dir = label2_root_dir
        self.label2_path_name = os.listdir(self.label2_root_dir)
       
        self.label2_path_name = sorted(
            self.label2_path_name, key=lambda x: int(x.split(".")[0])
        )
        for i in range(0, len(self.label2_path_name)):
            self.label2_path_name[i] = (
                self.label2_root_dir + "/" + self.label2_path_name[i]
            )

    def __getitem__(self, idx):
        cry_path = self.cry_path_name[idx]
        cry = np.load(cry_path)
        cry = torch.tensor(cry, dtype=torch.float32).to(self.args.device)
        cry = cry.permute(3, 0, 1, 2)

        label1_path = self.label1_path_name[idx]
        label1 = np.load(label1_path)
        label1 = torch.tensor(label1, dtype=torch.float32).to(self.args.device)

        label2_path = self.label2_path_name[idx]
        label2 = np.load(label2_path)
        label2 = torch.tensor(label2, dtype=torch.int8).to(self.args.device)

        return cry, label1, label2

    def __len__(self):
        return len(self.cry_path_name)

def CCMMOE_train(args):
    
    num_epochs = args.epochs
    lr = args.lr
    device = torch.device(args.device)
    model = MMOE().to(device)
    model_path = "models/model24.pth"
    model.load_state_dict(torch.load(model_path))
    model.train()
    criterion1 = nn.L1Loss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, target1, target2) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            target1 = target1.to(device)
            target2 = target2.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            output1 = outputs[0]
            output2 = outputs[1]
            output1=output1.view(1)
            loss1 = criterion1(output1, target1)
            loss2 = criterion2(output2, target2.long())
            loss = loss1 + loss2
            loss.requires_grad_()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        test_mae = evaluate1(args,model, test_dataloader)
        train_mae = evaluate1(args,model, train_dataloader)
        test_accuracy = evaluate2(args,model, test_dataloader)
        train_accuracy = evaluate2(args,model, train_dataloader)

        print(
            "Epoch %d, loss: %.4f,测试集MAE : %.4f,训练集MAE : %.4f"
            % (epoch + 1, running_loss, test_mae, train_mae)
        )
        print(
            "Epoch %d, loss: %.4f,测试集分类准确率 : %.4f,训练集分类准确率 : %.4f"
            % (epoch + 1, running_loss, test_accuracy, train_accuracy)
        )
        model_name = args.model_save_path+"/model"+ str(epoch) + ".pth"
        torch.save(model.state_dict(), model_name)
        
        with open('output1 1.25.txt', 'a') as f:
            print(
            "Epoch %d, loss: %.4f,测试集MAE : %.4f,训练集MAE : %.4f"
            % (epoch + 1, running_loss, test_mae, train_mae), file=f)
            print(
            "Epoch %d, loss: %.4f,测试集分类准确率 : %.4f,训练集分类准确率 : %.4f"
            % (epoch + 1, running_loss, test_accuracy, train_accuracy), file=f)

def train_loader(args):
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    global train_dataloader
    mydataset = CCMMOE_Dataset(args,args.path_npy2, args.path_label1, args.path_label2)
    train_size = int(args.lengths * len(mydataset))
    test_size = len(mydataset) - train_size
    train_dataset, test_dataset = random_split(mydataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return train_dataloader

def test_loader(args):
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    global test_dataloader
    mydataset = CCMMOE_Dataset(args,args.path_npy2, args.path_label1, args.path_label2)
    train_size = int(args.lengths * len(mydataset))
    test_size = len(mydataset) - train_size
    train_dataset, test_dataset = random_split(mydataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,shuffle=True, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return test_dataloader

def evaluate1(args,model, dataloader):
    mae = 0
    total = 0
    device = torch.device(args.device)
    with torch.no_grad():  
        for inputs, target1, target2 in dataloader:
            inputs = inputs.to(device)
            target1 = target1.to(device)
            outputs = model(inputs)
            output1 = outputs[0]
            for i in range(0, len(target1)):
                mae = abs(output1[i] - target1[i]) + mae
            total += target1.size(0)
        MAE = mae / total
    return MAE

def evaluate2(args,model, dataloader):
    correct = 0
    total = 0
    device = torch.device(args.device)
    with torch.no_grad(): 
        for inputs, target1, target2 in dataloader:
            inputs = inputs.to(device)
            target2 = target2.to(device)
            outputs = model(inputs)
            output2 = outputs[1]
            _, predicted = torch.max(output2.data, 1) 

            total += target2.size(0)
            correct += (predicted == target2).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def DNN_train(args):
    
    num_epochs = args.epochs
    lr = args.lr
    device = torch.device(args.device)
    model = DNN().to(device)
    model_path = "models/DNN_model23.pth"
    model.load_state_dict(torch.load(model_path))
    model.train()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, target) in enumerate(DNN_train_dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.requires_grad_()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        test_mae = DNN_evaluate(args,model, DNN_test_dataloader)
        train_mae = DNN_evaluate(args,model, DNN_train_dataloader)
        
        print(
            "Epoch %d, loss: %.4f,测试集MAE : %.4f,训练集MAE : %.4f"
            % (epoch + 1, running_loss, test_mae, train_mae)
        )

        model_name = args.model_save_path+"/DNN_model"+ str(epoch) + ".pth"
        torch.save(model.state_dict(), model_name)


class DNN_Dataset(Dataset):
    def __init__(self,args):
        
        self.args=args
        self.path_DNN_features=args.path_DNN_features
        self.DNN_features=pd.read_csv(self.path_DNN_features)

    def __getitem__(self, idx):
        
        features=self.DNN_features.iloc[idx]
        label=features.iloc[0]
        feature= features.iloc[1:]

        label=torch.tensor(label.astype(np.float32)).to(self.args.device)
        feature=torch.tensor(feature.astype(np.float32)).to(self.args.device)
        return feature,label

    def __len__(self):
        return len(self.DNN_features)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(46, 2048)
        self.r1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.r2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(1024, 512)
        self.r3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(512, 128)
        self.r4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(128, 12)
        self.r5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(12, 1)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        x = self.r3(x)
        x = self.fc4(x)
        x = self.r4(x)
        x = self.fc5(x)
        x = self.r5(x)
        x = self.fc6(x)
        x = x.view(1)
        return x
    
def DNN_evaluate(args,model,dataloader):
    mae = 0
    total = 0
    device = torch.device(args.device)
    with torch.no_grad():  
        for inputs, target in dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            mae = abs(output - target) + mae
            total = total+1
        MAE = mae / total
    return MAE

def DNN_train_loader(args):
    global DNN_train_dataloader
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    mydataset = DNN_Dataset(args)
    train_size = int(args.lengths * len(mydataset))
    test_size = len(mydataset) - train_size
    train_dataset, test_dataset = random_split(mydataset, [train_size, test_size])
    DNN_train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,shuffle=True, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return DNN_train_dataloader

def DNN_test_loader(args):
    global DNN_test_dataloader
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    mydataset = DNN_Dataset(args)
    train_size = int(args.lengths * len(mydataset))
    test_size = len(mydataset) - train_size
    train_dataset, test_dataset = random_split(mydataset, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,shuffle=True, num_workers=0
    )
    DNN_test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return DNN_test_dataloader

def DNN_shap_loader(args):
    global DNN_shap_dataloader
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    mydataset = DNN_Dataset(args)
    DNN_shap_dataloader = torch.utils.data.DataLoader(
        mydataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return DNN_shap_dataloader

class Expert_test(nn.Module):
    def __init__(self):
        super(Expert_test, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv3d(13, 256, kernel_size=2, stride=2),
            CABlock(256, 16, 16, 16),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

        self.b2 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=2, stride=2),
            CABlock(256, 4, 4, 4),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

        self.b3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(inplace=True),
        )
        
        self.b4 = nn.Sequential(
            nn.Linear(64, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 1) ,   
        )
         
    def forward(self, x):

        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        
        return x

    
def Expert_train(args):
    
    num_epochs = args.epochs
    lr = args.lr
    device = torch.device(args.device)
    model = Expert_test().to(device)
    #model_path = "models/model24.pth"
    #model.load_state_dict(torch.load(model_path))
    model.train()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, target,target0) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target)
            loss.requires_grad_()
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        test_mae = evaluate1(args,model, test_dataloader)
        train_mae = evaluate1(args,model, train_dataloader)
    
        print(
            "Epoch %d, loss: %.4f,测试集MAE : %.4f,训练集MAE : %.4f"
            % (epoch + 1, running_loss, test_mae, train_mae)
        )
        model_name = args.model_save_path+"/expertmodel"+ str(epoch) + ".pth"
        torch.save(model.state_dict(), model_name) 
        with open('outputexpert.txt', 'a') as f:
            print(
            "Epoch %d, loss: %.4f,测试集MAE : %.4f,训练集MAE : %.4f"
            % (epoch + 1, running_loss, test_mae, train_mae), file=f)