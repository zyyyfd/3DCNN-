


import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from dataset import data_loader
from model import CNN3D
from options import parse_args




def train(args):
    num_epochs=args.num_epochs
    patience=args.patience
    
    model = CNN3D().to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)
    for epoch in range(num_epochs):
        running_loss=0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 统计损失
        running_loss += loss.item()
        scheduler.step(running_loss)
        
        #统计train test mae
        mae_train=0
        for x, y in train_dataloader:
            output = model(x)
            mae_train += F.l1_loss(output, y, reduction='sum').item()
        mae_train /= (len(train_dataloader.dataset))
    
        mae_test=0
        for x, y in test_dataloader:
            output = model(x)
            mae_test += F.l1_loss(output, y, reduction='sum').item()
        mae_test /= (len(test_dataloader.dataset))
    
        # 输出训练结果
        print("Epoch %d, loss: %.4f , train_mae : %.4f , test_mae : %.4f" % (epoch+1, running_loss,mae_train,mae_test))
    
        print('learn_rate : ',optimizer.state_dict()['param_groups'][0]['lr'])
    
        model_name='/root/autodl-tmp/'+'model'+str(epoch)+'.pth'
        torch.save(model.state_dict(),model_name)




def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()

