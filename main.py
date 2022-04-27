#!/usr/bin/env python
# coding: utf-8



import argparse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
#from xception import xception
from ESPNetV2 import EESPNet
import pickle
import numpy as np
from pytorchtool import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,recall_score,precision_score

from torch.nn.modules.loss import _Loss
import json

class MyDataset(TensorDataset): 
    def __init__(self, data, labels): 
        self.data= data
        self.labels = labels  
    
    def __getitem__(self, index): 
        img, target = self.data[index], self.labels[index] 

        return img, target
    def __len__(self):
        return len(self.data)
class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
def get_cls_num_list(y):
    cls_num_list = []
    cls_num_list.append((y == 0).sum())
    cls_num_list.append((y == 1).sum())
    return cls_num_list
def DRW(cls_num_list):
    idx = EPOCH // 160
    betas = [0, 0.9999]
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights)

def training(epoch):
    net.train()
    for batch_index,(images,labels) in enumerate(train_loader):
        if use_cuda:
            images, labels=images.cuda(),labels.cuda()
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = net(images)
        loss=criterion(outputs,labels.long())
        correct_1=0
        _,pred=outputs.topk(1,1,largest=True,sorted=True)
        labels=labels.view(labels.size(0),-1).expand_as(pred)
        correct=pred.eq(labels).float()
        correct_1 += correct[:, :1].sum()
        loss.backward()
        optimizer.step()
        print("Training Epoch:{epoch}[{trained_samples}/{total_samples}]\tLoss:{:0.4f}\ttop-1 accuracy:{:0.4f}".format(
            loss.item(),
            (100.*correct_1)/len(outputs),
            epoch=epoch,
            trained_samples=batch_index * 16 + len(images),
            total_samples=len(train_loader.dataset)
        ))
        #scheduler.step()

def eval_training(epoch):
    net.eval()
    val_loss = 0.0 # cost function error
    correct = 0.0
    y_pred = []
    y_true = []
    for (images, labels) in val_loader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        loss = criterion(outputs, labels.long())
        val_loss += loss.item()
        _, preds = outputs.max(1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        correct += preds.eq(labels).sum()
    print('Validate set: Average loss: {:.4f}, Accuracy: {:.4f},Recall:{:.4f}, Precision:{:.4f}, F1-score:{:.4f}'.format(
        val_loss / len(val_loader.dataset),
        (100.*correct.float())/ len(val_loader.dataset),
        100*recall_score(y_true, y_pred, average='binary'),
        100*precision_score(y_true, y_pred, average='binary'),
        100*f1_score(y_true, y_pred, average='binary')
    ))
    acc=(100.*correct.float())/ len(val_loader.dataset)
    return acc,val_loss
def testing():
    model = torch.load("torchmodel.pth")
    test_loss = 0.0 # cost function error
    correct = 0.0
    y_pred = []
    y_true = []
    for (images, labels) in test_loader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        with torch.no_grad():
            images, labels = Variable(images), Variable(labels)
        outputs = model(images)
        loss = criterion(outputs, labels.long())
        test_loss += loss.item()
        _, preds = outputs.max(1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        correct += preds.eq(labels).sum()
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f},Recall:{:.4f}, Precision:{:.4f}, F1-score:{:.4f}'.format(
        val_loss / len(val_loader.dataset),
        (100.*correct.float())/ len(val_loader.dataset),
        100*recall_score(y_true, y_pred, average='binary'),
        100*precision_score(y_true, y_pred, average='binary'),
        100*f1_score(y_true, y_pred, average='binary')
    ))
    acc=(100.*correct.float())/ len(test_loader.dataset)


import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
class Dataset(TensorDataset): 
    def __init__(self, data): 
        self.data= data
    
    def __getitem__(self, index): 
        img = self.data[index] 

        return img
    def __len__(self):
        return len(self.data)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 500
batch_size = 16
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
file1=open("DTW2-pitch.pkl","rb")
data1=pickle.load(file1)
X=data1[0].astype(np.float64)
Y=data1[1].astype(np.float64)
file2=open("DTW2-rhythm.pkl","rb")
data2 = pickle.load(file2)
X = np.expand_dims(X, axis = 1)
X = np.append(X, np.expand_dims(data2[0].astype(np.float64),axis = 1),axis = 1)
X=torch.tensor(X,dtype=torch.float32)
dataset = Dataset(X)
print("=================")
print(dataset.data.shape)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2,padding = 1),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 5, stride=2, padding=2),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1,padding = 1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2,padding = 1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=2),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 2, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x_out = self.decoder(x)
        return x,x_out


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    total_loss = 0
    for data in dataloader:
        img= data
        img = Variable(img).cuda()
        # ===================forward=====================
        _,output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, total_loss))
    #if epoch % 10 == 0:
        #pic = to_img(output.cpu().data)
        #save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')
#ESPNet
X = []
for data in dataloader:
        img= data
        img = Variable(img).cuda()
        x,output = model(img)
        x= x.cpu().detach().numpy()
        X.extend(x)
X = np.array(X)
x_train,x_val_test, y_train, y_val_test = train_test_split(X,Y,test_size=0.3)
x_test,x_val, y_test, y_val = train_test_split(x_val_test,y_val_test,test_size=0.5)
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
x_train = torch.unsqueeze(torch.from_numpy(x_train),dim=1)
x_train = torch.tensor(x_train,dtype=torch.float32)
x_train = torch.squeeze(x_train)
y_train = torch.from_numpy(y_train)
y_train = torch.tensor(y_train,dtype=torch.float32)
x_test = torch.unsqueeze(torch.from_numpy(x_test),dim=1)
x_test = torch.tensor(x_test,dtype=torch.float32)
x_test = torch.squeeze(x_test)
y_test = torch.from_numpy(y_test)
y_test = torch.tensor(y_test,dtype=torch.float32)
x_val = torch.unsqueeze(torch.from_numpy(x_val),dim=1)
x_val = torch.tensor(x_val,dtype=torch.float32)
x_val = torch.squeeze(x_val)
y_val = torch.from_numpy(y_val)
y_val = torch.tensor(y_val,dtype=torch.float32)
EPOCH = 100
acc_all=0
cls_num_list = get_cls_num_list(y_train)
per_cls_weights = DRW(cls_num_list)
traindataset = MyDataset(x_train,y_train)
testdataset = MyDataset(x_test,y_test)
valdataset = MyDataset(x_val,y_val)
train_loader=DataLoader(traindataset,batch_size=16,shuffle=True)
test_loader=DataLoader(testdataset,batch_size=16,shuffle=True)
val_loader=DataLoader(valdataset,batch_size=16,shuffle=True)
#net = xception()
net=EESPNet()
net = net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-8)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
#criterion = nn.CrossEntropyLoss()
criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)
#patience = 20 # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
#early_stopping = EarlyStopping(patience, verbose=True)
acc_list = []
loss_list = []
for epoch in range(1, EPOCH+1):
    training(epoch)
    acc,val_loss = eval_training(epoch)
    acc = acc.cpu().numpy()
    acc_list.append(acc)
    loss_list.append(val_loss)
    #early_stopping(val_loss, net)
    # 若满足 early stopping 要求
    #if early_stopping.early_stop:
        #print("Early stopping")
        # 结束模型训练
        #break
x1 = range(0, EPOCH)
x2 = range(0, EPOCH)
y1 = acc_list
y2 = loss_list
plt.subplot(2, 1, 1)
plt.plot(y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
plt.savefig(f"accuracy_loss.jpg")
torch.save(net,"torchmodel.pth")
testing()





