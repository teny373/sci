# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:38:16 2023

@author: Lenovo
"""

import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms #导入图像预处理模块
from torch.utils.data import DataLoader #匹量读取数据并打乱数据
import torchvision.models as models
import torch.nn as nn #导入神经模块
import torch #导入pytorch主库

#接着检查是否可以使用 GPU 加速，并打印 GPU 名称。
print("是否使用GPU训练：{}".format(torch.cuda.is_available()))    #打印是否采用gpu训练
if torch.cuda.is_available:
    print("GPU名称为：{}".format(torch.cuda.get_device_name()))  #打印相应的gpu信息
#所有图像统一缩放到 640×640。
# 归一化操作将像素值压缩到 [-1, 1]，有助于模型收敛。
# 将 PIL 图像转换为 PyTorch Tensor。
normalize=transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5]) #规范化
transform=transforms.Compose([transforms.Resize((640, 640)),transforms.ToTensor(),normalize]) #数据处理，转为张量
dataset_train=ImageFolder('add_label/label_picture/train',transform=transform)     #训练数据集
# print(dataset_tran[0])
dataset_test=ImageFolder('add_label/label_picture/test',transform=transform)     #验证或测试数据集


# print(dataset_train.classer)#返回类别
print(dataset_train.class_to_idx)                               #返回类别及其索引
# print(dataset_train.imgs)#返回图片路径
print(dataset_test.class_to_idx)

train_data_size=len(dataset_train)                              #放回数据集长度
test_data_size=len(dataset_test)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#每批加载16张图像，打乱顺序。
#设置 drop_last=True，确保每个 batch 大小一致。
dataloader_train=DataLoader(dataset_train,batch_size=16,shuffle=True,num_workers=0,drop_last=True)
dataloader_test=DataLoader(dataset_test,batch_size=16,shuffle=True,num_workers=0,drop_last=True)

#模型加载
model_ft=models.resnet50(pretrained=True)#使用迁移学习，加载预训练权重
# print(model_ft)

in_features=model_ft.fc.in_features
model_ft.fc=nn.Sequential(nn.Linear(in_features,36),nn.Linear(36,4))#将最后的全连接改为（36，6），使输出为六个小数，对应四个树种的置信度


model_ft=model_ft.cuda()#将模型迁移到gpu

#pytorch2.0编译模型部分
# model_ft=torch.compile(model_ft)

#使用交叉熵损失函数，适合分类任务。
#使用带动量的 SGD 进行权重更新。
loss_fn=nn.CrossEntropyLoss()

loss_fn=loss_fn.cuda()  #将loss迁移到gpu
learn_rate=0.0001       #设置学习率
optimizer=torch.optim.SGD(model_ft.parameters(),lr=learn_rate,momentum=0.001)#可调超参数


total_train_step=0
total_test_step=0


epoch=100              #迭代次数
writer=SummaryWriter("logs")
best_acc=-1
ss_time=time.time()

#迭代所有训练数据：前向传播 → 反向传播 → 权重更新。
for i in range(epoch):
    start_time = time.time()
    print("--------第{}轮训练开始---------".format(i+1))
    model_ft.train()
    for data in dataloader_train:
        imgs,targets=data
        imgs=imgs.cuda()
        targets=targets.cuda()
        outputs=model_ft(imgs)
        loss=loss_fn(outputs,targets)

        optimizer.zero_grad()   #梯度归零
        loss.backward()         #反向传播计算梯度
        optimizer.step()        #梯度优化

        total_train_step=total_train_step+1
        if total_train_step%100==0:#一轮时间过长可以考虑加一个
            end_time=time.time()
            print("使用GPU训练100次的时间为：{}".format(end_time-start_time))
            print("训练次数：{},loss:{}".format(total_train_step,loss.item()))

    model_ft.eval()
    
    total_test_loss=0
    total_test_accuracy=0
    
    total_train_loss=0
    total_train_accuracy=0


    with torch.no_grad():       #验证数据集时禁止反向传播优化权重
      #记录训练集精度和损失值
        for data in dataloader_train:
            imgs,targets=data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs=model_ft(imgs)
            train_loss=loss_fn(outputs,targets)
            total_train_loss=total_train_loss+train_loss.item()
            train_accuracy=(outputs.argmax(1)==targets).sum()
            total_train_accuracy=total_train_accuracy+train_accuracy
        print("整体训练集上的loss：{}(越小越好,与上面的loss无关此为测试集的总loss)".format(total_train_loss))
        print("整体训练集上的正确率：{}(越大越好)".format(total_train_accuracy / len(dataset_train)))
       
        for data in dataloader_test:
            imgs,targets=data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs=model_ft(imgs)
            test_loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+test_loss.item()
            test_accuracy=(outputs.argmax(1)==targets).sum()
            total_test_accuracy=total_test_accuracy+test_accuracy
        print("整体测试集上的loss：{}(越小越好,与上面的loss无关此为测试集的总loss)".format(total_test_loss))
        print("整体测试集上的正确率：{}(越大越好)".format(total_test_accuracy / len(dataset_test)))


        # 记录到 TensorBoard
        writer.add_scalar("Train Accuracy", total_train_accuracy / len(dataset_train), i)
        writer.add_scalar("Train Loss", total_train_loss, i)
        writer.add_scalar("Test Accuracy", total_test_accuracy/len(dataset_test), i)
        writer.add_scalar("Test Loss", total_test_loss, i)


        total_test_step = total_test_step + 1
      #若当前验证集准确率超过历史最好，保存模型权重。
      #确保最终部署的是性能最好的模型。
        if total_test_accuracy > best_acc:   #保存迭代次数中最好的模型
            print("已修改模型")
            best_acc = total_test_accuracy
            torch.save(model_ft.state_dict(), "best_model.pth")      #只保留权重的参数即可 
# # 绘制训练和测试精度和损失曲线
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(total_train_loss, label='Train Loss')
# plt.plot(total_test_loss, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()


# plt.subplot(1, 2, 2)
# plt.plot(total_train_accuracy / len(dataset_train), label='Train Accuracy')
# plt.plot(total_test_accuracy/len(dataset_test), label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.show()
ee_time=time.time()
zong_time=ee_time-ss_time
print("训练总共用时:{}h:{}m:{}s".format(int(zong_time//3600),int((zong_time%3600)//60),int(zong_time%60))) #打印训练总耗时
writer.close()