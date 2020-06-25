#!/usr/local/python3/bin/python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import cv2

NUM_CLASSES = 10  # CIFAR10データは、10種類のdata
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = './data/'
IMG_SIZE_AND_CHANNEL = 0
TRAIN_NUM_EPOCHS = 50   #50エポック


def main():
    (train_loader, test_loader) = get_train_test_data()
    print("train_loader: ", train_loader)
    print("test_loader: ", test_loader)

    (net, criterion, optimizer) = get_net_criterion_optimizer()

    (train_loss_list, train_acc_list, val_loss_list, val_acc_list) = \
        train(train_loader, test_loader, net, criterion, optimizer)

    output_to_file(train_loss_list, train_acc_list, val_loss_list, val_acc_list)

    # torch.save(net.state_dict(), 'net.ckpt')

    # net2 = MLPNet().to(device)
    # net2.load_state_dict(torch.load('net.ckpt'))

    # net2.eval()
    # with torch.no_grad():
    #     total = 0
    #     test_acc = 0
    #     for images, labels in test_loader:
    #         images, labels = images.view(-1, 32*32*3).to(device), labels.to(device)
    #         outputs = net2(images)
    #         test_acc += (outputs.max(1)[1] == labels).sum().item()
    #         total += labels.size(0)
    #     print('精度: {} %'.format(100 * test_acc / total))


def get_train_test_data():
    # CIFAR10とは、ラベル付けされた5万枚の訓練画像と1万枚のテスト画像のデータセットで
    # 以下で、STEP1) www.cs.toronto.edu から自動的にダウンロード & 解凍します
    #         STEP2) 画像と正解ラベルのペアを返却 が行われます
    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR,
                                                 train=True,
                                                 transform=transforms.ToTensor(),
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR,
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)
    # 試しに1つの訓練用データの内容を見ると、3チャネル , 32x32size だと分かります
    image, label = train_dataset[0]
    # print( image.size() )
    # print(label)
    global IMG_SIZE_AND_CHANNEL
    IMG_SIZE_AND_CHANNEL = image.size()[0] * image.size()[1] * image.size()[2]

    # DataLoader() は、batch_size分だけ、画像と正解ラベルを返します
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=2)
    return train_loader, test_loader
    
# MLP ネットワークの定義
# (多層パーセプトロン, 入力層,隠れ層,出力層が全結合である最も単純なdeep learning)
class MLPNet (nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # 32x32size & 3チャネル, 隠れ層のunit数は、600
        self.conv1 = nn.Conv2d(3,  32, 5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.Conv2d = nn.Conv2d()
        self.conv3 = nn.Conv2d()
            
    def forward(self, x):
        x =nn.MaxPool2d(F.relu(self.conv1(x)),2)
        import pdb;pdb.set_trace()
        x = nn.MaxPool2d(F.relu(self.conv2(x)),2)
        x = nn.MaxPool2d(F.relu(self.conv3(x)),2)
        x = x.view(-1, 28*28*2)

def get_net_criterion_optimizer():
    net = MLPNet().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # 以下のparameterの妥当性は理解していません
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    return net, criterion, optimizer


def train(train_loader, test_loader, net, criterion, optimizer):
    #最後にlossとaccuracyのグラフを出力する為
    train_loss_list = []
    train_acc_list =  []
    val_loss_list =   []
    val_acc_list =    []

    for epoch in range(TRAIN_NUM_EPOCHS):
        #エポックごとに初期化
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
    
        net.train()  #訓練モードへ切り替え


        
        for i, (images, labels) in enumerate(train_loader):  #ミニバッチで分割し読込み
            #viewで縦横32x32 & 3channelのimgを1次元に変換し、toでDEVICEに転送
            """
            images, labels = images.view(-1, IMG_SIZE_AND_CHANNEL).to(DEVICE), labels.to(DEVICE)
            """
            images = images.resize((96, 96))
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()               # 勾配をリセット
            outputs = net(images)               # 順伝播の計算
            loss = criterion(outputs, labels)   #lossの計算
            train_loss += loss.item()           #lossのミニバッチ分を溜め込む
            #accuracyをミニバッチ分を溜め込む
            #正解ラベル（labels）と予測値のtop1（outputs.max(1)）が合っている場合に1が返る
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            #逆伝播の計算
            loss.backward()
            #重みの更新
            optimizer.step()
            #平均lossと平均accuracyを計算
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)
    

        net.eval()   #評価モードへ切り替え
        
        #評価するときに必要のない計算が走らないようtorch.no_gradを使用
        with torch.no_grad():
            for images, labels in test_loader:        
                images, labels = images.view(-1,IMG_SIZE_AND_CHANNEL).to(DEVICE),labels.to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)
    

        # 訓練データのlossと検証データのlossとaccuracyをログ出力.
        print ("Epoch [{}/{}], Loss: {loss:.4f},val_loss: {val_loss:.4f},val_acc: {val_acc:.4f}"
               .format(epoch+1,
                       TRAIN_NUM_EPOCHS,
                       i+1,
                       loss=avg_train_loss,
                       val_loss=avg_val_loss, val_acc=avg_val_acc))

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    return train_loss_list,train_acc_list,val_loss_list,val_acc_list

def output_to_file(train_loss_list,train_acc_list,val_loss_list,val_acc_list):

    plt.figure()
    plt.plot(range(TRAIN_NUM_EPOCHS),
             train_loss_list,
             color='blue',
             linestyle='-',
             label='train_loss')
    plt.plot(range(TRAIN_NUM_EPOCHS),
             val_loss_list,
             color='green',
             linestyle='--',
             label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    plt.savefig( 'test_1.png' )

    plt.figure()
    plt.plot(range(TRAIN_NUM_EPOCHS),
             train_acc_list,
             color='blue',
             linestyle='-',
             label='train_acc')
    plt.plot(range(TRAIN_NUM_EPOCHS),
             val_acc_list,
             color='green',
             linestyle='--',
             label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and validation accuracy')
    plt.grid()
    plt.savefig( 'test_2.png' )

    
if __name__ == '__main__':
    main()

#https://end0tknr.hateblo.jp/entry/20191012/1570880798
