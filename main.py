# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from dataset import get_data_transforms, MVTecDataset, CamelyonDataset
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation_camelyon16, test,evaluation_wsss4luad
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_fucntion(a, b, labels, count, batch_size):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    gama = 2.0
    alpha = 0.1
    ploss = 0
    nloss = 0
    
 
    count = count + (batch_size-sum(labels))
        
    for item in range(len(a)):
        a_2d = a[item].view(a[item].shape[0], -1)
        b_2d = b[item].view(b[item].shape[0], -1)
        loss_all = cos_loss(a_2d, b_2d)
      
        focal_loss = 0
        
        for label in range(len(labels)):              
            if labels[label] == 1:
                pt = loss_all[label]
                alpha_t = alpha
                
                ploss += loss_all[label]
                                
            else:
                pt = 1 - loss_all[label]
                alpha_t = 1-alpha
                
                nloss += loss_all[label] 
                                             
           # if pt <= 0.3:
                #print(pt)
            focal_loss += -alpha_t * (1 - pt).pow(gama) * torch.log(pt)
        focal_loss = focal_loss/len(labels)
        loss += focal_loss
    ploss = ploss/3
    nloss = nloss/3
    return loss, count, ploss, nloss


def train(_class_):
    print(_class_)
    epochs = 10
    learning_rate = 0.000001
    batch_size = 32
    image_size = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)

    train_path = './data/' + _class_ + '/train'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if _class_ == 'camelyon16':
        test_path = './data/' + _class_ + '/test'
        test_data = CamelyonDataset(root=test_path, transform=data_transform)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        val_path = './data/' + _class_ + '/val'
        val_data = ImageFolder(root=val_path, transform=data_transform)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    else:
        test_path = './data/' + _class_
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate,
                                 betas=(0.5, 0.999))
    if _class_ == 'camelyon16':
        for epoch in range(epochs):

            test_nloss = 0
            test_ploss = 0
            count = 0

            bn.train()
            decoder.train()
            loss_list = []
            ckp_path = './checkpoints/' + 'detection_' +_class_ + str(epoch+1)+'.pth'

            for batch_idx, (img, label) in enumerate(train_dataloader):
                img, label = img.to(device), label.to(device)
                inputs = encoder(img)
                outputs = decoder(bn(inputs))  # bn(inputs))
                loss, count, ploss, nloss = loss_fucntion(inputs, outputs, label,count, batch_size)
                test_nloss += nloss
                test_ploss += ploss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            print('ploss',float(test_ploss/(len(train_dataloader.dataset)-count)), 'nloss', float(test_nloss/count))
            print('epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list)))
            if (epoch + 1) >= 5:
                auroc_sp_mean, auroc_sp_max, accuracy_mean, accuracy_max = evaluation_camelyon16(encoder, bn, decoder,
                                                                                                 test_dataloader,
                                                                                                 device, _class_)
                print('Test: ',_class_, ': auroc_sp_mean ', str(auroc_sp_mean), ', auroc_sp_max', str(auroc_sp_max), ', accuracy_mean',
                      str(accuracy_mean), ', accuracy_max', str(accuracy_max))

                auroc_sp_mean, auroc_sp_max, accuracy_mean, accuracy_max = evaluation_camelyon16(encoder, bn, decoder,
                                                                                                 val_dataloader,
                                                                                                 device, _class_)
                print('Val: ',_class_, ': auroc_sp_mean ', str(auroc_sp_mean), ', auroc_sp_max', str(auroc_sp_max), ', accuracy_mean',
                      str(accuracy_mean), ', accuracy_max', str(accuracy_max))

                torch.save({'bn': bn.state_dict(),'decoder': decoder.state_dict()}, ckp_path)
    else:
        for epoch in range(epochs):
            bn.train()
            decoder.train()
            loss_list = []
            ckp_path = './checkpoints/' + 'detection_' + _class_ + str(epoch + 1) + '.pth'
            for batch_idx, (img, label) in enumerate(train_dataloader):
                img, label = img.to(device), label.to(device)
                inputs = encoder(img)
                outputs = decoder(bn(inputs))  # bn(inputs))
                loss,_,_,_ = loss_fucntion(inputs, outputs, label,0,batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            print('epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list)))
            if (epoch + 1) % 5 == 0:
                auroc_px_bm, auroc_sp_bm_mean, accuracy = evaluation_wsss4luad(encoder, bn, decoder, test_dataloader,
                                                                               device, _class_)
                print(_class_, ': auroc_px_bm ', auroc_px_bm, ', auroc_sp_bm_mean', auroc_sp_bm_mean, ', accuracy',
                      accuracy)
                torch.save({'bn': bn.state_dict(),
                            'decoder': decoder.state_dict()}, ckp_path)

if __name__ == '__main__':
    setup_seed(111)
    train('camelyon16')

    path = "checkpoints"
    item_list = os.listdir(path)
    for i in item_list:
        print(i)
        test('camelyon16', i, vis = False)




