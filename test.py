import torch
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset, CamelyonDataset,get_data_transforms
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve

import sys
import os
import argparse
import logging
import json
import time
from torch.autograd import Variable
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from preprocess.wsi.data.wsi_producer import GridWSIPatchDataset  # noqa


import warnings
warnings.filterwarnings('ignore')

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)   #Mk(h,w)

        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def cal_anomaly_map_mul(fs_list, ft_list, out_size=224, amap_mode='mul'):
    anomaly_map = np.zeros((len(fs_list[0]),out_size, out_size))
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)   #Mk(h,w)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[:, 0, :, :].to('cpu').detach().numpy()
        anomaly_map += a_map
    return anomaly_map

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def get_threshold(gt,score):
    gt_mask = np.asarray(gt)
    score = np.asarray(score)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score.flatten())
    a = 2*precision*recall
    b = precision + recall
    f1 = np.divide(a,b,out=np.zeros_like(a), where=b !=0)
    threshold = thresholds[np.argmax(f1)]
    return threshold

def evaluation_camelyon16(encoder, bn, decoder, dataloader,device,_class_):
    bn.eval()
    decoder.eval()

    pr_list_sp_mean = []
    pr_list_sp_max = []
    gt_list_sp = []

    with torch.no_grad():
        for img, label, _ in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            if int(label) == 1:
                gt = [1]
            else:
                gt = [0]
              
            gt_list_sp.append(np.max(gt))
            pr_list_sp_mean.append(np.mean(anomaly_map))
            pr_list_sp_max.append(np.max(anomaly_map))

        auroc_sp_mean = round(roc_auc_score(gt_list_sp, pr_list_sp_mean), 3)
        auroc_sp_max = round(roc_auc_score(gt_list_sp, pr_list_sp_max), 3)

        th_auroc_sp_mean = get_threshold(gt_list_sp, pr_list_sp_mean)
        binary_list_sp_mean = np.zeros_like(pr_list_sp_mean, dtype=np.bool)
        binary_list_sp_mean[pr_list_sp_mean>= th_auroc_sp_mean]= 1
        binary_list_sp_mean[pr_list_sp_mean < th_auroc_sp_mean]= 0
        count = 0
        for i in range(len(binary_list_sp_mean)):
            if binary_list_sp_mean[i]  == gt_list_sp[i]:
                count += 1
        assert len(binary_list_sp_mean) == len(gt_list_sp), "Something wrong with test and ground truth pair!"
        accuracy_mean = count/len(binary_list_sp_mean)

        th_auroc_sp_max = get_threshold(gt_list_sp, pr_list_sp_max)
        binary_list_sp_max = np.zeros_like(pr_list_sp_max, dtype=np.bool)
        binary_list_sp_max[pr_list_sp_max>= th_auroc_sp_max]= 1
        binary_list_sp_max[pr_list_sp_max < th_auroc_sp_max]= 0
        count = 0
        for i in range(len(binary_list_sp_max)):
            if binary_list_sp_max[i]  == gt_list_sp[i]:
                count += 1
        assert len(binary_list_sp_max) == len(gt_list_sp), "Something wrong with test and ground truth pair!"
        accuracy_max = count/len(binary_list_sp_max)
       


    return auroc_sp_mean, auroc_sp_max, accuracy_mean, accuracy_max


def evaluation_wsss4luad(encoder, bn, decoder, dataloader,device,_class_):
    bn.eval()
    decoder.eval()
    gt_bm_list_px = []
    pr_bm_list_px = []

    gt_bm_list_sp = []
    pr_bm_list_sp_mean = []

    with torch.no_grad():
        for img, gt, background, label, _ in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            background[background > 0.5] = 1
            background[background <= 0.5] = 0

            gt = gt.cpu().numpy().astype(int)
            background = background.cpu().numpy().astype(int)

            ##background mask
            background_list = []
            for pixel in range(len(anomaly_map.flatten())):
                if background.flatten()[pixel] == 1:
                    background_list.append(pixel)
            anomaly_map_ub = np.delete(anomaly_map.flatten(),background_list)
            gt_ub = np.delete(gt.flatten(),background_list)

            gt_bm_list_px.extend(gt_ub)
            pr_bm_list_px.extend(anomaly_map_ub)

            gt_bm_list_sp.append(np.max(gt_ub))
            pr_bm_list_sp_mean.append(np.mean(anomaly_map_ub))


        th_auroc_px = get_threshold(gt_bm_list_px, pr_bm_list_px)
        binary_list_sp_mean = np.zeros_like(pr_bm_list_px, dtype=np.bool)
        binary_list_sp_mean[pr_bm_list_px>= th_auroc_px]= 1
        binary_list_sp_mean[pr_bm_list_px < th_auroc_px]= 0
        count = 0
        for i in range(len(binary_list_sp_mean)):
            if binary_list_sp_mean[i]  == gt_bm_list_px[i]:
                count += 1
        assert len(binary_list_sp_mean) == len(gt_bm_list_px), "Something wrong with test and ground truth pair!"
        accuracy = count/len(binary_list_sp_mean)
        auroc_px_bm = round(roc_auc_score(gt_bm_list_px, pr_bm_list_px), 3)
        auroc_sp_bm_mean = round(roc_auc_score(gt_bm_list_sp, pr_bm_list_sp_mean), 3)

    return auroc_px_bm, auroc_sp_bm_mean, accuracy

def test(_class_, name,vis = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)


    if _class_ == 'camelyon16':
        test_path = './data/' + _class_ + '/test'
        test_data = CamelyonDataset(root=test_path, transform=data_transform)
    else:
        test_path = './data/' + _class_
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    ckp_path = './checkpoints/' + name

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    if _class_ == 'camelyon16':
        if vis:
            parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                                         ' patch predictions given a WSI')
            parser.add_argument('wsi_path', metavar='WSI_PATH', type=str,
                                help='Path to the input WSI file')
            parser.add_argument('ckpt_path', metavar='CKPT_PATH',
                                type=str,
                                help='Path to the saved ckpt file of a pytorch model')
            parser.add_argument('cfg_path', metavar='CFG_PATH',
                                type=str,
                                help='Path to the config file in json format related to'
                                     ' the ckpt file')
            parser.add_argument('mask_path', metavar='MASK_PATH',
                                type=str,
                                help='Path to the tissue mask of the input WSI file')
            parser.add_argument('probs_map_path', metavar='PROBS_MAP_PATH',
                                type=str, help='Path to the output probs_map numpy file')
            parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                                                                     ', default 0')
            parser.add_argument('--num_workers', default=0, type=int, help='number of '
                                                                           'workers to use to make batch, default 5')
            parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                                                                         ' of the 8 direction predictions for each patch,'
                                                                         ' default 0, which means disabled')
            args = parser.parse_args()
            images = os.listdir(args.wsi_path)
            for image in images:
                image = image.replace('.tif', '')
                print(image)
                wsi_path_simple = args.wsi_path + image + '.tif'
                print(wsi_path_simple)
                mask_path_simple = args.mask_path + image + '.npy'
                print(mask_path_simple)
                probs_map_path_simple = args.probs_map_path + image + '.npy'
                print(probs_map_path_simple)
                vis_WSI(wsi_path_simple, mask_path_simple, probs_map_path_simple, args)
        auroc_sp_mean, auroc_sp_max, accuracy_mean, accuracy_max = evaluation_camelyon16(encoder, bn, decoder, test_dataloader, device,_class_)
        print(_class_,': auroc_sp_mean ',auroc_sp_mean,', auroc_sp_max',auroc_sp_max,', accuracy_mean',accuracy_mean,', accuracy_max',accuracy_max)
    else:
        auroc_px_bm, auroc_sp_bm_mean, accuracy = evaluation_wsss4luad(encoder, bn, decoder, test_dataloader, device,_class_)
        print(_class_, ': auroc_px_bm ', auroc_px_bm, ', auroc_sp_bm_mean', auroc_sp_bm_mean, ', accuracy', accuracy)
        if vis:
            visualization(_class_, name)


def get_probs_map(encoder, bn, decoder, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        data = data.to(device)
        inputs = encoder(data)
        outputs = decoder(bn(inputs))
        anomaly_map= cal_anomaly_map_mul(inputs, outputs, data.shape[-1], amap_mode='a')
        output = []
        for a_map in anomaly_map:
            anomaly_map_filtered = gaussian_filter(a_map, sigma=4)
            output.append(np.mean(anomaly_map_filtered))
        output = torch.Tensor(output)

        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        if len(output.shape) == 1:
            probs = output.cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           :].cpu().data.numpy().flatten()
        print(probs)
        probs_map[x_mask, y_mask] = probs
        count += 1

        time_spent = time.time() - time_now
        time_now = time.time()
        print(logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent)))

    return probs_map


def make_dataloader(wsi_path_simple, mask_path_simple, args, cfg, flip='NONE', rotate='NONE'):
    batch_size = 15 #cfg['batch_size']
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(wsi_path_simple, mask_path_simple,
                            image_size=cfg['image_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def vis_WSI(wsi_path_simple,mask_path_simple,probs_map_path_simple,args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    mask = np.load(mask_path_simple)
    ckp = torch.load(args.ckpt_path)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    bn.eval()
    decoder.eval()

    if not args.eight_avg:
        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(encoder, bn, decoder, dataloader)
    else:
        probs_map = np.zeros(mask.shape)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='NONE', rotate='NONE')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            wsi_path_simple, mask_path_simple, args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        probs_map /= 8

    np.save(probs_map_path_simple, probs_map)

def visualization(_class_, name):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './mvtec/' + _class_
    ckp_path = './checkpoints/'+name
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, background, label, _ in test_dataloader:
            if (label.item() == 0):
                continue
            decoder.eval()
            bn.eval()
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            background = background.cpu().numpy().astype(int).squeeze(0).squeeze(0)


            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map[background >= 0.5] = np.min(ano_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            if not os.path.exists('./result/'+name):
                os.makedirs('./result/'+name)

            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./result/'+name+'/'+str(count)+'_'+'ad.png', ano_map)

            gt = gt.cpu().numpy().astype(int)[0][0]*255
            cv2.imwrite('./result/'+name+'/'+str(count)+'_'+'gt.png', gt)

            count += 1

# python
def mIOU(pred,target,n_classes = 2 ):
    ious = []
    # ignore IOU for background class
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        # target_sum = target_inds.sum()
        intersection = (pred_inds[target_inds]).sum()
        print(intersection)
        union = pred_inds.sum() + target_inds.sum() - intersection
        print(pred_inds.sum(),target_inds.sum())
        if union == 0:
            ious.append(float('nan')) # If there is no ground truth，do not include in evaluation
        else:
            ious.append(float(intersection)/float(max(union,1)))
    return ious