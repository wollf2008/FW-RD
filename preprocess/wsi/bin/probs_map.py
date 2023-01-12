import sys
import os
import argparse
import logging
import json
import time


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from wsi.data.wsi_producer import GridWSIPatchDataset  # noqa
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                 ' patch predictions given a WSI')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('ckpt_path', default=None, metavar='CKPT_PATH', type=str,
                    help='Path to the saved ckpt file of a pytorch model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help='Path to the config file in json format related to'
                    ' the ckpt file')
parser.add_argument('mask_path', default=None, metavar='MASK_PATH', type=str,
                    help='Path to the tissue mask of the input WSI file')
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                    ', default 0')
parser.add_argument('--num_workers', default=0, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--eight_avg', default=0, type=int, help='if using average'
                    ' of the 8 direction predictions for each patch,'
                    ' default 0, which means disabled')

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

def get_probs_map(encoder, bn, decoder, dataloader):
    probs_map = np.zeros(dataloader.dataset._mask.shape)
    num_batch = len(dataloader)

    count = 0
    time_now = time.time()
    for (data, x_mask, y_mask) in dataloader:
        data = Variable(data.cuda(non_blocking=True), volatile=True)
        input = encoder(data)
        output = decoder(bn(input))
        anomaly_map, _ = cal_anomaly_map(input, output, data.shape[-1], amap_mode='a')
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        output = torch.Tensor([np.mean(anomaly_map)])
        torch.unsqueeze(output, 0)
        # because of torch.squeeze at the end of forward in resnet.py, if the
        # len of dim_0 (batch_size) of data is 1, then output removes this dim.
        # should be fixed in resnet.py by specifying torch.squeeze(dim=2) later
        if len(output.shape) == 1:
            probs = output.sigmoid().cpu().data.numpy().flatten()
        else:
            probs = output[:,
                           :].sigmoid().cpu().data.numpy().flatten()
        probs_map[x_mask, y_mask] = probs
        count += 1

        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, flip : {}, rotate : {}, batch : {}/{}, Run Time : {:.2f}'
            .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), dataloader.dataset._flip,
                dataloader.dataset._rotate, count, num_batch, time_spent))

    return probs_map


def make_dataloader(args, cfg, flip='NONE', rotate='NONE'):
    batch_size = cfg['batch_size']
    num_workers = args.num_workers

    dataloader = DataLoader(
        GridWSIPatchDataset(args.wsi_path, args.mask_path,
                            image_size=cfg['image_size'],
                            crop_size=cfg['crop_size'], normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    mask = np.load(args.mask_path)
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
            args, cfg, flip='NONE', rotate='NONE')
        probs_map = get_probs_map(encoder, bn, decoder, dataloader)
    else:
        probs_map = np.zeros(mask.shape)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='NONE')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_90')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_180')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='NONE', rotate='ROTATE_270')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='NONE')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_90')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_180')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        dataloader = make_dataloader(
            args, cfg, flip='FLIP_LEFT_RIGHT', rotate='ROTATE_270')
        probs_map += get_probs_map(encoder, bn, decoder, dataloader)

        probs_map /= 8

    np.save(args.probs_map_path, probs_map)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
