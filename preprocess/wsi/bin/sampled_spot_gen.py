import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import copy

sys.path.append(os.path.join(os.path.abspath(__file__), "/../../"))


parser = argparse.ArgumentParser(description="Get center points of patches "
                                             "from mask")
parser.add_argument("mask_path", default=None, metavar="MASK_PATH", type=str,
                    help="Path to the mask npy file")
parser.add_argument("coords_path", default=None, metavar="TXT_PATH", type=str,
                    help="Path to the txt file")
parser.add_argument("patch_number", default=None, metavar="PATCH_NUMB", type=int,
                    help="The number of patches extracted from WSI")
parser.add_argument("--level", default=6, metavar="LEVEL", type=int,
                    help="Bool format, whether or not")


class patch_point_in_mask_gen(object):
    '''
    extract centre point from mask
    inputs: mask path, centre point number
    outputs: centre point
    '''

    def __init__(self, mask_path, number):
        self.mask_path = mask_path
        self.number = number

    def get_patch_point(self):
        mask_tissue = np.load(self.mask_path)
        mask_tissue_hard = copy.deepcopy(mask_tissue)

        X_idcs, Y_idcs = np.where(mask_tissue)
        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)

        for i in range(len(centre_points)):
            try:
                if mask_tissue[centre_points[i][0]+2,centre_points[i][1]+2] == False or mask_tissue[centre_points[i][0]-2,centre_points[i][1]-2] == False or mask_tissue[centre_points[i][0]-2,centre_points[i][1]+2] == False or mask_tissue[centre_points[i][0]+2,centre_points[i][1]-2] == False:
                    mask_tissue_hard[centre_points[i][0],centre_points[i][1]] = False
            except IndexError:
                pass

        X_idcs, Y_idcs = np.where(mask_tissue_hard)
        centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)

        if centre_points.shape[0] > self.number:
            sampled_points = centre_points[np.random.randint(centre_points.shape[0],
                                                                 size=self.number), :]
        else:
            sampled_points = centre_points
        return sampled_points


def run(args, mask_path):
    sampled_points = np.array([[0,0]])
    count = 0
    while len(sampled_points) < args.patch_number and count < 10:
        count+=1
        sampled_point_more = patch_point_in_mask_gen(mask_path, args.patch_number).get_patch_point()
        if len(sampled_point_more) > 0:
            sampled_points = np.concatenate((sampled_points,(sampled_point_more * 2 ** args.level).astype(np.int32)),axis=0) # make sure the factor
        for i in range(0,len(sampled_points)):
            no_cross_sample = sampled_points[i]
            for j in range(i+1,len(sampled_points)):
                #for j in range
                if ((no_cross_sample[0]+128 <= sampled_points[j][0] or no_cross_sample[0]-128 >= sampled_points[j][0])
                    or (no_cross_sample[1]+128 <= sampled_points[j][1] or no_cross_sample[1]-128 >= sampled_points[j][1])):
                    no_cross_sample = no_cross_sample
                else:
                    sampled_points[j] = [0,0]
        sampled_points = np.array([x for x in sampled_points if x.sum() != 0])
    sampled_points = sampled_points[0:args.patch_number]
    print(len(sampled_points))
    if len(sampled_points) > 0:
        mask_name = os.path.split(mask_path)[-1].split(".")[0]
        name = np.full((sampled_points.shape[0], 1), mask_name)
        center_points = np.hstack((name, sampled_points))

        txt_path = args.coords_path

        with open(txt_path, "a") as f:
            np.savetxt(f, center_points, fmt="%s", delimiter=",")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    masks = os.listdir(args.mask_path)
    for mask in masks:
        mask_path = os.path.join(args.mask_path, mask)
        run(args, mask_path)

if __name__ == "__main__":
    main()
