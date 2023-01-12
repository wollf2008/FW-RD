import sys
import os
import argparse
import logging

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

parser = argparse.ArgumentParser(description="Get the normal region"
                                             " from tumor WSI ")
parser.add_argument("tumor_path", default=None, metavar='TUMOR_PATH', type=str,
                    help="Path to the tumor mask npy")
parser.add_argument("tissue_path", default=None, metavar='TISSUE_PATH', type=str,
                    help="Path to the tissue mask npy")
parser.add_argument("normal_path", default=None, metavar='NORMAL_PATCH', type=str,
                    help="Path to the output normal region from tumor WSI npy")


def run(args,file):
    a = file.replace('_tumor.npy','')
    b = a+'.npy'
    c = a+'_normal.npy'

    tumor_mask = np.load(os.path.join(args.tumor_path, file))
    tissue_mask = np.load(os.path.join(args.tissue_path, b))
    normal_mask = tissue_mask & (~ tumor_mask)
    np.save(os.path.join(args.normal_path, c), normal_mask)

def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    files = os.listdir(args.tumor_path)  # 读入文件夹
    for file in files:
        run(args,file)



if __name__ == "__main__":
    main()
