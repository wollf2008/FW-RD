import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


parser = argparse.ArgumentParser()
parser.add_argument('image_path', default=None, metavar='IMAGE_PATH',
                    type=str, help='Anomoly image path')
args = parser.parse_args()

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

heatmap = np.load(args.image_path)

th = 0.1
img = gaussian_filter(heatmap, sigma=1)


img = min_max_norm(img)
img = img*255
img[img<102] = 0.2*img[img<102]
img[img>=102] = 2.206*(img[img>=102] -102)+20.4
img[img>170] = (img[img>170]-20.4)/2.206 + 102


img = img/255
img[img<th] = 0
plt.imshow(img)
plt.show()
img = cvt2heatmap(img * 255)

plt.imshow(img)
plt.show()
cv2.imwrite('ad.png', img)


