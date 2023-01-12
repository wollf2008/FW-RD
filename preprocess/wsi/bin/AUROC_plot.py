import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os
from sklearn.metrics import roc_auc_score,roc_curve, auc
import logging
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


parser = argparse.ArgumentParser()
parser.add_argument('probs_map_path', default=None, metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the input probs_map numpy file')
args = parser.parse_args()

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)
def get_threshold(gt,score):
    gt_mask = np.asarray(gt)
    score = np.asarray(score)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score.flatten())
    a = 2*precision*recall
    b = precision + recall
    f1 = np.divide(a,b,out=np.zeros_like(a), where=b !=0)
    threshold = thresholds[np.argmax(f1)]
    return threshold

def plot_roc(gt_y, prob_predicted_y):
    fpr, tpr, _ = roc_curve(gt_y, prob_predicted_y)
    print(fpr, tpr)

    plt.figure(0).clf()

    roc_auc = auc(fpr, tpr)
    # r'$\alpha_i > \beta_i$'
    plt.plot(fpr, tpr, 'r', label=r'$Our = %0.4f$' % roc_auc)

    plt.title('ROC curves')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    lg = plt.legend(loc='lower right', borderaxespad=1.)
    lg.get_frame().set_edgecolor('k')
    plt.grid(True, linestyle='-')
    plt.show()

gt = []
pr = []
th = 2.01
bad_path = os.path.join(args.probs_map_path,'bad/')
files = os.listdir(bad_path)
for file in files:
    heatmap = np.load(os.path.join(bad_path,file))
    heatmap = heatmap.flatten()
    heatmap_th = heatmap[heatmap>=th]
    if len(heatmap_th) == 0:
        heatmap_th = np.array([0])

    heatmap_th.sort()
    real = heatmap_th#[-1]
    real = real.mean()
    pr.append(real)
    gt.append(1)

good_path = os.path.join(args.probs_map_path,'good/')
files = os.listdir(good_path)
for file in files:
    heatmap = np.load(os.path.join(good_path,file))
    heatmap = heatmap.flatten()
    heatmap_th = heatmap[heatmap>=th]
    if len(heatmap_th) == 0:
        heatmap_th = np.array([0])

    heatmap_th.sort()
    real = heatmap_th#[-1]
    real = real.mean()
    pr.append(real)
    gt.append(0)


plot_roc(gt, pr)
th = get_threshold(gt,pr)
print(th)
pr_bool = []
for i in pr:
    if i >th:
        pr_bool.append(1)
    else:
        pr_bool.append(0)

logging.info('confusion matrix:')