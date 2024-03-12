import numpy as np
import torch
from torch.autograd import Function


def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        print(thresh)
        statistics = []

        for pred, gt in zip(pred_list, gt_list):
            gt_img   = (gt/255).astype('uint8')
            pred_img = (pred/255 > thresh).astype('uint8')
            statistics.append(get_statistics(pred_img, gt_img))

        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)
        # calculate recall
        r_acc = tp/(tp+fn)
        # calculate f-score
        final_accuracy_all.append([thresh, p_acc, r_acc, 2*p_acc*r_acc/(p_acc+r_acc)])
    return final_accuracy_all


def calmetrics(pred_list, gt_list):
    final_accuracy_all = []
    statistics = []

    for pred, gt in zip(pred_list, gt_list):
        # calculate each image
        statistics.append(get_statistics(pred, gt))
    # get tp, fp, fn
    tp = np.sum([v[0] for v in statistics])
    fp = np.sum([v[1] for v in statistics])
    fn = np.sum([v[2] for v in statistics])

    # calculate precision
    p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
    # calculate recall
    r_acc = tp / (tp + fn)
    # calculate f-score
    final_accuracy_all.append([ p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])
    return final_accuracy_all


def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    return [tp, fp, fn] 


class dirpf(Function):

    def forward(self, input, target):
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        self.n_target=torch.sum(target) + eps
        self.n_input = torch.sum(input) + eps

        dice= (2 * self.inter.float() + eps) / self.union.float()
        iou=( self.inter.float() + eps) / (self.union.float()- self.inter.float())   #分母表示两个区域的并集
        recall=(self.inter.float()+eps)/self.n_target
        precition=(self.inter.float()+eps)/self.n_input
        # f1=(2*recall*precition)/(recall+precition)  #f1去下面算个总的
        t=torch.FloatTensor([dice,iou,recall,precition,0,0]).cuda()

        return t


def DIRPF(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor([0,0,0,0,0,0]).cuda
    batch_size=len(input)
    for i, c in enumerate(zip(input, target)):
        s = s + dirpf().forward(c[0], c[1])

    for j in range(4):
         s[j]=s[j]/batch_size
    s[4]=(2*s[2]*s[3]/(s[2]+s[3]))
    return s