import numpy as np

from uts.prf_metrics import get_statistics


def PR_Curve(pred, gt, thresh_step=0.01):
    thresh_list = []
    presicion_list = []
    recall_list = []
    f1_list = []

    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []

        last_pred = (pred > thresh).float()

        last_pred = last_pred.view(-1).data.cpu().numpy()
        last_gt = gt.view(-1).data.cpu().numpy()

        statistics.append(get_statistics(last_pred, last_gt))

        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        # calculate recall
        r_acc = tp / (tp + fn)
        # calculate f-score
        f1 = 2 * p_acc * r_acc / (p_acc + r_acc)

        thresh_list.append(thresh)
        presicion_list.append(p_acc)
        recall_list.append(r_acc)
        f1_list.append(f1)

    return thresh_list, presicion_list, recall_list, f1_list
