import torch.nn as nn
import logging
from uts.LossFunctions import BinaryFocalLoss, FocalLoss, SoftDiceLoss, BCEDiceLoss


def get_loss(args):
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args.loss == 'BFL':
        criterion = BinaryFocalLoss()
    elif args.loss == 'FL':
        criterion = FocalLoss()
    elif args.loss == 'BCE+DL':
        criterion = BCEDiceLoss()
    else:
        criterion = SoftDiceLoss()
    logging.info(f'Load model {args.loss} successful')
    return criterion
