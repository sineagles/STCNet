import os
import torch
from eval.eval import pre_test
from torch.utils.data import DataLoader
from utils.LossFunctions import BCEDiceLoss
from utils.dataset import BasicDataset_for_predict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_imgs_dir = r'/dataset/CC-CCII/test/imgs'
    test_masks_dir = r'/dataset/CC-CCII/test/labels'

    net = torch.load(
        r'/checkpoints/CP_BestDice_epoch_Allnet.pth')

    net.to(device)
    val_dataset = BasicDataset_for_predict(test_imgs_dir, test_masks_dir, 1)

    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    criterion = BCEDiceLoss()
    val_score = pre_test(net=net, loader=val_loader, device=device,
                         save_path=r'/pre/')
    print(val_score)
