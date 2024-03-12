from utils.dataset import BasicDataset

# COVID-19-CT-Seg Dataset
covid_train_imgs_dir = r'/dataset/COVID-19-CT-Seg/train/imgs'
covid_train_masks_dir = r'/dataset/COVID-19-CT-Seg/train/labels'
covid_test_imgs_dir = r'/dataset/COVID-19-CT-Seg/test/imgs'
covid_test_masks_dir = r'/dataset/COVID-19-CT-Seg/test/labels'

# CC-CCII Dataset
ccii_train_imgs_dir = r'/dataset/CC-CCII/train/imgs'
ccii_train_masks_dir = r'/dataset/CC-CCII/train/labels'
ccii_test_imgs_dir = r'/dataset/CC-CCII/test/imgs'
ccii_test_masks_dir = r'/dataset/CC-CCII/test/labels'


def covid_dataset():
    train_dataset = BasicDataset(covid_train_imgs_dir, covid_train_masks_dir)
    test_dataset = BasicDataset(covid_test_imgs_dir, covid_test_masks_dir)
    return train_dataset, test_dataset


def ccii_dataset():
    train_dataset = BasicDataset(ccii_train_imgs_dir, ccii_train_masks_dir)
    test_dataset = BasicDataset(ccii_test_imgs_dir, ccii_test_masks_dir)
    return train_dataset, test_dataset


def get_dataset(dataset, img_scale, AU):
    if dataset == 'COVID-19-CT-Seg':
        train_dataset, test_dataset = covid_dataset()
    elif dataset == 'CC-CCII':
        train_dataset, test_dataset = ccii_dataset()
    else:
        pass
    return train_dataset, test_dataset

