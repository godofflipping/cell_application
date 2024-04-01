import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as album
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")


def get_data():
    IS_COLAB = False
    if IS_COLAB:
        dataset_dir = '/content/drive/MyDrive/Degree/bone_marrow_cell_dataset/'
        dataframe_dir = '/content/drive/MyDrive/Degree/dataset_paths.csv'
    else:
        dataset_dir = 'bone_marrow_cell_dataset/'
        dataframe_dir = 'dataset_paths.csv'

    df = pd.read_csv(dataframe_dir)

    categories = list(df.groupby('labels').count().index)
    replace_dict = {key: value for value, key in enumerate(categories)}
    reverse_dict = {key: value for key, value in enumerate(categories)}

    print(df.shape)
    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)

    strat = df['labels']
    train_df, test_df = train_test_split(df,
        train_size = 0.75,
        shuffle = True,
        random_state = RANDOM_STATE,
        stratify = strat
    )

    train_df_group = train_df.groupby('labels').count()
    test_df_group = test_df.groupby('labels').count()

    NUM_PICS = 15000
    TRAIN_SAMPLE_SIZE = 1500
    TEST_SAMPLE_SIZE = 500

    for label in train_df_group.index.to_list():
        if train_df_group.loc[label, 'filepaths'] < TRAIN_SAMPLE_SIZE:
            df_minor = train_df[train_df.labels == label]
            df_major = train_df[train_df.labels != label]

            df_minor_up = resample(
                df_minor,
                replace = True,
                n_samples = TRAIN_SAMPLE_SIZE,
                random_state = RANDOM_STATE
            )
            train_df = pd.concat([df_major, df_minor_up])

        if test_df_group.loc[label, 'filepaths'] < TEST_SAMPLE_SIZE:
            df_minor = test_df[test_df.labels == label]
            df_major = test_df[test_df.labels != label]

            df_minor_up = resample(
                df_minor,
                replace = True,
                n_samples = TEST_SAMPLE_SIZE,
                random_state = RANDOM_STATE
            )
            test_df = pd.concat([df_major, df_minor_up])
            
    print(train_df.shape)
    print(test_df.shape)

    train_df = train_df.groupby('labels', as_index=False).apply(lambda x: x.sample(TRAIN_SAMPLE_SIZE, replace=True, random_state=RANDOM_STATE)).reset_index()
    train_df = train_df.drop(['level_0', 'level_1'], axis=1)
    train_df = train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    test_df = test_df.groupby('labels', as_index=False).apply(lambda x: x.sample(TEST_SAMPLE_SIZE, replace=True, random_state=RANDOM_STATE)).reset_index()
    test_df = test_df.drop(['level_0', 'level_1'], axis=1)
    test_df = test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(train_df.shape)
    print(test_df.shape)

    train_df['labels'].replace(replace_dict, inplace=True);
    test_df['labels'].replace(replace_dict, inplace=True);

    class ImageData(Dataset):
        def __init__(self, df, transform, dataset_dir):
            super().__init__()
            self.df = df
            self.image_paths = self.df.filepaths.tolist()
            self.classes = self.df.labels.tolist()
            self.transform = transform
            self.dataset_dir = dataset_dir

        def __len__(self):
            return len(self.df)

        def __getitem__(self, index):
            label = self.classes[index]
            img_path = dataset_dir + self.image_paths[index]
            image = plt.imread(img_path)[:,:,:3]
            image = self.transform(image=image)['image']
            return image, label
        
    BATCH_SIZE = 20
    image_side = 250
    
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)

    train_transform = album.Compose([
        #transforms.ToPILImage(),
        album.HorizontalFlip(),
        album.VerticalFlip(),
        album.SmallestMaxSize(max_size=256),
        album.RandomCrop(height=image_side, width=image_side),
        album.OneOf([
            album.Blur(),
            album.GaussNoise(),
            album.RandomBrightnessContrast(),
            album.RGBShift(),
        ], p=1),
        album.Resize(image_side, image_side),
        ToTensorV2()
    ])
    train_data = ImageData(
        df = train_df,
        transform = train_transform,
        dataset_dir = dataset_dir
    )
    train_loader = DataLoader(
        dataset = train_data,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    test_transform = album.Compose([
        #transforms.ToPILImage(),
        album.HorizontalFlip(),
        album.VerticalFlip(),
        album.SmallestMaxSize(max_size=256),
        album.CenterCrop(image_side, image_side),
        album.Resize(image_side, image_side),
        ToTensorV2()
    ])
    test_data = ImageData(
        df = test_df,
        transform = test_transform,
        dataset_dir = dataset_dir
    )
    test_loader = DataLoader(
        dataset = test_data,
        batch_size = BATCH_SIZE,
        shuffle = False
    )
    
    return train_loader, test_loader, reverse_dict, categories