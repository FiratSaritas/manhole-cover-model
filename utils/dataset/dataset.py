import torch
from torch.utils.data import DataLoader, Dataset
import PIL
import os
from torchvision import transforms as transforms
import pandas as pd
import numpy as np
import pickle


class MHCoverDataset(Dataset):
    """
    Class defines custom Dataset as a workaround to the ImageFolder class
    """

    def __init__(self, root_dir: str, df: pd.DataFrame,  fp_label_translator: 'str', 
                 transform: 'transforms.Compose' = None, label_indexer: str = 'label'):
        """
        Initializes the dataset class.

        Params:
        ------------------
        root_dir: str              Defines the path from where all images should be imported from
        df: pd.DataFrame           Prefiltered!! Dataframe to load labels from (filtered according to Train-Val-test set)
        transform: Compose         Compose of different transforms applied during import of an image.
        fp_label_translator: str   Path to pickle file with label translator.
        label_indexer: str         Column name of the index to take.
        _images: list              List of all images names in the root dir.
        """
        self.root_dir = root_dir
        self.df = df
        self.fp_label_translator = fp_label_translator
        self.transform = transform
        self.label_indexer = label_indexer
        self.label_dict = None
        self.label_dict_r = None
        self._images = None
        self._init_attributes()
        
    def _init_attributes(self):
        """
        Creates Dictionary for Label Encoding
        Matches Data in Folder with the ones in the filtered pd.DataFrame
        """
        # Label dict
        with open(self.fp_label_translator, 'rb') as pkl_file:
            self.label_dict = pickle.load(pkl_file)
        self.label_dict_r = {i:label for label, i in self.label_dict.items()}
        
        # Load images
        self._images = list(self.df["image"])
        
        
        # Only select thos images within the dataframe        
        #assert self.df.shape[0] == self._images.shape[0], f'{self.df.shape[0]} - {self._images.shape[0]}'
        
        return self
    
    def label_encode(self, X):
        if type(X) == list:
            return [self.label_dict_r[lab] for lab in X]
        else:
            return self.label_dict_r[X]

    def __len__(self):
        """Returns length of Dataset"""
        return len(self._images)

    def __getitem__(self, idx):
        """Returns Item with given IDX"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = os.path.join(self.root_dir, self._images[idx])
        image = PIL.Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        # Load label        
        label = self.df.loc[self.df['image'] == self._images[idx], self.label_indexer].to_list()
        
        assert len(label) == 1
        
                
        return image, self.label_dict[label[0]]


def get_dataloader(root_dir: str, df: pd.DataFrame, transformations: 'transforms.Compose', 
                   fp_label_translator: str, batch_size: int, workers: int, **dlkwargs):
    """
    Function returns a dataloader with given parameters

    Params:
    ---------------
    root_dir: str          Defines the path from where all images should be imported from
    df: pd.DataFrame       Prefiltered Dataframe to load labels from (filtered according to Train-Val-test set)
    transformations:       Compose of different transforms applied during import of an image.
    fp_label_translator:   File path to pickle file with label translations.
    batch_size; int        Size of the imported batch
    workers: int           Amount of CPU workers for the loading of data into gpu.
    """
    custom_dataset = MHCoverDataset(root_dir=root_dir, df=df, 
                                    transform=transformations, 
                                    fp_label_translator=fp_label_translator)

    return DataLoader(dataset=custom_dataset,
                      batch_size=batch_size,
                      num_workers=workers, 
                      **dlkwargs)
