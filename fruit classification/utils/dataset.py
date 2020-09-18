# import required libraries
import os
import pandas as pd
import numpy as np
import csv
import shutil  
import sklearn
import sklearn.model_selection
from PIL import Image
from torch.utils.data.dataset import Dataset

def create_meta_csv(dataset_path, destination_path):
    """Create a meta csv file given a dataset folder path of images.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The purpose behind creating this file is to allow loading of images on demand as required. Only those images required are loaded randomly but on demand using their paths.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta file if None provided, it'll store file in dataset_path

    Returns:
        True (bool): Returns True if 'dataset_attr.csv' was created successfully else returns an exception
    """

    # Change dataset path accordingly
    DATASET_PATH = os.path.abspath(dataset_path)

    if not os.path.exists(os.path.join(DATASET_PATH, "/dataset_attr.csv")):

        csvpath = os.path.join(DATASET_PATH, "dataset_attr.csv")
        # Make a csv with full file path and labels
        folderlist = os.listdir(DATASET_PATH)
        fruits=["Apple","Banana","Orange","Pineapple","Strawberry"]
        value = np.linspace(1, 10000, 10000)
        value = np.append(value, 'label')
        
        with open(csvpath, "a") as f:
            writer = csv.writer(f)
            writer.writerow(value)

        # write out as dataset_attr.csv in destination_path directory
        for x in folderlist:
            folderpath = os.path.join(DATASET_PATH, x)
            #print("folder path: ", folderpath)
            if(os.path.isdir(folderpath)):
                filelist = os.listdir(folderpath)
                for files in filelist:
                    imagepath = os.path.join(folderpath, files)
                    #print("Image path: ", imagepath)
                    img = Image.open(imagepath)
                    img_grey = img.convert('L')
                    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
                    value = value.flatten()
                    #add pixels and label to value
                    value = np.append(value, fruits.index(x)) 
                    #print("value: ", value)
                    with open(csvpath,'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(value) 
                
                print('Done :::: ', x)

        # change destination_path to DATASET_PATH if destination_path is None 
        if destination_path == None:
            destination_path = DATASET_PATH

        #write out as dataset_attr.csv in destination_path directory
        if(shutil.move(csvpath ,destination_path)):
            return True
    else:
        return False

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):
    """Create a meta csv file given a dataset folder path of images and loads it as a pandas dataframe.

    This function creates and saves a meta csv file named 'dataset_attr.csv' given a dataset folder path of images.
    The file will contain images and their labels. This file can be then used to make
    train, test and val splits, randomize them and load few of them (a mini-batch) in memory
    as required. The file is saved in dataset_path folder if destination_path is not provided.

    The function will return pandas dataframes for the csv and also train and test splits if you specify a 
    fraction in split parameter.
    
    Args:
        dataset_path (str): Path to dataset folder
        destination_path (str): Destination to store meta csv file
        randomize (bool, optional): Randomize the csv records. Defaults to True
        split (double, optional): Percentage of train records. Defaults to None

    Returns:
        dframe (pandas.Dataframe): Returns a single Dataframe for csv if split is none, else returns more two Dataframes for train and test splits.
        train_set (pandas.Dataframe): Returns a Dataframe of length (split) * len(dframe)
        test_set (pandas.Dataframe): Returns a Dataframe of length (1 - split) * len(dframe)
    """
    if not os.path.exists(os.path.join(destination_path, "dataset_attr.csv")):
        if create_meta_csv(dataset_path, destination_path=destination_path):
            dframe = pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'))
    else:
        print("csv file for dataset is already present.")
        dframe = pd.read_csv(os.path.join(destination_path, 'dataset_attr.csv'))
    print("Dataframe is created.")

    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
    if randomize == True or (split != None and randomize == None):
        # shuffle the dataframe here
        dframe = sklearn.utils.shuffle(dframe)
        train_set, test_set = train_test_split(dframe, split)
        pass

    if split != None:
        train_set, test_set = train_test_split(dframe, split)
        return dframe, train_set, test_set 
    
    return dframe

def train_test_split(dframe, split_ratio):
    """Splits the dataframe into train and test subset dataframes.

    Args:
        split_ration (float): Divides dframe into two splits.

    Returns:
        train_data (pandas.Dataframe): Returns a Dataframe of length (split_ratio) * len(dframe)
        test_data (pandas.Dataframe): Returns a Dataframe of length (1 - split_ratio) * len(dframe)
    """
    # divide into train and test dataframes
    train_no = int((split_ratio) * len(dframe))
    test_no = int((1 - split_ratio) * len(dframe))+1
    train_data = dframe[0:train_no]
    test_data = dframe[(len(dframe) - test_no):]

    return train_data, test_data

class ImageDataset(Dataset):
    """Image Dataset that works with images
    
    This class inherits from torch.utils.data.Dataset and will be used inside torch.utils.data.DataLoader
    Args:
        data (str): Dataframe with path and label of images.
        transform (torchvision.transforms.Compose, optional): Transform to be applied on a sample. Defaults to None.
    
    Examples:
        >>> df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize=randomize, split=0.99)
        >>> train_dataset = dataset.ImageDataset(train_df)
        >>> test_dataset = dataset.ImageDataset(test_df, transform=...)
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        # get unique classes from data dataframe
        self.classes = self.data['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #img_path = self.data.iloc[idx]['path']
        #print("idx: ", idx)
        image = np.asarray(self.data.iloc[idx][0:10000]).reshape(100,100).astype('uint8')
        #image = Image.fromarray(img_np)
        
        # load PIL image
        #image = image.convert('L')
        
        # get label (derived from self.classes; type: int/long) of image
        label = self.data.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    # test config
    dataset_path = '../../Data/fruits'
    dest = '../../Code'
    classes = 5
    total_rows = 4323
    randomize = True
    clear = True
    
    # test_create_meta_csv()
    df, trn_df, tst_df = create_and_load_meta_csv_df(dataset_path, destination_path=dest, randomize=randomize, split=0.99)
    print(df.describe())
    print(trn_df.describe())
    print(tst_df.describe())
