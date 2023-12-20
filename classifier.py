import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import random_split
from utils import AudioDataLoader
from model import *

data_path = "UrbanSound8K/audio"
download_path = Path.cwd()/'UrbanSound8K'

# Read metadata file
metadata_file = download_path/'metadata'/'UrbanSound8K.csv'


df = pd.read_csv(metadata_file)

# construct file path by concatenating fold and file name

df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

#filter out irrelevant features
df =  df[['relative_path', 'classID']]

data = AudioDataLoader(df, data_path)
print(len(data))

num_items = len(data)
train_pct = round(num_items * 0.8)
val_pct = num_items - train_pct
train_ds, val_ds = random_split(data, [train_pct, val_pct])
print("Train: ", len(train_ds))
print("Validation: ", len(val_ds))


# create dataloader
train_dl = torch.utils.data.DataLoader(train_ds, batch_size = 16, shuffle = True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size = 16, shuffle = True)


# create the model and put it on the GPU if available
audioclf = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
audioclf = audioclf.to(device)
# Check that it is on Cuda
next(audioclf.parameters()).device