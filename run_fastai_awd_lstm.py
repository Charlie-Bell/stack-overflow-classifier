# Dataset: https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data?select=train-sample.csv

from fastai.text.all import *
import pandas as pd
from src.util import download
from src.preprocessing import preprocess_fastai

### SETTINGS - for users
use_pretrained = True
download_data = True
train_sample_url = 'https://drive.google.com/uc?export=download&id=1X65WA4__5h0oOE0uuTo1IGfotNrFNMTp&confirm=t&uuid=08b2fcd9-f2fc-4b2b-b200-43061c7aba69&at=ALgDtszCWvg2lCTFeVKcLfMchrXU:1679497955602'
df_preprocessed_url = 'https://drive.google.com/uc?export=download&id=18rSp1jWkoTJsaOG6BUrSDi9AFnJNtUfl&confirm=t&uuid=6cfaa95f-eef2-4586-a0e9-e0cb69dab257&at=ALgDtszIDGE2nx9QhvjVFgf43qAS:1679498577470'
model_url = 'https://drive.google.com/uc?export=download&id=1--hTWBismsrTXbYZRRuVgfnN5crzKe_g&confirm=t&uuid=efd93b10-af2b-4277-962d-70927f59379f&at=ALgDtsyymWc9X5sFqmiQct2rKSw1:1679498713842'

# Download/Read dataframe
if download_data:
    download(train_sample_url, 'data/train_sample.csv')
    download(df_preprocessed_url, 'data/df_preprocessed.csv')
else:    
    preprocess_fastai('data/train-sample.csv')
df = pd.read_csv('data/df_preprocessed.csv', quoting=csv.QUOTE_ALL, encoding='utf-8')

# Instantiate dataloader
dls = TextDataLoaders.from_df(df, text_col='Text', label_col='OpenStatus', valid_pct=0.1)

# Instantiate text classifier
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# Training loop
if use_pretrained: # download pretrained model
    download(model_url, 'models/high_level_API.pth')
    learn.load('high_level_API')
else: # train a new model
    num_epochs, lr = 4, 1e-2
    learn.fine_tune(num_epochs, lr)
    learn.save('high_level_API')

# Show results
learn.show_results()