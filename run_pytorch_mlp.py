# Dataset: https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data?select=train-sample.csv

import torch
import numpy as np
from src.MLP import MLPClassifier
from src.preprocessing import preprocess_pytorch
from src.util import print_metrics, download
from os.path import exists

import warnings
warnings.filterwarnings('ignore')


### Settings and hyperparameters - for users to change
use_pretrained = False
download_data = False
batch_size = 512
num_epochs = 10

# Download data
train_sample_url = 'https://drive.google.com/uc?export=download&id=1X65WA4__5h0oOE0uuTo1IGfotNrFNMTp&confirm=t&uuid=08b2fcd9-f2fc-4b2b-b200-43061c7aba69&at=ALgDtszCWvg2lCTFeVKcLfMchrXU:1679497955602'
if download_data:
    download(train_sample_url, 'data/train_sample.csv')

# Preprocessing
CSV_PATH = "data/train_sample.csv"
train_dataset, X_cv, y_cv = preprocess_pytorch(CSV_PATH)

# Load or Train the model    
device = ("cuda" if torch.cuda.is_available() else "cpu")
clf_mlp = MLPClassifier().to(device)
MODEL_PATH = 'models/MLP.pth'
if exists(MODEL_PATH) and use_pretrained:
    clf_mlp.load_state_dict(torch.load(MODEL_PATH))
else:
    clf_mlp = clf_mlp.fit(train_dataset, batch_size, num_epochs, PATH=MODEL_PATH, device=device)

# Make predictions
predicted = clf_mlp.predict(X_cv)

# Metrics
print_metrics(predicted, np.argmax(y_cv, axis=1))

