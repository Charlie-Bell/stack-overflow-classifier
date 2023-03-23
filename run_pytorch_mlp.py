# Dataset: https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data?select=train-sample.csv

import torch
import numpy as np
from src.MLP import MLPClassifier
from src.preprocessing import preprocess_pytorch
from src.util import print_metrics
from os.path import exists

import warnings
warnings.filterwarnings('ignore')


### SETTINGS - for users
use_pretrained = False
device = ("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "data/train_sample.csv"
MODEL_PATH = 'models/MLP.pth'
batch_size = 512
num_epochs = 10

# Preprocessing
train_dataset, X_cv, y_cv = preprocess_pytorch(CSV_PATH)

# Load or Train the model    
clf_mlp = MLPClassifier().to(device)
if exists(MODEL_PATH) and use_pretrained:
    clf_mlp.load_state_dict(torch.load(MODEL_PATH))
else:
    clf_mlp = clf_mlp.fit(train_dataset, batch_size, num_epochs, PATH=MODEL_PATH, device=device)

# Make predictions
predicted = clf_mlp.predict(X_cv)

# Metrics
print_metrics(predicted, np.argmax(y_cv, axis=1))

