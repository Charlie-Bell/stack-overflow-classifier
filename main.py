import torch
import numpy as np
from src.MLP import MLPClassifier
from src.preprocessing import preprocess, print_metrics
from src.util import print_metrics
from os.path import exists


if __name__=="__main__":

    # Settings
    overwrite = False
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "data/train-sample.csv"
    MODEL_PATH = 'models/MLP.pth'
    batch_size = 512
    num_epochs = 10
    

    # Preprocessing
    train_dataset, X_cv, y_cv = preprocess(CSV_PATH)
    
    # Load or Train the model    
    clf_mlp = MLPClassifier().to(device)
    if exists(MODEL_PATH) and not overwrite:
        clf_mlp.load_state_dict(torch.load(MODEL_PATH))
    else:
        clf_mlp = clf_mlp.fit(train_dataset, batch_size, num_epochs, PATH=MODEL_PATH, device=device)
    
    # Make predictions
    predicted = clf_mlp.predict(X_cv)

    # Metrics
    print_metrics(predicted, np.argmax(y_cv, axis=1))
    
    