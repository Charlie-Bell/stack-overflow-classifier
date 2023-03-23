import torch
from torch.utils.data import Dataset
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
import texthero as hero
from texthero import preprocessing

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx].float() # Both the data and model parameters should have the same dtype.
        label = self.y[idx].float()
        return sample, label
    
def preprocess_pytorch(PATH):

        data = pd.read_csv(PATH)
        data.head()

        # Should include Owner creation date range and ratio Post Creation Date
        data_train = data[['Title', 'BodyMarkdown', 'OpenStatus', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']]

        custom_pipeline = [preprocessing.fillna,
                           preprocessing.remove_whitespace]

        data_train['Title'] = hero.clean(data_train['Title'], custom_pipeline)
        data_train['BodyMarkdown'] = hero.clean(data_train['BodyMarkdown'], custom_pipeline)
        for i in range(1,6):
            data_train['Tag'+str(i)] = hero.clean(data_train['Tag'+str(i)], custom_pipeline)

        x = data_train[['Title', 'BodyMarkdown', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']].agg(' '.join, axis=1)
        y = pd.get_dummies(data_train['OpenStatus'], prefix='OpenStatus')



        X_train, X_cv, y_train, y_cv = train_test_split(x, y, test_size = 0.3, random_state = 203)
        y_train, y_cv = y_train.to_numpy(), y_cv.to_numpy()

        vectorizer = TfidfVectorizer(max_features = 10000)  # 10k features

        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()  # vectorizing X_train
        X_cv_tfidf = vectorizer.transform(X_cv).toarray()   # vectorizing X_cv

        train_dataset = CustomDataset(X_train_tfidf, y_train)

        return train_dataset, X_cv_tfidf, y_cv


def preprocess_fastai(PATH):    
    data = pd.read_csv(PATH)

    custom_pipeline = [
                        preprocessing.fillna,
                        preprocessing.remove_whitespace,
                        ]

    data['Title'] = hero.clean(data['Title'], custom_pipeline)
    data['BodyMarkdown'] = hero.clean(data['BodyMarkdown'], custom_pipeline)
    for i in range(1,6):
        data['Tag'+str(i)] = hero.clean(data['Tag'+str(i)], custom_pipeline)

    X = data[['Title', 'BodyMarkdown', 'Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']].agg(' '.join, axis=1)
    df = pd.concat( [X, data[['OpenStatus']]], axis=1).rename(columns={0: 'Text'}, inplace=False)

    df.to_csv('data/df_preprocessed.csv')