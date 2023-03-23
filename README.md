# Transfer learning with fast.ai to predict Stack Overflow question answering

<i>The objective of this repository is to use machine learning to predict with a Stack Overflow question will be closed based on text-only inputs. A multi-layer perceptron network is implemented in Pytorch and an AWD LSTM is implemented with fastai. The AWD LSTM greatly outperforms the multi-layer perceptron network. Code is provided to replicate the experiment.</i>

## The dataset

The original dataset consists of a balanced 140,271 labeled stack overflow questions with a question title, body and 5 tags in the form of text features, along with a few numerical features. For the output label there are 5 possible classes, one being 'open' and four being some form of close.

Dataset: https://www.kaggle.com/competitions/predict-closed-questions-on-stack-overflow/data?select=train-sample.csv

Preprocessing was done by removing numerical features, one-hot encoding the output labels, and combining the text features into one single text feature. When cleaning the text, only null values and white-space were removed, since it is preferred to keep stopwords and punctuation for the more complex language models. For use in Pytorch the text was vectorized using tf-idf with 10,000 features.

## The models

In this repository two models have been trained:
1. A fully connected Multi-Layer Perceptron (MLP) network implemented in Pytorch.
2. A state-of-the-art LSTM network 'AWD-LSTM' implemented with the fastai API.

The AWD-LSTM is implemented in the high-level API, with a default language model (LM) trained on Wikipedia with a task to predict the next word. A classifier is finetuned on the text dataset with a training:validation split of 9:1.

### Pytorch

The Pytorch model is a small and simple neural network designed to be quick to train and easy to use as a baseline. It has only 3 hidden layers with no recurrence and can be found in 'MLP.py'.

### Fastai

Fast.ai is a deep learning framework built on top of Pytorch and meant to simplify pipelines using state-of-the-art architectures into only a few lines of code. 

It has APIs designed for multiple levels of abstraction:
 - High Level API - Consisting of a Learner and a Datablock.
 - Mid Level API - Here is the Data Core, Generic Optimizers and Metrics, and Callbacks.
 - Low Level API - Here is API closely interfacing with Pytorch, dealing with Object-Oriented Tensors, Optimized Operations, Reversible Transforms and the Pipeline.
 
<br>
<p align="center">
    <img src = "https://docs.fast.ai/images/layered.png" width=60%>
</p>

The advantage of using this high level API is for rapid prototyping already optimized in the Pytorch framework. With this framework, models such as a the ASGD Weight-Dropped LSTM (AWD LSTM) can quickly be created, trained, and used for inference.

The benefit of using an AWD LSTM is its DropConnect on the hidden states for regularization, and also the non-monotonic averaged gradient descent method used to train it outperforms typical methods. More on AWD LSTMs can be read in the original paper: https://arxiv.org/abs/1708.02182v1

## How to use

Two scripts were created for both models which run data preprocessing, model training, model inference and calculating metrics.
 - To work with the MLP network, run 'run_pytorch_mlp.py'. The variable 'use_pretrained' is used to decide whether to load parameters from a pretrained model, or to train again.
 - To work with the AWD-LSTM, run 'run_fastai_awd_lstm.py'. Again, the variable 'use_pretrained' is used to download a pretrained model from google drive, or train another model. Do note that the training time for the AWD-LSTM can be very long.

Other parameters can be changed in these scripts -
 - In 'run_pytorch_mlp.py' ('epochs': num_epochs, 'batch size': batch_size)
 - In 'run_fastai_awd_lstm.py' ('learning rate' : lr, 'epochs': num_epochs)

 For users on Windows it is best to use the colab notebook for fastai implementations due to compatibility issues: https://colab.research.google.com/drive/1mEt2tYYIYS8aSes_R7EPLEeiT2qWHjBV?usp=sharing

## Results

### Metrics 

The MLP Network achieved a 0.61 accuracy on the validation dataset, while the AWD LSTM achieved 0.68 after only 4 epochs of fine tuning.

### Improvements
The trained models could both definitely make improvements given a more thorough preprocessing of the text, as well as a larger dataset. Including metadata from the original dataset such as user creation date and post creation date would probably be a 

Many improvements can be made to the MLP network, the first and most obvious being the inclusion of recurrence, as well as regularization methods, depth, a hyperparameter search and so on. The list is pretty exhaustive as it would be with a baseline model.

The biggest improvement for the AWD LSTM would be to not only fine-tune the classification model on the Stack Overflow data, but to train the Language Model underneath on a large corpus of Stack Overflow text. The code for this is already setup to be run in the google colab notebook 'fastai_implementation.ipynb', but the compute required would be far to large to run on their cloud machines.

## Conclusion
Fastai provides a very easy to work with API for implementing state of the art NLP models in Pytorch, already machine optimized for pretraining, finetuning and inference. Given more compute, it would be nice to see just how well an AWD LSTM would perform when pretraining the language model on a large corpus of Stack Overflow text.