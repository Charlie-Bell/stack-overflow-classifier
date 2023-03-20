from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def print_metrics(predicted, y_cv): 
    print('Accuracy Score \n',accuracy_score(predicted, y_cv))
    print('Confusion Matrix \n', confusion_matrix(predicted, y_cv))
    print('Classification Report \n', classification_report(predicted, y_cv))