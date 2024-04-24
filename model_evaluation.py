import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import joblib



# Function definitions
def plot_roc_curve(fpr, tpr, title='ROC Curve'):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def evaluate_models(X_test, y_test, model_path):
    model = joblib.load(model_path)  # Load the model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, predictions))

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plot_roc_curve(fpr, tpr)

    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm)


# Main execution block
if __name__ == "__main__":
    # Example loading data
    X_test = np.load('path_to_X_test.npy')
    y_test = np.load('path_to_y_test.npy')

    # Assuming models are saved with joblib
    evaluate_models(X_test, y_test, 'path_to_trained_model.pkl')
