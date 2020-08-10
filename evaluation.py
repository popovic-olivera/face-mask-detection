import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def evaluate_model(labels, predictions, probabilities=None, msg=None):
    print(msg)

    draw_matrix(labels, predictions, plt.cm.Greens)

    print(f'Accuracy: {metrics.accuracy_score(labels, predictions)}')
    print(f'Precision(with mask): {metrics.precision_score(labels, predictions)}')
    print(f'Recall(with mask): {metrics.recall_score(labels, predictions)}')
    print(f'F1: {metrics.f1_score(labels, predictions)}')

    # precision, recall, _ = metrics.precision_recall_curve(labels, probabilities)
    # plot_curve(precision, recall)


def plot_curve(precision, recall):
    plt.figure()
    
    plt.plot(recall, precision)
    
    plt.title("Precision recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")


def draw_matrix(labels, predicted, cmap):
    conf_mat = metrics.confusion_matrix(labels, predicted)
    
    fig, ax = plt.subplots()
    img = ax.imshow(conf_mat, interpolation = 'nearest', cmap = cmap)
    ax.figure.colorbar(img, ax = ax)

    ax.set(xticks = np.arange(conf_mat.shape[1]),
           yticks = np.arange(conf_mat.shape[0]),
           xticklabels = ['Without mask', 'With mask'], yticklabels = ['Without mask', 'With mask'],
           title = 'Confusion matrix',
           ylabel = 'True class',
           xlabel = 'Predicted class')

    fmt = '.2f' 
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], fmt),
                    ha = "center", va = "center",
                    color = "white" if conf_mat[i, j] > thresh else "black")

    fig.tight_layout()

    plt.savefig('conf_mat_last.png')
    plt.show()



def confusion_matrix(labels, predicted):
    cm = np.zeros((2, 2))

    correct = labels == predicted
    cm[0, 0] = sum(correct & labels)
    cm[0, 1] = sum(~correct & ~labels)
    cm[1, 0] = sum(~correct & labels)
    cm[1, 1] = sum(correct & ~labels) 

    return cm


def accuracy(labels, predicted):
    tp, fp, fn, tn = confusion_matrix(labels, predicted).ravel()
    
    return (tp + tn)/ (tp + tn + fp + fn)


def precision(labels, predicted, image_class):
    tp, fp, fn, tn = confusion_matrix(labels, predicted).ravel()
    
    return tp / (fp + tp) if image_class else tn / (tn + fn)


def recall(labels, predicted, image_class):
    tp, fp, fn, tn = confusion_matrix(labels, predicted).ravel()
    
    return tp / (tp + fn) if image_class else tn / (tn + fp)
