import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(cm,
                    target_names,
                    title='Confusion matrix',
                    cmap=None,
                    normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=9)
        plt.yticks(tick_marks, target_names, fontsize=9)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.svg", format='svg')
    plt.show()


def plot_roc(y_true, y_scores, class_name):
    plt.title('Receiver Operating Characteristic(ROC) curves')
    fpr = []
    tpr = []
    roc_auc = []
    for i in range(len(class_name)):
        f, t, thresholds = roc_curve(y_true, y_scores[i], pos_label=class_name[i])
        fpr.append(f)
        tpr.append(t)
        roc_auc.append(auc(f, t))
        plt.plot(fpr[i], tpr[i], linewidth=1, label=class_name[i] + u'(AUC = %0.4f)'% roc_auc[i])


    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.savefig("roc_curves.svg", format='svg')
    plt.show()


