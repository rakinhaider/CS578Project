from matplotlib import pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import src.constants as constants
from src.confusion_matrix import ConfusionMatrix
from src import constants as constants


def train_test_split(df, fraction=0.2):
    test = df.sample(frac=fraction, random_state=47)
    train = df.drop(index=test.index)

    return train, test


def get_model(model_type, **kwargs):
    if model_type == constants.NAIVE_BAYES:
        model = GaussianNB()
    elif model_type == constants.LOG_REGRESSION:
        reg = kwargs.get('regularizer', 'l2')
        c = kwargs.get('reg_param', 1)
        if reg == 'l2':
            solver = 'lbfgs'
        else:
            solver = 'saga'
        model = LogisticRegression(penalty=reg,
                                   C=c,
                                   solver=solver,
                                   max_iter=10000,
                                   random_state=47)
    
    elif model_type == constants.SVM:
        c1 = kwargs.get('reg_param', 1)
        kern = kwargs.get('kernel', 'rbf')
        if kern == 'rbf':
            gamma = kwargs.get('gamma', 'scale')
            model = SVC(C=c1, kernel=kern, gamma=gamma,
                        random_state=47, max_iter=10000)
        else:
            model = SVC(C=c1, kernel=kern,
                        random_state=47, max_iter=10000)

    return model


def get_xy(df):
    return df[df.columns[:-1]], df[df.columns[-1]]


def get_confusion_mat(orig, pred):
    tp, fp, fn, tn = (0, 0, 0, 0)
    for i in range(len(orig)):
        if orig.iloc[i] == pred[i]:
            if orig.iloc[i] == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if orig.iloc[i] == 0:
                fp = fp + 1
            else:
                fn = fn + 1
    return ConfusionMatrix(tp, fp, fn, tn)


def get_filename(plot_type, model_type, extension,
                 param_name=''):
    if model_type == constants.NAIVE_BAYES:
        model_type = 'NB'
    elif model_type == constants.LOG_REGRESSION:
        model_type = 'LR'
    elif model_type == constants.SVM:
        model_type = 'SVM'

    filename = '../src/outputs/' + plot_type + '_'
    filename += model_type + '_'
    filename += param_name + '.' + extension
    return filename


def plot_errorbar(x, y, yerr, model_type, param_name):
    plt.clf()
    ticks = [i for i in range(len(x))]
    plt.errorbar(ticks, y, yerr=yerr)
    plt.xticks(ticks, x)
    plt.xlabel(param_name)
    plt.ylabel('Average Validation Accuracy with Standard Deviation')
    filename = get_filename('acc', model_type, 'pdf', param_name)
    plt.savefig(filename, format='pdf')
    plt.show()


def get_weights(fitted_model):
    return fitted_model.coef_


#plot correlation for each feature
def feature_plot(classifier, feature_names, top_features=4):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(18, 7))
    colors = ['green' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1 + 2 * top_features), feature_names[top_coefficients], rotation=45, ha='right')
    plt.show()
