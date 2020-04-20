import pandas as pd
import numpy as np
from src.util import get_model, train_test_split, get_xy
import src.constants as constants
from src.cv import k_fold_cross_validation
import matplotlib.pyplot as plt


def get_filename(model_type, param_name):
    if model_type == constants.NAIVE_BAYES:
        model_type = 'NB'
    elif model_type == constants.LOG_REGRESSION:
        model_type = 'LR'
    elif model_type == constants.SVM:
        model_type = 'SVM'

    filename = '../src/outputs/' + model_type + '_'
    filename += param_name + '.pdf'
    return filename


def plot_errorbar(x, y, yerr, model_type, param_name):
    plt.clf()
    plt.errorbar(x, y, yerr=yerr)
    plt.show()
    plt.savefig(get_filename(model_type, param_name))


def cross_validate(model_type, dict):
    model = get_model(model_type, **dict)
    acc, std = k_fold_cross_validation(10, train_x, train_y, model)
    return acc, std


def run_xval_and_plot(model_type, param_name='none', values=None):
    if values is None:
        acc, std = cross_validate(model_type, {})
        print('Mean Accuracy:', acc)
        print('Standard Deviation:', std)
        return
    accs = []
    stds = []
    for val in values:
        dict = {param_name: val}
        acc, std = cross_validate(model_type, dict)
        accs.append(acc)
        stds.append(std)

    print(accs)
    print(stds)
    plot_errorbar(values, accs, stds,
                  model_type, param_name)


if __name__ == "__main__":
    df = pd.read_excel('../X_data.xlsx', index_col=0)
    # print(df)
    df = df.drop(columns=['task_id', 'formula'])
    y = pd.read_excel('../y_label.xlsx', index_col=0)
    # print(y)
    df['labels'] = y[0]
    # print(df)

    train, test = train_test_split(df)
    train_x, train_y = get_xy(train)

    run_xval_and_plot(constants.NAIVE_BAYES)

    """
    regularizers = ['l2', 'l1']
    run_xval_and_plot(constants.LOG_REGRESSION,
                      'regularizer',
                      regularizers)
    """
    c_values = [1, 0.8, 0.6, 0.4, 0.2]
    run_xval_and_plot(constants.LOG_REGRESSION,
                      'reg_param',
                      c_values)

    # model = get_model(constants.SVM)
    # k_fold_cross_validation(10, train_x, train_y, model)
