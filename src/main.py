import pandas as pd
import numpy as np
from src.util import get_model, train_test_split, get_xy, get_confusion_mat, get_filename, plot_errorbar
from src.confusion_matrix import ConfusionMatrix
import src.constants as constants
from src.cv import cross_validate
import matplotlib.pyplot as plt


def run_xval_and_plot(data, model_type, param_name='none', values=None):
    accs = []
    stds = []
    for val in values:
        dict = {param_name: val}
        acc, std = cross_validate(data, model_type, dict)
        accs.append(acc)
        stds.append(std)

    # print(accs)
    # print(stds)
    plot_errorbar(values, accs, stds,
                  model_type, param_name)
    return accs, stds


def plot_ROC_curve(train, test, model_type, params):
    train_x, train_y = get_xy(train)
    test_x, test_y = get_xy(test)
    model = get_model(model_type, **params)
    model.fit(train_x, train_y)

    pred_prob = model.predict_proba(test_x)
    pred_prob = pred_prob[:, 1]
    thresholds = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    specs = []
    sens = []
    for thresh in thresholds:
        pred = [1 if i > thresh else 0 for i in pred_prob]
        conf_mat = get_confusion_mat(test_y, pred)
        specs.append(conf_mat.get_specificity())
        sens.append(conf_mat.get_sensitivity())

    plt.clf()
    plt.plot(specs, sens)
    plt.plot([0, 1], [1, 0], '-')
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    filename = get_filename('ROC', model_type, 'pdf')
    plt.savefig(filename, format='pdf')
    plt.show()


def get_best_param_combination(model_type, param_dict):
    best_param = {}
    for param_name in param_dict.keys():
        accs, stds = run_xval_and_plot(train, model_type,
                                       param_name,
                                       param_dict[param_name])
        max_ind = np.argmax(accs)
        best_param[param_name] = param_dict[param_name][max_ind]

    return best_param


if __name__ == "__main__":
    df = pd.read_excel('../X_data.xlsx', index_col=0)
    # print(df)
    df = df.drop(columns=['task_id', 'formula'])
    y = pd.read_excel('../y_label.xlsx', index_col=0)
    # print(y)
    df['labels'] = y[0]
    # print(df)

    train, test = train_test_split(df)
    # print('Naive Bayes')
    # acc, std = cross_validate(train, constants.NAIVE_BAYES, {})
    # print('Mean Accuracy:', acc)
    # print('Standard Deviation:', std)

    param_dict = {constants.LOG_REGRESSION: {
        'regularizer': ['l2', 'l1'],
        'reg_param': [.000001, 0.0001, 1, 1000, 10000, 100000, 1000000]
    },
        constants.SVM: {
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'reg_param': [10000, 1000, 1, 0.0001],
            'gamma': [2, 1, 0.5]
        }}

    best_param = {}
    model_type = constants.LOG_REGRESSION
    param_comb = get_best_param_combination(model_type,
                                            param_dict[model_type])
    best_param[model_type] = param_comb

    # model_type = constants.SVM
    # param_comb = get_best_param_combination(model_type,
    #                                         param_dict[model_type])
    # best_param[model_type] = param_comb

    # The best param is a dictionary that contains the
    # best combination of the parameters
    print(best_param)

    model_type = constants.LOG_REGRESSION
    plot_ROC_curve(train, test,
                   model_type, best_param[model_type])


    model_type = constants.SVM
    plot_ROC_curve(train, test,
                   model_type, best_param[model_type])


    """
    # model = get_model(constants.SVM)
    # k_fold_cross_validation(10, train_x, train_y, model)
    # print weights for fitted SVM
    # print(model.coef_)
    """