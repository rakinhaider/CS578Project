import pandas as pd
import numpy as np
from src.util import get_model, train_test_split, get_xy
import src.constants as constants
from src.cv import k_fold_cross_validation

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
    model = get_model(constants.NAIVE_BAYES)

    k_fold_cross_validation(10, train_x, train_y, model)

    regularizer = ['l2', 'l1']
    for reg in regularizer:
        model = get_model(constants.LOG_REGRESSION, regularizer=reg)
        k_fold_cross_validation(10, train_x, train_y, model)

    C_values = [1, 0.8, 0.6, 0.4, 0.2]
    for c in C_values:
        model = get_model(constants.LOG_REGRESSION, reg_param=c)
        k_fold_cross_validation(10, train_x, train_y, model)
        
    model = get_model(constants.SVM)
    k_fold_cross_validation(10, train_x, train_y, model)
