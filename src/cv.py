import pandas as pd
import numpy as np

from src.confusion_matrix import ConfusionMatrix
from src.util import get_xy, get_confusion_mat, get_model


def k_fold_cross_validation(k, X, y, model):
    df = pd.DataFrame(X)
    df['label'] = np.array(y)
    conf_mats = []
    for i in range(k):
        train, test = get_ith_fold(k, df, i)
        train_x, train_y = get_xy(train)
        test_x, test_y = get_xy(test)
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        conf_mats.append(get_confusion_mat(test_y, pred))

    accs = [mat.get_accuracy() for mat in conf_mats]
    print(accs)
    return np.average(accs), np.std(accs)


def get_ith_fold(k, df, i):
    n = len(df)
    start = int(np.floor(n * i / k))
    end = int(np.floor(n * (i + 1) / k))
    test = df.iloc[start:end, :]
    train = df.drop(test.index)
    return train, test


def cross_validate(data, model_type, dict):
    x, y = get_xy(data)
    model = get_model(model_type, **dict)
    acc, std = k_fold_cross_validation(10, x, y, model)
    return acc, std


if __name__ == "__main__":
    pass


