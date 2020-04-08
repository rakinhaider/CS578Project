import pandas as pd
import numpy as np

from src.util import get_xy, get_measures


def k_fold_cross_validation(k, X, y, model):
    df = pd.DataFrame(X)
    df['label'] = np.array(y)

    for i in range(k):
        train, test = get_ith_fold(k, df, i)
        train_x, train_y = get_xy(train)
        test_x, test_y = get_xy(test)
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        tp, fp, fn, tn = get_measures(test_y, pred)
        print(tp , fp, fn, tn)


def get_ith_fold(k, df, i):
    n = len(df)
    start = int(np.floor(n * i / k))
    end = int(np.floor(n * (i + 1) / k))
    test = df.iloc[start:end, :]
    train = df.drop(test.index)
    return train, test


if __name__ == "__main__":
    pass