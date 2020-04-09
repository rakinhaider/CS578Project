from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import src.constants as constants


def train_test_split(df,fraction):
    test = df.sample(frac=fraction)
    train = df.drop(index=test.index)

    return train, test


def get_model(model_type, **kwargs):
    if model_type == constants.NAIVE_BAYES:
        model = GaussianNB()
    elif model_type == constants.LOG_REGRESSION:
        reg = kwargs.get('regularizer', 'l2')
        c = kwargs.get('reg_param', 0.8)
        if reg == 'l2':
            solver = 'lbfgs'
        else:
            solver = 'saga'
        model = LogisticRegression(penalty=reg,
                                   C=c,
                                   solver=solver,
                                   max_iter=10000)
    
    elif model_type == constants.SVM:
        c1 = kwargs.get('reg_param', 0.8)
        kern = kwargs.get('Kernel','rbf')
        model=SVC(C=c1,kernel=kern)
        
        
    else:
        return None

    return model


def get_xy(df):
    return df[df.columns[:-1]], df[df.columns[-1]]


def get_measures(orig, pred):

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

    return tp, fp, fn, tn
