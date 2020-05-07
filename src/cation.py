import pandas as pd
from src.util import get_xy, get_confusion_mat, get_model
from src.constants import p_block,s_block,d_block,f_block
from itertools import combinations
import matplotlib.pyplot as plt
import src.constants as constants

cation_type={"s":s_block,
             "p":p_block,
             "d":d_block,
             "f":f_block}


def get_cation_confusion_matrix(test_data,model):
    test = pd.DataFrame(test_data)
    S=[]
    list_1=list(combinations(cation_type,2))
    list_2=[('s','s'),('p','p'),('d','d'),('f','f')]
    list_final=list_1+list_2
    for comb in list_final:
        block_1=comb[0]
        block_2=comb[1]
        cation_data=test.loc[(test['atomic_num_1'].isin(cation_type[block_1])) & (test['atomic_num_2'].isin(cation_type[block_2]))]
        test_x,test_y=get_xy(cation_data)
        pred = model.predict(test_x)
        conf=get_confusion_mat(test_y,pred)
        S.append({"Block_1":block_1,
                "Block_2":block_2,
                "Accuracy":round(conf.get_accuracy(),3)})

    return S

def plot_cation(best_param,train_x,train_y,test):
    model_type=constants.SVM
    model = get_model(model_type,**best_param[model_type])
    model.max_iter=1000
    model.fit(train_x, train_y)
    S1=get_cation_confusion_matrix(test,model)
    S1_df=pd.DataFrame(S1)
    model_type=constants.NAIVE_BAYES
    model = get_model(model_type)
    model.max_iter=1000
    model.fit(train_x, train_y)
    S2=get_cation_confusion_matrix(test,model)
    S2_df=pd.DataFrame(S2)
    model_type=constants.LOG_REGRESSION
    model = get_model(model_type,**best_param[model_type])
    model.max_iter=1000
    model.fit(train_x, train_y)
    S3=get_cation_confusion_matrix(test,model)
    S3_df=pd.DataFrame(S3)
    S1_df["Block"]=S1_df["Block_1"]+S1_df["Block_2"]
    S2_df["Block"]=S2_df["Block_1"]+S2_df["Block_2"]
    S3_df["Block"]=S3_df["Block_1"]+S3_df["Block_2"]
    plt.plot(S1_df["Block"],S1_df["Accuracy"],label='SVM')
    plt.plot(S2_df["Block"],S2_df["Accuracy"],label='Naive Bayes')
    plt.plot(S3_df["Block"],S3_df["Accuracy"],label='Logistic regression')
    plt.legend(loc='lower left')
    plt.xlabel('Block')
    plt.ylabel('Accuracy')
    plt.savefig('xx.png', format='png')
    plt.show()
    