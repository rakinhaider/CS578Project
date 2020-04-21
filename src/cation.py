import pandas as pd
from src.util import get_xy, get_measures
from src.confusion_matrix import ConfusionMatrix
from src.constants import p_block,s_block,d_block,f_block
from itertools import combinations


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
        tp, fp, fn, tn = get_measures(test_y, pred)
        conf=ConfusionMatrix(tp, fp, fn, tn)
        S.append({"Block_1":block_1,
                "Block_2":block_2,
                "Accuracy":round(conf.get_accuracy(),3)})

    return S