from random_forest import RandomForestClassifier
from metrics import accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import openml
from openml import tasks
from SMOTE import smote
import torch

def encod_dict(X):
    dic={}
    values=X.unique()
    num=0
    for value in values:
        dic.update({value: num})
        num+=1
    
    return dic

#This function is currently not doing anything but it could be helpfull
#It takes the values of classes and and one hot-encodes
#   Y=[0,1,2]
#   transforms to:
#   Y=[[1,0,0],[0,1,0],[0,0,1]]
def get_classes_binarized(y, n_class):
    y_bin=[]
    for row in y:
        row_bin=[]
        for i in range(n_class):
            if(i==row):
                row_bin.append(1)
            else:
                row_bin.append(0)
        y_bin.append(row_bin)
    return y_bin


dataset=openml.datasets.get_dataset(11)
X,Y,_,_=dataset.get_data(target=dataset.default_target_attribute , dataset_format="dataframe")
map_dict=encod_dict(Y)
Y.replace(map_dict, inplace=True)
# Aplicar o SMOTE aos conjuntos de dados de treinamento
        
X_train, X_test, Y_train, Y_test =train_test_split(X,Y)

#Criar uma instância do SMOTE com os parâmetros desejados
smote_i = smote(distance='euclidian', dims=X.shape[1], k=5)# dimns is number of atributes that are not the target

print(f"dataset antes smote {X_train.shape}")
model = RandomForestClassifier(max_depth=15, smote=smote_i , smote_type="binary")
model.fit(X_train,Y_train)
predictions=model.predict(X_test)

#this is currently working however I am not sure which parameters are the best for our case 
print(f"Area under the curve: {roc_auc_score(Y_test,predictions, average='weighted', multi_class='ovr')}")
print(f"Area under the curve per classe: {roc_auc_score(Y_test,predictions, average=None, multi_class='ovr')}")
print(f"Accuracy: {accuracy(Y_test,predictions)}")


