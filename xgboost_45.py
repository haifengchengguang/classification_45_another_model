import json

from numpy import loadtxt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import DMatrix
import xgboost as xgb
import numpy as np
#import matplotlib.pyplot as plt
import os
#from keras.utils import np_utils
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

os.environ["PATH"] += os.pathsep + r'D:\Program Files\Graphviz\bin'
# load data
dataset = pd.read_csv(r'C:\Users\Administrator\Desktop\full_match_rizjhkw1_id_ra_dec_distance_extinc_1009_45_copy1.csv')
# split data into X and y
X = dataset.values[:, 22:67]

Y = dataset.values[:, 70]
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
classes_1=encoder.classes_
classes_1_list=classes_1.tolist()

subclass_amount=len(classes_1_list)
print(subclass_amount)
a=list(range(subclass_amount))

d=zip(a,classes_1)
c=dict(d)
print(c)
json_str = json.dumps(c)
with open('class_indices_21.json', 'w') as json_file:
    json_file.write(json_str)


seed = 27
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=test_size, random_state=seed)
# train_set = xgb.DMatrix(X_train, label=y_train)
num_class=21

print(Counter(Y))
print(X.shape)
print(y_train.shape)
## 可视化测试集的loss
model = XGBClassifier(
    objective='multi:softprob',
    num_class=num_class,
    booster='gbtree',
    learning_rate=0.1,
    max_depth=8,
    n_estimators=70,
    reg_lambda=1,
    seed=27,

)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=['mlogloss','auc','merror'], eval_set=eval_set, verbose=True)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#save model
model.save_model('my45_model.xgb')
print("hello")
"""
 对每个特征的重要性进行分析
"""
from xgboost import plot_importance
from matplotlib import pyplot

model.fit(X, Y)

plot_importance(model, max_num_features=10)
pyplot.show()



