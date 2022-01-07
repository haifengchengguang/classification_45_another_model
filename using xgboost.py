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
os.environ["PATH"] += os.pathsep + r'D:\Program Files\Graphviz\bin'
# load data
dataset = loadtxt(r'pieces1.csv', delimiter=",")
# split data into X and y
X = dataset[1:, :-1]
Y = dataset[1:, -1]

seed = 27
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# train_set = xgb.DMatrix(X_train, label=y_train)
## 可视化测试集的loss
model = XGBClassifier(
    booster='gbtree',
    learning_rate=0.1,
    max_depth=8,
    n_estimators=70,
    reg_lambda=1,
    seed=27,
)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=['logloss','auc','error'], eval_set=eval_set, verbose=True)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#save model
model.save_model('model.xgb')
print("hello")
"""
 对每个特征的重要性进行分析
"""
from xgboost import plot_importance
from matplotlib import pyplot

model.fit(X, Y)

plot_importance(model, max_num_features=40)
pyplot.show()



