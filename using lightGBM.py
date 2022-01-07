from numpy import loadtxt
import lightgbm as lgb
#import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import os
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler

os.environ["PATH"] += os.pathsep + r'D:\Program Files\Graphviz\bin'

# load data
dataset = loadtxt(r'pieces1.csv', delimiter=",")
# split data into X and y
X = dataset[1:, :-1]
Y = dataset[1:, -1]
x = dataset[0:1, :-1]
x = np.transpose(x)

# dtrain是我的训练数据（自变量矩阵是X，分类结果即因变量矩阵是y，特征字段重命名成我设置的features列表）
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=27)

# 建成lgb特征的数据集格式
lgb_train = lgb.Dataset(X_train, label=y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
# lgb_train = lgb.DMatrix(X_train, label=y_train, feature_names=x)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据
# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'auc'},  # 评估函数
    'max_depth': 7,
    'num_leaves': 80,  # 叶子节点数
    'learning_rate': 0.1,  # 学习速率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)  # 训练数据需要参数列表和数据集
print('Save model...')
gbm.save_model('model.txt')  # 训练后保存模型到文件
print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
# 评估模型
print('测试集前20个样本的类别为:', y_test[:20].tolist())
print('预测的类别为:', y_pred[:20])

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差
gbm_score = accuracy_score(y_test, y_pred.round())  # 准确率
print('gbm_score:', gbm_score)
# feature importances
lgb.plot_importance(gbm, max_num_features=40)
plt.title("Featurertances")
plt.show()

fig,ax = plt.subplots()
fig.set_size_inches(600,300)
lgb.plot_tree(gbm,ax = ax)
#plt.show()

