from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
import json

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

clf = GaussianNB()
#拟合数据
clf.fit(X_train, y_train)