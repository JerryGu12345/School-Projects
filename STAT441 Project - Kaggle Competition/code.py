# the following code is my contributions to my team in the Kaggle competition
# this code is run to obtain estimates which are submitted and obtained a cross-entropy score of 1.20897
# https://www.kaggle.com/competitions/stat-441-w2022-data-challenge

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
from sklearn import ensemble
import time

# import os
# for dirname, _, filenames in os.walk(folder):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

folder = ''

# We get the necessary data
train_data = pd.read_csv(folder+'train.csv') # training data
test_data = pd.read_csv(folder+'test.csv') # test data
sample_sub = pd.read_csv(folder+'sample_submission.csv')

train_data = pd.read_csv(folder+'train.csv').drop(columns=["uniqueid","year","personid"])
X=train_data.drop(columns="health").to_numpy()
y=train_data["health"]





###### random forests ######
print(round(0.6 * train_data.shape[0]))
train_data_clean_1 = train_data.dropna(axis = 1, thresh = 10440)
train_data_clean_1.shape
  
for name in train_data_clean_1.columns:
    train_data_clean_1.loc[train_data_clean_1[name].isnull(), name] = train_data_clean_1[name].mode()[0]

X=train_data_clean_1.drop(columns="health").to_numpy()
y=train_data_clean_1["health"]

start=time.time()
params={'n_estimators': [300], 'max_depth': [300],'max_leaf_nodes': [300], 'max_features': [100], 'bootstrap':[False], 'min_samples_split':[0.01,0.1,1]}
np.random.seed(441)
M=sk.ensemble.RandomForestClassifier(criterion="entropy")
M=sk.model_selection.GridSearchCV(M,params)
M.fit(X,y)
end=time.time()
start-end
#M.best_estimator_.predict_proba(X)
M.best_score_


train_data_clean_2 = train_data.dropna(axis = 1, thresh = 100)
train_data_clean_2.shape

for name in train_data_clean_2.columns:
    train_data_clean_2.loc[train_data_clean_2[name].isnull(), name] = train_data_clean_2[name].mode()[0]

X=train_data_clean_2.drop(columns="health")
y=train_data_clean_2["health"]
X.to_csv("X_train")
y.to_csv("y_train")

start=time.time()
params={'n_estimators': [3,10,30,100,300], 'max_depth': [3,10,30,100,300],'max_leaf_nodes': [3,10,30,100,300], 'max_features': [3,10,30,100,300]}
np.random.seed(441)
M=sk.ensemble.RandomForestClassifier(criterion="entropy")
M=sk.model_selection.GridSearchCV(M,params)
M.fit(X,y)
end=time.time()
start-end
M.best_estimator_.predict_proba(X)
M.best_score_

test_data_clean_2=test_data[train_data_clean_2.drop(columns="health").columns]
for name in test_data_clean_2.columns:
    test_data_clean_2.loc[test_data_clean_2[name].isnull(), name] = test_data_clean_2[name].mode()[0]

#,index=["uniqueid","p1","p2","p3","p4","p5"]

X_test=test_data_clean_2.to_numpy()
ans=M.best_estimator_.predict_proba(X_test) #use X from test
ans=pd.DataFrame(ans).rename(index=test_data["uniqueid"])
ans.to_csv('ans.csv')


###### neural ######
X=train_data_clean_1.drop(columns="health").to_numpy()
y=train_data_clean_1["health"]

start=time.time()
params={'hidden_layer_sizes':[(10),(100)],'activation':["logistic","tanh","relu"],'solver':["sgd"]}
#[(10,1),(100,1),(10,10),(10,100),(100,10),(100,100)]
np.random.seed(441)
M=sk.neural_network.MLPClassifier()
M=sk.model_selection.GridSearchCV(M,params,scoring="accuracy")
M.fit(X,y)
end=time.time()
start-end
#M.best_estimator_.predict_proba(X)
M.best_score_

for hidden_layer_sizes in [(132),(671,26)]:
  for activation in ["logistic","tanh","relu"]:
    M=sk.neural_network.MLPClassifier(hidden_layer_sizes,activation)
    print("Score for layers=%s, activation=%s is %s" % (hidden_layer_sizes,activation,np.mean([M.fit(X[train],y[train]).score(X[test],y[test])
      for train,test in sk.model_selection.KFold().split(train_data)])))

'''
train_data_clean_3 = train_data.dropna(axis = 1, thresh = 100)
train_data_clean_3.shape

X_train=train_data_clean_3[train_data_clean_3["x3"].isnull()==False].drop(columns="x3").to_numpy()
X_test=train_data_clean_3[train_data_clean_3["x3"].isnull()].drop(columns="x3").to_numpy()
y=train_data_clean_3["x3"].dropna()

###### Linear Regression ######
M=sk.linear_model.LinearRegression()
M.fit(X,y)
'''
