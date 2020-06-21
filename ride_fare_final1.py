# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 08:56:33 2020

@author: Pramith
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:12:39 2020

@author: Pramith
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:04:24 2020

@author: Pramith
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:59:41 2020

@author: Pramith
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')


X_train = dataset.iloc[:, 1:13].values
Y_train = dataset.iloc[:,[13]].values
df_y_train= pd.DataFrame(Y_train)
df_x_train  = pd.DataFrame(X_train)

### binarize Y train column 
Y_train =np.where(Y_train == 'correct',1,0)

## handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values ='NaN', strategy ='most_frequent',axis =0)
imputer_mean = Imputer(missing_values ='NaN', strategy ='mean',axis =0)
imputer_mean.fit(X_train[:,[1,2,3,4,11]])
X_train[:,[1,2,3,4,11]] = imputer_mean.transform(X_train[:,[1,2,3,4,11]])

imputer.fit(X_train[:,[0]])
X_train[:,[0]] = imputer.transform(X_train[:,[0]])
df = pd.DataFrame(X_train)


##try to remove raws with NaN values.......
#df_x_train=df_x_train.dropna()
#X_train=df_x_train.to_numpy()
##DO the same for Test set
test_dataset =pd.read_csv('test.csv')

X_test = test_dataset.iloc[:, 1:13].values
df_x_test  = pd.DataFrame(X_test)

Test_ids = test_dataset.iloc[:,0:1]
imputer = Imputer(missing_values ='NaN', strategy ='most_frequent',axis =0)
imputer_mean = Imputer(missing_values ='NaN', strategy ='mean',axis =0)
imputer_mean.fit(X_test[:,[1,2,3,4,11]])
X_test[:,[1,2,3,4,11]] =imputer_mean.transform(X_test[:,[1,2,3,4,11]])
imputer.fit(X_test[:,[0]])
X_test[:,[0]] = imputer.transform(X_test[:,[0]])





#function to get actual distance between two points
from math import cos, asin, sqrt, pi

def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...

# end of the function

#calculating distance for training dataset and getting (duration-waiting0/fare
#X_train=np.delete(X_train,4,0)
X_train=np.append(X_train,[[1]]*len(X_train),axis=1)
X_train=np.append(X_train,[[1]]*len(X_train),axis=1)
X_train=np.append(X_train,[[1]]*len(X_train),axis=1)
X_train=np.append(X_train,[[1]]*len(X_train),axis=1)


X_test=np.append(X_test,[[1]]*len(X_test),axis=1)
X_test=np.append(X_test,[[1]]*len(X_test),axis=1)
X_test=np.append(X_test,[[1]]*len(X_test),axis=1)
X_test=np.append(X_test,[[1]]*len(X_test),axis=1)


i=0
while i< len(X_train):
    lat1 = X_train[:,7][i]
    lon1  = X_train[:,8][i]
    lat2 = X_train[:,9][i]
    lon2  = X_train[:,10][i]
    dist = distance(lat1,lon1,lat2,lon2)
    X_train[:,5][i] = dist*1000
    
    #
    orig_fair = float(X_train[:,11][i])-float(X_train[:,0][i])#-float(X_train[:,3][i])
    orig_duration = float(X_train[:,1][i])-float(X_train[:,2][i])-float(X_train[:,4][i])
    X_train[:,12][i]=orig_fair
    X_train[:,13][i]=orig_duration
    
    #
    feature=0
    if float(X_train[:,11][i])==0.0:
        feature = (float(X_train[:,1][i]) -float(X_train[:,2][i]))/float(np.mean(X_train[:,11][0:i]))
    else:
        feature = (float(X_train[:,1][i]) -float(X_train[:,2][i]))/float(X_train[:,11][i])
    X_train[:,14][i]=feature
    
    fare_per_dist = 0
    if dist==0:
        fare_per_dist = orig_fair/float(np.mean(X_train[:,5][0:i]))
    else:
        fare_per_dist = orig_fair/float(dist)
        
    X_train[:,15][i] = fare_per_dist
    #feature =float(X_train[:,11][i])/(float(X_train[:,1][i]) -float(X_train[:,2][i]))
    #X_train[:,6][i] = feature
    i=i+1


j=0
while j< len(X_test):
    lat1 = X_test[:,7][j]
    lon1  = X_test[:,8][j]
    lat2 = X_test[:,9][j]
    lon2  = X_test[:,10][j]
    dist = distance(lat1,lon1,lat2,lon2)
    X_test[:,5][j] = dist*1000
    
    #
    orig_fair = float(X_test[:,11][j])-float(X_test[:,0][j])#-float(X_test[:,3][j])
    orig_duration = float(X_test[:,1][j])-float(X_test[:,2][j])-float(X_test[:,4][j])
    X_test[:,12][j]=orig_fair
    X_test[:,13][j]=orig_duration
    #feature =float(X_test[:,11][i])/(float(X_test[:,1][i]) -float(X_test[:,2][i]))
    #X_test[:,6][i] = feature
    feature=0
    if float(X_test[:,11][j])==0.0:
        feature = (float(X_test[:,1][j]) -float(X_test[:,2][j]))/float(np.mean(X_test[:,11][0:j]))
    else:
        feature = (float(X_test[:,1][j]) -float(X_test[:,2][j]))/float(X_test[:,11][j])
    X_test[:,14][j]=feature
    
    
    
    fare_per_dist = 0
    if dist==0:
        fare_per_dist = orig_fair/float(np.mean(X_test[:,5][0:j]))
    else:
        fare_per_dist = orig_fair/float(dist)
    
    X_test[:,15][j] = fare_per_dist
    j=j+1



## Now back to work................
X_train=X_train[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15]]
X_test =X_test[:,[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15]]


##below is a testing whether we can remove location fieldss longs and lats from the model... result!!!!.not worked as assumed
#X_train=X_train[:,[0,1,2,3,4,11]]
#X_test =X_test[:,[0,1,2,3,4,11]]



from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#classifier = LogisticRegression(random_state=0)
#classifier.fit(X_train,Y_train)

#Use of BaggingClassifier......
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(base_estimator = DecisionTreeClassifier(),n_estimators=200,bootstrap=True)
bagging =bagging.fit(X_train,Y_train)
y_preds=bagging.predict(X_test)

#y_preds = classifier.predict(X_test)
Y_pred_dataframe = pd.DataFrame(y_preds)
Y_test_ids_dataframe = pd.DataFrame(Test_ids)

submission =pd.concat([Y_test_ids_dataframe,Y_pred_dataframe],axis=1)
submission =submission.rename(columns={"tiripid":"tripid"})
submission =submission.rename(columns={0:"prediction"})

submission.to_csv('submission.csv',index=False)



















