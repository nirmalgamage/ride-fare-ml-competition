
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
imputer_mode =Imputer(missing_values ='NaN', strategy ='most_frequent',axis =0)
imputer_mode.fit(X_train[:,[0]])
X_train[:,[0]]=imputer_mode.transform(X_train[:,[0]])


imputer = Imputer(missing_values ='NaN', strategy ='most_frequent',axis =0)
imputer.fit(X_train[:,[1,2,3,4,11]])
X_train[:,[1,2,3,4,11]] = imputer.transform(X_train[:,[1,2,3,4,11]])
df = pd.DataFrame(X_train)

##DO the same for Test set
test_dataset =pd.read_csv('test.csv')

X_test = test_dataset.iloc[:, 1:13].values
df_x_test  = pd.DataFrame(X_test)

Test_ids = test_dataset.iloc[:,0:1]

imputer_mode =Imputer(missing_values ='NaN', strategy ='most_frequent',axis =0)
imputer_mode.fit(X_test[:,[0]])
X_test[:,[0]]=imputer_mode.transform(X_test[:,[0]])


imputer = Imputer(missing_values ='NaN', strategy ='most_frequent',axis =0)
imputer.fit(X_test[:,[1,2,3,4,11]])
X_test[:,[1,2,3,4,11]] = imputer.transform(X_test[:,[1,2,3,4,11]])


## Now back to work................

from sklearn.preprocessing import StandardScaler



##This is for adding distance feature...
from math import cos, asin, sqrt, pi

def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...



i=0
while i< len(X_train):
    lat1 = X_train[:,7][i]
    lon1  = X_train[:,8][i]
    lat2 = X_train[:,9][i]
    lon2  = X_train[:,10][i]
    dist = distance(lat1,lon1,lat2,lon2)
    X_train[:,5][i] = dist*1000
    i=i+1


j=0
while j< len(X_test):
    lat1 = X_test[:,7][j]
    lon1  = X_test[:,8][j]
    lat2 = X_test[:,9][j]
    lon2  = X_test[:,10][j]
    dist = distance(lat1,lon1,lat2,lon2)
    X_test[:,5][j] = dist*1000
    j=j+1


#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train[:,[0,1,2,3,4,7,8,9,10,11]])
#X_test = scaler.fit_transform(X_test[:,[0,1,2,3,4,7,8,9,10,11]])
X_train=X_train[:,[0,1,2,3,4,5,11]]
X_test =X_test[:,[0,1,2,3,4,5,11]]


#from sklearn.svm import SVC
#
#classifier = SVC(kernel='sigmoid', random_state =0)
#classifier.fit(X_train,Y_train)


#Import test values

from xgboost import XGBClassifier
classifier= XGBClassifier()
classifier.fit(X_train,Y_train)

y_preds = classifier.predict(X_test)
Y_pred_dataframe = pd.DataFrame(y_preds)
Y_test_ids_dataframe = pd.DataFrame(Test_ids)

submission =pd.concat([Y_test_ids_dataframe,Y_pred_dataframe],axis=1)
submission =submission.rename(columns={"tiripid":"tripid"})
submission =submission.rename(columns={0:"prediction"})

submission.to_csv('submission.csv',index=False)


