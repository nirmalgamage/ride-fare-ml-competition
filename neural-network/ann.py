
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

#calculating distance for training dataset 
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



## Now back to work................
##DO it with longs and lats.....
#X_train=X_train[:,[0,1,2,3,4,5,7,8,9,10,11]]
#X_test =X_test[:,[0,1,2,3,4,5,7,8,9,10,11]]

#do it without the longs and lats....
X_train=X_train[:,[0,1,2,3,4,5,11]]
X_test =X_test[:,[0,1,2,3,4,5,11]]

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.fit_transform(X_test)
##below is a testing whether we can remove location fieldss longs and lats from the model... result!!!!.not worked as assumed
#X_train=X_train[:,[0,1,2,3,4,11]]
#X_test =X_test[:,[0,1,2,3,4,11]]
# neural NEtwork ..........
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics

classifier = Sequential()
classifier.add(Dense(output_dim =4 , init ='uniform',activation ='relu',input_dim=7))
classifier.add(Dense(output_dim =4 , init ='uniform',activation ='relu'))
classifier.add(Dense(output_dim =1 , init ='uniform',activation ='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

classifier.fit(X_train,Y_train,batch_size=10, epochs =100)
y_preds =classifier.predict(X_test)
y_preds = np.where(y_preds>0.5,1,0)





#y_preds = classifier.predict(X_test)
Y_pred_dataframe = pd.DataFrame(y_preds)
Y_test_ids_dataframe = pd.DataFrame(Test_ids)

submission =pd.concat([Y_test_ids_dataframe,Y_pred_dataframe],axis=1)
submission =submission.rename(columns={"tiripid":"tripid"})
submission =submission.rename(columns={0:"prediction"})

submission.to_csv('submission.csv',index=False)












