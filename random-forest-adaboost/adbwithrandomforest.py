
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


## Now back to work................

X_train=X_train[:,[0,1,2,3,4,7,8,9,10,11]]
X_test =X_test[:,[0,1,2,3,4,7,8,9,10,11]]

##below is a testing whether we can remove location fieldss longs and lats from the model... result!!!!.not worked as assumed
#X_train=X_train[:,[0,1,2,3,4,11]]
#X_test =X_test[:,[0,1,2,3,4,11]]




from sklearn.ensemble import RandomForestClassifier
#classifier = LogisticRegression(random_state=0)
#classifier.fit(X_train,Y_train)

#Use of BaggingClassifier......
#from sklearn.ensemble import BaggingClassifier
#bagging = BaggingClassifier(base_estimator = RandomForestClassifier(),n_estimators=3000,bootstrap=True)
#bagging =bagging.fit(X_train,Y_train)
#y_preds=bagging.predict(X_test)



#Use of Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
adbclassifier = AdaBoostClassifier(RandomForestClassifier(),n_estimators=200)
adbclassifier= adbclassifier.fit(X_train,Y_train)
y_preds=adbclassifier.predict(X_test)




#y_preds = classifier.predict(X_test)
Y_pred_dataframe = pd.DataFrame(y_preds)
Y_test_ids_dataframe = pd.DataFrame(Test_ids)

submission =pd.concat([Y_test_ids_dataframe,Y_pred_dataframe],axis=1)
submission =submission.rename(columns={"tiripid":"tripid"})
submission =submission.rename(columns={0:"prediction"})

submission.to_csv('submission.csv',index=False)




