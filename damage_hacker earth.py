import pandas as pd
import numpy as np

train=pd.read_csv(r"C:\Users\TEJA\Downloads\Dataset\train.csv")
train.head()
target=train["damage_grade"]
train.drop(["damage_grade"],1,inplace=True)
target=target.replace({'Grade 1':1,'Grade 2':2,'Grade 3':3,'Grade 4':4,'Grade 5':5})
target

train.loc[(train["vdcmun_id"]>=701) &(train["vdcmun_id"]<1811.25),"vdcmun_id"]=1
train.loc[(train["vdcmun_id"]>1811.25) &(train["vdcmun_id"]<2921.5),"vdcmun_id"]=2
train.loc[(train["vdcmun_id"]>2921.5) &(train["vdcmun_id"]<4031.75),"vdcmun_id"]=3
train.loc[(train["vdcmun_id"]>4031.75) &(train["vdcmun_id"]<=5142),"vdcmun_id"]=4
train.info()
train["risk"]=train["has_geotechnical_risk"]+train["has_geotechnical_risk_fault_crack"]+train["has_geotechnical_risk_flood"]+train["has_geotechnical_risk_land_settlement"]+train["has_geotechnical_risk_landslide"]+train["has_geotechnical_risk_liquefaction"]+train["has_geotechnical_risk_other"]+train["has_geotechnical_risk_rock_fall"]
train["risk"].replace({0.0:0,1.0:0,2.0:0,3.0:1,4.0:1,5.0:2,6.0:2,7.0:2})
train["has_repair_started"]=train["has_repair_started"].fillna(0)
train["has_repair_started"].unique()
train.head(5)
train.drop(["has_geotechnical_risk","has_geotechnical_risk_fault_crack","has_geotechnical_risk_flood","has_geotechnical_risk_land_settlement","has_geotechnical_risk_landslide","has_geotechnical_risk_liquefaction","has_geotechnical_risk_other","has_geotechnical_risk_rock_fall"],1,inplace=True)
train.head()
z=pd.get_dummies(train["vdcmun_id"])
train=pd.concat([train,z],axis=1)

train["area_assesed"]=train["area_assesed"].replace({"Interior":1,"Exterior":1,"Not able to inspect":2,"Both":2,"Building removed":3,})
train.drop(["building_id"],1,inplace=True)
train.head()
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,target,random_state=0)

dtrain=xgb.DMatrix(x_train,y_train)
dtest=xgb.DMatrix(x_test,y_test)

import numpy as np
# "Learn" the mean from the training data
mean_train = np.mean(y_train)

# Get predictions on the test set
baseline_predictions = np.ones(y_test.shape) * mean_train

# Compute MAE
mae_baseline = mean_absolute_error(y_test, baseline_predictions)

print("Baseline MAE is {:.2f}".format(mae_baseline))

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    'objective':'reg:linear',
}

params['eval_metric'] = "mae"
num_boost_round = 999
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))







X_train,X_test,Y_train,Y_test=train_test_split(train,target,test_size=0.4)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_leaf=2,verbose=0,n_jobs=-1)
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
np.shape(Y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(pred,Y_test)
confusion_matrix
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
clf1=GradientBoostingClassifier()
clf1.fit(X_train,Y_train)
pred1=clf1.predict(X_test)
pred1

confusion_matrix2=confusion_matrix(pred1,Y_test)
confusion_matrix2






test=pd.read_csv(r"C:\Users\TEJA\Downloads\Dataset\test.csv")
ts=test["building_id"]
test.loc[(test["vdcmun_id"]>=701) &(test["vdcmun_id"]<1811.25),"vdcmun_id"]=1
test.loc[(test["vdcmun_id"]>1811.25) &(test["vdcmun_id"]<2921.5),"vdcmun_id"]=2
test.loc[(test["vdcmun_id"]>2921.5) &(test["vdcmun_id"]<4031.75),"vdcmun_id"]=3
test.loc[(test["vdcmun_id"]>4031.75) &(test["vdcmun_id"]<=5142),"vdcmun_id"]=4
test.info()
test["risk"]=test["has_geotechnical_risk"]+test["has_geotechnical_risk_fault_crack"]+test["has_geotechnical_risk_flood"]+test["has_geotechnical_risk_land_settlement"]+test["has_geotechnical_risk_landslide"]+test["has_geotechnical_risk_liquefaction"]+test["has_geotechnical_risk_other"]+test["has_geotechnical_risk_rock_fall"]
test["risk"].replace({0.0:0,1.0:0,2.0:0,3.0:1,4.0:1,5.0:2,6.0:2,7.0:2})
test["has_repair_started"]=test["has_repair_started"].fillna(0)
test["has_repair_started"].unique()
test.head(5)
test.drop(["has_geotechnical_risk","has_geotechnical_risk_fault_crack","has_geotechnical_risk_flood","has_geotechnical_risk_land_settlement","has_geotechnical_risk_landslide","has_geotechnical_risk_liquefaction","has_geotechnical_risk_other","has_geotechnical_risk_rock_fall"],1,inplace=True)
test.head()
z=pd.get_dummies(test["vdcmun_id"])
test=pd.concat([test,z],axis=1)




test["area_assesed"]=test["area_assesed"].replace({"Interior":1,"Exterior":1,"Not able to inspect":2,"Both":2,"Building removed":3,})
test.drop(["building_id"],1,inplace=True)
test.head()
predict=clf1.predict(test)
predict
predict=list(predict)
predict

x=[]
type(x)



for i in predict:
    i=a+str(i)
    x.append(i)
    
    
y=np.array(x)
y
np.shape(x)
predict
y












data2={"building_id":ts,"damage_grade":y}
sub=pd.DataFrame(data2)
sub.to_csv(r"C:\Users\TEJA\Downloads\Untitled spreadsheet - Sheet1.csv",index=False)
result=pd.read_csv(r"C:\Users\TEJA\Downloads\Untitled spreadsheet - Sheet1.csv")
result


import xgboost as xgb
params={     
    "learning_rate":0.1,
    "subsample":0.8,
    "colsample_bytree": 0.8,
        'eval_metric':'auc',
    "max_depth":6,
    'silent':1,
    'nthread':3,
#     'num_class':2
        
       }
params
