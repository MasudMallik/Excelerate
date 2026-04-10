from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

cross=KFold(n_splits=5,random_state=42,shuffle=True)
def scores(y_test,y_pred):
    r2=r2_score(y_test,y_pred)
   
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    return r2,mae,mse

def linear_regression_model(x_train,x_test,y_train,y_test):
    linear_model=LinearRegression()
    linear_model.fit(x_train,y_train)
    y_pred=linear_model.predict(x_test)
    r2,mae,mse=scores(y_test,y_pred)
    return r2,mae,mse,linear_model

def decisiontree_regression(x_train,x_test,y_train,y_test):
    param={
        "criterion":["squared_error","absolute_error"],
        "max_depth":[5,10,15,20,30]
        }
    decision_tree_model=DecisionTreeRegressor(random_state=42)
    grid=GridSearchCV(
        estimator=decision_tree_model,
        cv=cross,
        scoring="r2",
        param_grid=param
    )
    grid.fit(x_train,y_train)
    print(grid.best_params_)
    y_pred=grid.predict(x_test)
    r2,mae,mse=scores(y_test,y_pred)
    return r2,mae,mse,grid.best_params_,grid.best_estimator_

def ridge_regression(x_train,x_test,y_train,y_test):
    param={
        "alpha":[0.2,0.4,0.7,0.9,1,2,2.6,4]
    }
    rid=Ridge(random_state=42)
    grid=GridSearchCV(
        estimator=rid,
        cv=cross,
        param_grid=param,
        scoring="r2"
    )
    grid.fit(x_train,y_train)
    y_pred=grid.predict(x_test)
    r2,mae,mse=scores(y_test,y_pred)
    return r2,mae,mse,grid.best_params_,grid.best_estimator_

def lasso_regression(x_train,x_test,y_train,y_test):
    param={
        "alpha":[0.2,0.4,0.7,0.9,1,2,2.6,4]
    }
    lass=Lasso(random_state=42)
    grid=GridSearchCV(
        estimator=lass,
        cv=cross,
        param_grid=param,
        scoring="r2"
    )
    grid.fit(x_train,y_train)
    y_pred=grid.predict(x_test)
    r2,mae,mse=scores(y_test,y_pred)
    return r2,mae,mse,grid.best_params_,grid.best_estimator_ 

def random_forest_regression(x_train,x_test,y_train,y_test):
    param={
        "criterion":["squared_error","absolute_error"],
        "max_depth":[5,10,15,20,25],
        "n_estimators":[10,20,30]
    }
    rf=RandomForestRegressor(random_state=42)
    grid=RandomizedSearchCV(
        estimator=rf,
        cv=cross,
        param_distributions=param,
        scoring="r2",
        n_iter=15
    )
    grid.fit(x_train,y_train)
    y_pred=grid.predict(x_test)
    r2,mae,mse=scores(y_test,y_pred)
    return r2,mae,mse,grid.best_params_,grid.best_estimator_

def svm_regression(x_train,x_test,y_train,y_test):
    param={
        "epsilon":[0.1,0.01,0.4,0.7],
        "C":[0.5,0.9,1,2,5]
    }
    sr=SVR()
    grid=GridSearchCV(
        estimator=sr,
        param_grid=param,
        scoring="r2",
        cv=cross
    )
    grid.fit(x_train,y_train)
    y_pred=grid.predict(x_test)
    r2,mae,mse=scores(y_test,y_pred)
    return r2,mae,mse,grid.best_params_,grid.best_estimator_

def xgboost_regression(x_train,x_test,y_train,y_test):
    xg=XGBRegressor()
    xg.fit(x_train,y_train)
    y_pred=xg.predict(x_test)
    r2,mae,mse=scores(y_test,y_pred)
    return r2,mae,mse,xg
