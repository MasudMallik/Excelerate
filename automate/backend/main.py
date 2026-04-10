from fastapi import FastAPI,Request,File,BackgroundTasks,UploadFile
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
import matplotlib.pyplot as plt
from Regression_models import linear_regression_model,decisiontree_regression,ridge_regression,xgboost_regression,lasso_regression,svm_regression,random_forest_regression

app=FastAPI(
    title="Excelerate",
    description="this is a machine learning project which take raw csv/excel file anad automatically preprocess and create muyltiple model for prediction or classification",
    version="0.0.0.1",
    summary="automatic ml model creation",
    contact={
        "ph-number":"0987654321",
        "email":"mas@123@gmail.com"
    },

)
global x_train,x_test,y_train,y_test,metrics
metrics={}

class MlModel:
    def __init__(self):
        pass
    def preprocess(self,df,input_label,output_label):
        after_drop=df.dropna()
        if len(after_drop)<900:
            x_char=df.select_dtypes(exclude="number")
            x_number=df.select_dtypes(include="number")
            if not x_char.empty:
                df[x_char.columns].fillna(df[x_char.columns].mode().iloc[0],inplace=True)
            if not x_number.empty:
                df[x_number.columns].fillna(np.mean(df[x_number.columns]),inplace=True)
        else:
            df.dropna(inplace=True)
        x=df[input_label]
        y=df[output_label]
        x_char=x.select_dtypes(exclude="number")
        x_num=x.select_dtypes(include="number")

        if not x_char.empty and not  x_num.empty:
            trans=ColumnTransformer(
                transformers=[
                    ("numeric_value",StandardScaler(),x_num.columns),
                    ("string",OneHotEncoder(handle_unknown="ignore"),x_char.columns)
                ],
                remainder="passthrough"
            )
            change=trans.fit_transform(x)
        elif not x_char.empty and  x_num.empty:
            on=OneHotEncoder(handle_unknown="ignore")
            change=on.fit_transform(x)
        elif not x_num.empty and x_char.empty:
            sd=StandardScaler()
            change=sd.fit_transform(x)
        return change,y

            


@app.get("/")
def home():
    return {"response":"hello world"}

@app.post("/prediction")
async def predict(request:Request,file:UploadFile =File(...)):
    global x_train,x_test,y_train,y_test
    # data=request.json()
    content= await file.read()
    if file.filename.endswith(".csv"):
        df=pd.read_csv(io.BytesIO(content))
    else:
        df=pd.read_excel(io.BytesIO(content))
    
    if  df.empty:
        return {"data":"please submit again your file, file not loaded"}
    else:
        model=MlModel()
        output_label=["Salary"]
        input_label=["YearsExperience"]
        after_preprocess_x,y=model.preprocess(df,input_label,output_label)
        x_train,x_test,y_train,y_test=train_test_split(after_preprocess_x,y,random_state=42,test_size=0.25)
        if x_train.any():
            return {"preprocess":"succesfully done"}
        else:
            return {"preprocess":"preprocess error"}
    
@app.get("/linear_regression")
async def linear(request:Request):
    global x_train,x_test,y_train,y_test,metrics
    r2,mse,mae,lin_model=linear_regression_model(x_train,x_test,y_train,y_test)
    metrics["linear_model"]={"r2":r2,"mae":mae,"mse":mse}
    return {"r2":r2,"mae":mae,"mse":mse}

@app.get("/decision_regression")
async def tree(request:Request):
    global x_train,x_test,y_train,y_test,metrics
    r2,mse,mae,params,dec_model=decisiontree_regression(x_train,x_test,y_train,y_test)
    metrics["decision tree"]={"r2":r2,"mae":mae,"mse":mse,"params":params}
    return {"r2":r2,"mae":mae,"mse":mse,"params":params}


@app.get("/Ridge_regression")
async def ridge(request:Request):
    global x_train,x_test,y_train,y_test,metrics
    r2,mse,mae,params,ridge_model=ridge_regression(x_train,x_test,y_train,y_test)
    metrics["Ridge"]={"r2":r2,"mae":mae,"mse":mse,"params":params}
    return {"r2":r2,"mae":mae,"mse":mse,"params":params}

@app.get("/lasso_regression")
async def lasso(request:Request):
    global x_train,x_test,y_train,y_test,metrics
    r2,mse,mae,params,lasso_model=lasso_regression(x_train,x_test,y_train,y_test)
    metrics["lasso"]={"r2":r2,"mae":mae,"mse":mse,"params":params}
    return {"r2":r2,"mae":mae,"mse":mse,"params":params}

@app.get("/Randomforest_regression")
async def randomforest(request:Request):
    global x_train,x_test,y_train,y_test,metrics
    r2,mse,mae,params,randomforest_model=random_forest_regression(x_train,x_test,y_train,y_test)
    metrics["Randomforest"]={"r2":r2,"mae":mae,"mse":mse,"params":params}
    return {"r2":r2,"mae":mae,"mse":mse,"params":params}

@app.get("/svm_regression")
async def svm(request:Request):
    global x_train,x_test,y_train,y_test,metrics
    r2,mse,mae,params,svm_model=svm_regression(x_train,x_test,y_train,y_test)
    metrics["Svm"]={"r2":r2,"mae":mae,"mse":mse,"params":params}
    return {"r2":r2,"mae":mae,"mse":mse,"params":params}

@app.get("/xgb_regression")
async def xgb(request:Request):
    global x_train,x_test,y_train,y_test,metrics
    r2,mse,mae,xgb_model=xgboost_regression(x_train,x_test,y_train,y_test)
    metrics["Svm"]={"r2":r2,"mae":mae,"mse":mse}
    return {"r2":r2,"mae":mae,"mse":mse}

@app.get("/det")
def getdata():

    global metrics
    import matplotlib.pyplot as plt

    
    models = list(metrics.keys())
    r2_scores = [metrics[m]["r2"] for m in models]
    mae_scores = [metrics[m]["mae"] for m in models]
    mse_scores = [metrics[m]["mse"] for m in models]

    # Plot R² comparison
    plt.figure(figsize=(10,6))
    plt.bar(models, r2_scores, color='skyblue')
    plt.title("Model Comparison - R² Score")
    plt.ylabel("R²")
    plt.xticks(rotation=45)
    plt.show()

    # Plot MAE comparison
    plt.figure(figsize=(10,6))
    plt.bar(models, mae_scores, color='salmon')
    plt.title("Model Comparison - MAE")
    plt.ylabel("Mean Absolute Error")
    plt.xticks(rotation=45)
    plt.show()

    # Plot MSE comparison
    plt.figure(figsize=(10,6))
    plt.bar(models, mse_scores, color='lightgreen')
    plt.title("Model Comparison - MSE")
    plt.ylabel("Mean Squared Error")
    plt.xticks(rotation=45)
    plt.show()

    return {"all data":metrics}