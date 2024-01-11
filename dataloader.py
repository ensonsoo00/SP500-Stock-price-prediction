print("loading raw data and importing libraries...")
import preprocessing as loader
from models.mlr.mlr import MLR
from models.tree.tree import Tree
from models.lasso.lasso import LassoRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from models.nn.nn import NN


X,y,keys_X,keys_y = loader.extractData()
X = X[:,~np.any(np.isnan(X), axis=0)]
print(np.isnan(y))

print("DATA LOADED")
print("Input Shape:",X.shape)
print("Output Shape:", y.shape)
print("Input Keys:",keys_X)
print("Output Keys:",keys_y)


def createLogFile(models):
    with open('logfile.csv','w') as logFile:
        logFile.write("fold, min_error,min_error_model")
        for model in models:
            logFile.write(","+model.name)
        logFile.write("\n")
        logFile.flush()
        logFile.close()

def log(fold,evaluation,errs,min_err,min_err_model):
    with open('logfile.csv','a') as logFile:
        logFile.write(f"{fold},{min_err},{min_err_model.name}")
        for model in evaluation:
            logFile.write(","+str(model["r2"]))
        logFile.write("\n")
        logFile.flush()
        logFile.close()
    return

def standardize(X_train, X_test, y_train, y_test):
    """
    Standardize the training and testing features/target data using StandardScaler
    """
    scaler = StandardScaler()

    scaler.fit(X_train)
    # scale training and testing features data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # scale training and testing target data
    scaler.fit(np.array(y_train).reshape(-1,1))
    y_train = scaler.transform(np.array(y_train).reshape(-1,1)).reshape(-1)
    y_test = scaler.transform(np.array(y_test).reshape(-1,1)).reshape(-1)
    return X_train, X_test, y_train, y_test

#perform a k fold train-test split
n_splits = 20
kf = KFold(n_splits=n_splits,random_state=42,shuffle=True)

t_errs = []
errs_eval = []

#iterate over k folt train-test split
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train = X[train_index,:]
    X_test = X[test_index,:]
    y_train = y[train_index].reshape(-1)
    y_test = y[test_index].reshape(-1)

    # standardize training and testing data
    X_train, X_test, y_train, y_test = standardize(X_train, X_test, y_train, y_test)
    
    #run each model
    print(X_train.shape)

    mlr = MLR(X_train,X_test,y_train,y_test, keys_X)
    mlr_pca = MLR(X_train,X_test,y_train,y_test,keys_X, with_pca=True)
    lasso = LassoRegression(X_train,X_test,y_train,y_test,keys_X)
    tree = Tree(X_train,X_test,y_train,y_test, keys_X)
    tree_pca = Tree(X_train,X_test,y_train,y_test,keys_X,True)
    nn = NN(X_train,X_test,y_train,y_test)
    nn.train()
    nn.test()
    models = [mlr,mlr_pca,lasso,tree,tree_pca,nn]

    evaluation = [model.evaluate() for model in models]
    errs = [model.error() for model in models]
    min_err = max(errs)
    min_err_model = models[errs.index(min_err)]
    min_err_eval = evaluation[errs.index(min_err)]
    if i == 0:
        createLogFile(models)
    log(i,evaluation,errs,min_err,min_err_model)
    errs_eval.append({"model":min_err_model,"err":min_err,"name":min_err_eval["name"]})
    t_errs.append(min_err)

    """
    tree = Tree(X_train,X_test,y_train,y_test)
    nn = NN(X_train,X_test,y_train,y_test)
    models = [mlr,pcr,tree,nn]
    #run evaluation of each model
    evaluation = [model.evaluate() for model in models]
    errs = [model.error() for model in models]
    
    """

#print out the best model for each kfold and print out overall best

best_error = max(t_errs)
best_eval = errs_eval[t_errs.index(best_error)]
best_eval_name = best_eval["name"]
print(f"Best Model:{best_eval_name} with an r^2 error of {best_error}")

    