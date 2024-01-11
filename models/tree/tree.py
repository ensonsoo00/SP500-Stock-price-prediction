
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
import numpy as np

"""
This class provides implementation for training a random forest regressor model with/without PCA. 
"""
class Tree:

    def __init__(self, X_train,X_test,y_train,y_test,X_keys,with_pca=False,pca_size=None,n_features=None):
        """
        Args:
            X_train : training features dataset
            X_test : testing features dataset
            y_train : training target dataset
            y_test : testing target dataset
            X_keys : column names for features dataset
            with_pca : whether or not to project features onto PCA components
            pca_size : option to use a specified number of PCA components; if None, then pca_size is chosen through cross validation
            n_features : option to use a specified maximum depth in random forest; if None, then max depth is chosen through cross validation
        Attributes:
            self.name : either "Random Forest Regressor" or "PCA Random Forest Regressor"
            self.regr : trained random forest regression model
            self.r2_score : testing r2 score
            self.X_train : training features dataset (may be projected onto PCA components)
            self.X_test : testing features dataset (may be projected onto PCA components)
            self.y_train : training target dataset
            self.y_test : testing target dataset
            self.X_keys : feature/column names
            self.n_features : specified maximum depth of random forest
            self.with_pca : whether or not the features dataset was projected onto PCA components
            self.best_params : best hyperparameters for contructing random forest
            self.feature_importances : dictionary of each feature and its feature importance in the fitted random forest
            self.best_features : best features ordered by feature importance (descending order)
            self.optimal_num_components : best number of PCA components, if PCA was applied
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_keys = X_keys
        self.n_features = n_features
        if with_pca:
            # if pca size is not given, then choose number of components through cross validation
            if pca_size is None:
                pca_size = self.pca_cross_validation()
            self.optimal_num_components = pca_size
            pca = PCA(n_components=pca_size)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            self.X_train = X_train_pca
            self.X_test = X_test_pca
            self.name = "PCA Random Forest Regressor"
        else:
            self.name = "Random Forest Regressor"
        # select best hyperparameters for random forest through cross validation
        self.best_params = self.cross_validation()
        
        self.regr = RandomForestRegressor(random_state=44, **self.best_params)
        # use the n_features as max depth (if given)
        if n_features is not None:
            self.regr.set_params(**{"max_depth":n_features})
        # otherwise, set n_features as the max depth found through cross validation
        else:
            n_features = self.best_params["max_depth"]
        self.regr.fit(self.X_train,self.y_train)
        pred = self.regr.predict(self.X_test)
        score = r2_score(self.y_test,pred)
        self.r2_score = score
        # get feature importance scores
        self.feature_importances = {col:imp for col, imp in zip(X_keys, self.regr.feature_importances_)}
        # rank the most important features
        self.best_features = list(sorted(self.feature_importances, key=lambda x: self.feature_importances.get(x), reverse=True))

    def evaluate(self):
        """
        Return the testing r2 score, the trained model, and model name/type
        """
        return {"r2":self.r2_score,"model":self.regr,"name":self.name,"n_features":self.n_features}
    def error(self):
        """
        Return the testing r2 score
        """
        return self.r2_score
    
    def cross_validation(self):
        """
        Perform cross validation to select the best hyperparameters (number of estimators, max features, and max depth) to use in random forest 
        """
        param_grid = {
            "n_estimators" : np.arange(100,250,50),
            "max_features" : ['sqrt','log2'],
            "max_depth" : np.arange(4,10,2)
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=123)
        search = GridSearchCV(estimator=RandomForestRegressor(random_state=123), param_grid=param_grid, cv=cv, n_jobs=-1)
        search.fit(self.X_train, self.y_train)
        return search.best_params_
    
    def pca_cross_validation(self):
        """
        Perform cross validation to select best number of PCA components to use in random forest
        """
        pca_scores = {}
        for perc in np.arange(0.1, 1.0, 0.1):
            pca = PCA(n_components=perc)
            pca.fit(self.X_train)
            pca_features = pca.transform(self.X_train)
            forest = RandomForestRegressor(random_state=123)
            cv = KFold(n_splits=5, shuffle=True, random_state = 123)
            score = cross_val_score(forest, pca_features, self.y_train, cv=cv, scoring="neg_mean_squared_error")
            mean_score = np.mean(np.absolute(score))
            num_components = pca.n_components_
            pca_scores[num_components] = mean_score
            #print("MSE for M =", M, ":", mean_score)
            
        best_M = min(pca_scores.keys(), key=lambda x:pca_scores.get(x))
        # print("Best number of components:", best_M)
        return best_M