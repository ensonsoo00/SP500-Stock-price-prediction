from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import warnings


#print("Running lasso...")

"""
This class provides implementation for training a Lasso Regression model.
"""
class LassoRegression:

    def __init__(self, X_train, X_test, y_train, y_test, X_keys):
        """
        Args:
            X_train : training features dataset
            X_test : testing features dataset
            y_train : training target dataset
            y_test : testing target dataset
            X_keys : column names for features dataset
        Attributes:
            self.name : model name ("Lasso Regression") 
            self.model : trained Lasso model
            self.r2_score : testing r2 score
            self.X_train : standardized training features dataset
            self.X_test : standardized testing features dataset
            self.y_train : standardized training target dataset
            self.y_test : standardized testing target dataset
            self.best_alpha : optimal Lasso lambda hyperparameter chosen by cross validation
            self.feature_importances = dictionary with feature names as keys and model coefficient as value 
            self.best_features = list of feature names in order of importance (in descending order)
            self.nonzero_coef_features = list of feature names in order of importance (in descending order) where model coefficient is nonzero
        """
        self.X_train = pd.DataFrame(X_train, columns=X_keys)
        self.X_test = pd.DataFrame(X_test, columns=X_keys)
        self.y_train = y_train
        self.y_test = y_test

        self.name = "Lasso Regression"  
    
        warnings.filterwarnings("ignore")
        cv = KFold(n_splits=3, shuffle=True, random_state=123)
        lassocv = LassoCV(alphas=[10**x for x in range(-5,6)], 
                          cv=cv, random_state=123, n_jobs=-1)
        lassocv.fit(self.X_train, self.y_train)
        # get best lambda parameter from cross validation on lasso
        self.best_alpha = lassocv.alpha_
        self.model = Lasso(alpha=self.best_alpha).fit(self.X_train, self.y_train)
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        self.r2_score = test_r2
        self.feature_importances = {col: coef for col, coef in zip(X_keys, self.model.coef_)}
        self.best_features = sorted(list(self.feature_importances.keys()), key=lambda x: abs(self.feature_importances.get(x)), reverse=True)
        self.nonzero_coef_features = [feature for feature in self.best_features if abs(self.feature_importances.get(feature)) > 0]
        # print(f"Testing r2 score: {test_r2}")


    def evaluate(self):
        """
        Return the testing r2 score, the trained model, and model name/type
        """
        return {"r2": self.r2_score, "model": self.model, "name": self.name}

    def error(self):
        """ 
        Return the testing r2 score of the trained model
        """
        return self.r2_score
        
