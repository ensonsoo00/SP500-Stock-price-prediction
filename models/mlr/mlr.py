
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA


#print("Running mlr...")

"""
This class provides implementation for training a linear regression model with/without PCA. 
"""
class MLR:

    def __init__(self, X_train, X_test, y_train, y_test, X_keys, with_pca=False):
        """
        Args:
            X_train : training features dataset
            X_test : testing features dataset
            y_train : training target dataset
            y_test : testing target dataset
            X_keys : column names for features dataset
            with_pca : whether or not to project features onto PCA components (default is False)
        Attributes:
            self.name : either "Multiple Linear Regression" or "PCA Multiple Linear Regression"
            self.model : trained linear regression model
            self.r2_score : testing r2 score
            self.X_train : training features dataset (may be projected onto PCA components)
            self.X_test : testing features dataset (may be projected onto PCA components)
            self.y_train : training target dataset
            self.y_test : testing target dataset
            self.with_pca : whether or not the features dataset was projected onto PCA components
            self.feature_importances : dictionary with feature names as keys and model coefficients as values
            self.best_features : best features selected by forward stepwise selection (if applied) ordered by feature importance (descending order)
            self.optimal_num_components : best number of PCA components, if PCA was applied, otherwise equal to total number of features
        """
        self.X_train = pd.DataFrame(X_train, columns=X_keys)
        self.X_test = pd.DataFrame(X_test, columns=X_keys)
        self.y_train = y_train
        self.y_test = y_test
        self.with_pca = with_pca
        self.best_features = list(X_keys)
        self.optimal_num_components = X_train.shape[1]

        if self.with_pca:
            self.name = "PCA Multiple Linear Regression"
            # get best number of PCA components with cross validation
            num_components = self.pca_cross_validation()
            self.optimal_num_components = num_components
            pca = PCA(n_components=num_components)
            pca.fit(self.X_train)
            # project training and testing features onto PCA components
            pca_train_features = pca.transform(self.X_train)
            pca_test_features = pca.transform(self.X_test)

            self.X_train = pca_train_features
            self.X_test = pca_test_features

            self.model = LinearRegression().fit(self.X_train, self.y_train)
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            pca_train_r2 = r2_score(self.y_train, y_train_pred)
            pca_test_r2 = r2_score(self.y_test, y_test_pred)
            self.r2_score = pca_test_r2
            # print(f"Training r2 with {num_components} components:", pca_train_r2)
            # print(f"Testing r2 with {num_components} components:", pca_test_r2)
            
        else:
            self.name = "Multiple Linear Regression"  
            
            # train on all features
            features = sm.add_constant(self.X_train)
            all_model = sm.OLS(self.y_train, features).fit()
            # print("All features training r2 score:", all_model.rsquared)
            y_test_pred = all_model.predict(sm.add_constant(self.X_test))
            all_r2 = r2_score(self.y_test, y_test_pred)
            # print("All features testing r2 score:", all_r2)

            # perform subset selection
            selected_features_list = self.subset_selection()
            
            # train on selected features
            subset_features = sm.add_constant(self.X_train[selected_features_list])
            subset_model = sm.OLS(self.y_train, subset_features).fit()
            # print("Subset selection training r2 score:", subset_model.rsquared)

            subset_y_test_pred = subset_model.predict(sm.add_constant(self.X_test[selected_features_list]))
            subset_r2 = r2_score(self.y_test, subset_y_test_pred)
            # print("Subset selection testing r2 score:", subset_r2)


            # compare model using all features vs model using forward stepwise selection
            if all_r2 > subset_r2:
                self.model = all_model
                self.r2_score = all_r2
            else:
                self.best_features = list(selected_features_list)
                self.X_train = self.X_train[self.best_features]
                self.X_test = self.X_test[self.best_features]
                self.r2_score = subset_r2
                self.model = subset_model
            # get model coefficients for each feature
            self.feature_importances = self.model.params.to_dict()
            self.feature_importances.pop("const")
            # rank most important features by coefficient magnitude
            self.best_features = sorted(list(self.feature_importances.keys()), key=lambda x: abs(self.feature_importances.get(x)), reverse=True)
            # print("Best features:", self.best_features)
            # print(f"{len(self.best_features)} predictors selected out of {X_train.shape[1]} total predictors")


    def evaluate(self):
        """
        Return the testing r2 score, the trained model, and model name/type
        """
        return {"r2": self.r2_score, "model": self.model, "name": self.name}

    def error(self):
        """
        Return the testing r2 score
        """
        return self.r2_score

        
    def subset_selection(self):
        """
        Perform forward stepwise selection to select best subset of features for linear regression
        """

        train_size = self.X_train.shape[0]
        # set of selected subset predictors
        S = set()

        # fitting model with no predictors
        const_features = [1] * train_size
        const_model = sm.OLS(self.y_train, const_features)
        const_res = const_model.fit()
        # print("Current S:", S)
        # print("\tR-squared adjusted:", const_res.rsquared_adj)

        # initialize max r2 adjusted score as the r2 adjusted score of the 0-predictor model
        max_r2_adj = const_res.rsquared_adj

        # select up to all features
        for iter in range(self.X_train.shape[1]):
            # keep track of highest r2
            max_r2 = 0
            # max_ele and r2_adj correspond to the added predictor and adjusted R squared of the model with highest r2
            max_ele = None
            r2_adj = 0
            # iterate over all remaining predictors
            for predictor in self.X_train.columns:
                if predictor not in S:
                    # create copy of S with predictor added
                    temp_S = set(S)
                    temp_S.add(predictor)
                    
                    # generate features of given power
                    features = self.X_train[temp_S]
                    features = sm.add_constant(features)
                    
                    # fitting model with generated features
                    model = sm.OLS(self.y_train, features)
                    res = model.fit()
                    
                    # if r2 score of fitted model is higher than the previous models, update the max r2 score
                    score = res.rsquared
                    if score > max_r2:
                        max_r2 = score
                        max_ele = predictor
                        # keep track of r2 adjusted score of the best model
                        r2_adj = res.rsquared_adj
            # if adjusted r2 score of the "best" model is less than the adjusted r2 score of previous iteration, break out
            # of forward subset selection
            if r2_adj < max_r2_adj:
                break
            # otherwise add the selected feature to the selected subset
            S.add(max_ele)
            max_r2_adj = r2_adj
            # print("Current S:", S)
            # print("\tR-squared adjusted:", max_r2_adj)

        # print("Final subset selection S:", S)
        # print("\tR-squared adjusted:", max_r2_adj)
        # print(f"{len(S)} predictors selected out of {self.X_train.shape[1]} total predictors")
        return list(S)
        

    def pca_cross_validation(self):
        """
        Perform cross validation to select best number of PCA components to use in linear regression
        """
        pca_scores = {}
        for M in range(1,self.X_train.shape[1]):
            pca = PCA(n_components=M)
            pca.fit(self.X_train)
            pca_features = pca.transform(self.X_train)
            lin_model = LinearRegression()
            cv = KFold(n_splits=10, shuffle=True, random_state = 123)
            score = cross_val_score(lin_model, pca_features, self.y_train, cv=cv, scoring="neg_mean_squared_error")
            mean_score = np.mean(np.absolute(score))
            pca_scores[M] = mean_score
            #print("MSE for M =", M, ":", mean_score)
            
        best_M = min(pca_scores.keys(), key=lambda x:pca_scores.get(x))
        # print("Best number of components:", best_M)
        return best_M

        

