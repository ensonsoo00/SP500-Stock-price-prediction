# DS4400-Final-Project: Predicting Intrinsic Value
Group Members: Matt Kim, Marco Hampel, Enson Soo

## Introduction
In a trading era defined by speculation and drastic swings, it is important to utilize a company's financial statements as a source of truth in determining the quality of a company. Companies such as Silicon Valley Bank and Peloton were given lofty valuations as investors brought speculation about future markets to the extremes, when in reality, the companies' balance sheets were riddled with debt that had little to no clear picture of profitability potential. Furthermore, the most successful trader in history, Warren Buffet, holds an intensely focused strategy defined by long-term investments in companies with undeniable intrinsic value. This metric can be defined as the true book value of an asset and is often overlooked in today's market as investors clamor to predict the next big thing. 

Through our project, we aim to study the financial features that define the 500 largest companies listed on stock exchanges in search of a predictive model that can accurately predict share price through traditional financial inputs. In addition, to consider media perception and to not fall into overly antiquated practices - we aim to consider news headlines about the companies and how this information effects their price. We believe that this combination is the correct balance of financial statement consideration and playing the speculative field. By using the S&P500 companies as our subjects of analysis, the model will learn that characteristics that hold importance when evaluating the intrinsic value of a company. This practice is in-line with the work of financial analysts that use financial ratios and cash flows to predict where a stock price is moving, however implementing the machine learning practices we've learned will allow the models to determine what the source of truth should be. This methodology of "taking out the human element" when making investment decisions is a common modern practice implemented by hedge funds designed to perform in all types of market conditions. 

With 10-K financial information being publicly available for major companies, we believe that being able to predict intrinsic value through a combination of raw financials and speculative media - is a trading strategy that is built to adapt to modern markets as well as staying true to sound investment practice.

## Setup
The goal of this project is to develop a model to predict the intrinsic value of a company given its stock price, as well as its overall market perception. This was accomplished by analyzing the S&P 500 Stock Value Dataset: `S&P 500 Stock Value Dataset` - a dataset which hold stock price information of 10 years worth of stocks as well as metrics related to the stock such as volume and financial statements. We use this financial data to calculate intrinsic value for a given stock at a given time.

We quantified the headlines dataset by applying a sentiment analysis algorithm which evaluates how positive or negative a headline is, and mapped it to the appropriate stock and time period. Given that several options exist for designing a model to predict a quantitative outcome, we decided to train several models applying K-Fold cross validation and subset selection on each model to evaluate which predictors are most useful from our dataset and which model can provide us with the smallest testing error.


### Trained Models/Experiments
Before training the models, we decided to standardize our training and testing dataset to bring all features to a common scale. This prevented certain features with different distribution ranges from affecting principal component analysis and Lasso regularization, and also allowed for a meaningful interpretation of feature importance through the trained linear model coefficients. 

1. Multiple Linear Regression with/without PCA
- Effective baseline to observe linearity of data in input dataset, given likely non-linear nature of market data and compare effectiveness of other models with this.
- Implementation:
    - When PCA was not applied, we performed forward stepwise selection to select the best features for linear regression. Then we trained a model using the selected features and another model using all features. The model that yielded the higher testing r2 score was selected as the final model.
    - When PCA was applied, we performed cross validation to select the best number of PCA components. After projecting the feature data onto the selected number of PCA components, we trained a linear regression model on the projected training and testing data. 

2. Lasso Regression
- Effective at observing linearity of data, regularizing model coefficients, and automatic feature selection
- Implementation:
    - The Lasso regularization $\lambda$ hyperparameter was chosen by cross validation
    - A Lasso regression model was trained using this optimal value of $\lambda$
- Hyperparameters chosen by cross validation: regularization lambda parameter 
3. Regression trees with/without PCA
- Effective at matching highly nonlinear data, however also prone to overfitting.
- Implementation:
    - When PCA was not applied, we performed cross validation for the random forest hyperparameters including the number of estimators, the maximum depth of each tree, and the maximum number of features to consider at each split. A random forest regressor was trained using the selected hyperparameters.
    - When PCA was applied, we performed cross validation for the number of PCA components. Instead of testing a specific number of PCA components during cross validation, we decided to test the number of components that explained a certain threshold of explained variance in our training dataset (for the purpose of reducing training time). After projecting the feature data to the optimal number of PCA components, we performed the same process of cross validation and training as the random forest implementation without PCA (explained above).  
4. Neural Networks
- Handles highly non linear data well however is computationally expensive to train with poor interpretability and possibility of overfitting

## Results

### Main Results
The $R^2$ results are displayed in the figures below:

![image](https://cdn.discordapp.com/attachments/884300143548063754/1101346948247724072/image.png)
![image](https://cdn.discordapp.com/attachments/884300143548063754/1101348721603653652/image.png)

As of right now the model which has performed the best has been the Regression tree without PCA. With the highest $R^2$ of 0.97 and the lowest variance in $R^2$ values over each iteration of K-Fold the model out-performs Linear Regression and Tree based with PCA. While the Neural Network has low variance in its results, its r^2 is far worse sitting just above 0.6.

#### K Fold Results
![image](https://cdn.discordapp.com/attachments/884300143548063754/1101333038190841856/image.png)

**Tabular Data:**
| fold |  highest_r2 | highest_r2_model                | Multiple Linear Regression | PCA Multiple Linear Regression | Lasso Regression | Random Forest Regressor | PCA Random Forest Regressor | Neural Network |
| ---- | ---------- | ------------------------------ | -------------------------- | ------------------------------ | ---------------- | ----------------------- | --------------------------- | -------------- |
| 0    | 0.060688   | Random Forest Regressor        | \-0.62236                  | \-1.93415                      | 0.015952         | 0.060688                | \-0.09029                   | \-0.28757      |
| 1    | 0.975332   | PCA Random Forest Regressor    | \-152.215                  | \-1.29996                      | \-8.14091        | 0.851074                | 0.975332                    | \-66.2888      |
| 2    | 0.877746   | Random Forest Regressor        | \-1.88285                  | 0.603823                       | 0.449755         | 0.877746                | 0.504443                    | \-1.26621      |
| 3    | 0.678597   | Lasso Regression               | \-1.04859                  | 0.601162                       | 0.678597         | 0.624728                | \-0.13214                   | 0.629482       |
| 4    | 0.355368   | Random Forest Regressor        | \-0.84564                  | 0.015922                       | 0.095978         | 0.355368                | \-0.08595                   | \-0.85142      |
| 5    | 0.425308   | Multiple Linear Regression     | 0.425308                   | 0.255546                       | \-0.04363        | 0.288962                | \-0.09965                   | \-13.7551      |
| 6    | 0.896403   | PCA Multiple Linear Regression | 0.839883                   | 0.896403                       | 0.859526         | 0.717273                | 0.835978                    | 0.338535       |
| 7    | 0.676914   | Random Forest Regressor        | \-373.964                  | \-94.1388                      | \-455.156        | 0.676914                | \-1.58085                   | \-0.27654      |
| 8    | 0.347377   | PCA Random Forest Regressor    | \-0.76145                  | 0.09593                        | \-0.68846        | 0.102723                | 0.347377                    | 0.308129       |
| 9    | 0.244821   | PCA Multiple Linear Regression | \-2.14711                  | 0.244821                       | 0.078439         | 0.206067                | 0.026807                    | 0.049941       |
| 10   | \-0.33059  | PCA Random Forest Regressor    | \-6.64985                  | \-1.37973                      | \-7.14573        | \-0.39043               | \-0.33059                   | 0.013495       |
| 11   | 0.852741   | Random Forest Regressor        | 0.818802                   | 0.743987                       | 0.791892         | 0.852741                | 0.340548                    | \-7.92869      |
| 12   | 0.189031   | Random Forest Regressor        | \-0.27854                  | 0.038207                       | \-0.18245        | 0.189031                | 0.180562                    | \-2.97105      |
| 13   | 0.18141    | Random Forest Regressor        | \-0.74209                  | \-1.78286                      | \-1.46524        | 0.18141                 | \-0.08152                   | \-0.22986      |
| 14   | 0.86603    | PCA Multiple Linear Regression | 0.55402                    | 0.86603                        | 0.852998         | 0.736875                | 0.57083                     | \-10.954       |
| 15   | 0.727337   | Random Forest Regressor        | 0.310149                   | 0.146993                       | 0.564545         | 0.727337                | 0.508347                    | \-0.31918      |
| 16   | 0.976973   | Random Forest Regressor        | 0.953903                   | 0.947843                       | 0.955388         | 0.976973                | 0.972814                    | \-5.65867      |
| 17   | 0.889835   | PCA Multiple Linear Regression | 0.75171                    | 0.889835                       | 0.700397         | 0.703766                | 0.659637                    | \-6.11348      |
| 18   | 0.900096   | PCA Multiple Linear Regression | 0.445732                   | 0.900096                       | 0.596074         | 0.891048                | 0.781658                    | 0.597426       |
| 19   | 0.403788   | PCA Random Forest Regressor    | \-1.62349                  | \-0.17881                      | \-0.26151        | \-0.09278               | 0.403788                    | \-2.23286      |




### Supplementary Results 

We also observed the feature importances while training Multiple Linear Regression, Lasso Regression, and Random Forest Regressor. Before training each model, the training/testing data for the features and target variables were standardized. Therefore, for the linear models (multiple linear regression and lasso regression), the magnitude of the model's coefficients directly translated to the feature's importance. For each model, the top 10 important features were:
- Multiple Linear Regression: operatingIncome, netIncomeFromContinuingOps, incomeBeforeTax, totalCurrentLiabilities, totalOtherIncomeExpenseNet, totalOperatingExpenses, totalAssets, incomeTaxExpense, costOfRevenue, totalRevenue 
- Lasso Regression: researchDevelopment, repurchaseOfStock, totalCashFromOperatingActivities, netIncomeFromContinuingOps, commonStock, grossProfit, ebit, changeToLiabilities,, changeToAccountReceivables
- Random Forest Regressor: incomeBeforeTax, totalCashFromOperatingActivities, netIncomeApplicableToCommonShares, ebit, researchDevelopment, repurchaseOfStock, commonStock, incomeTaxExpense, totalStockholderEquity, totalRevenue

Over the top 15 important features for all 3 models, the common important features were:
- ebit
- grossProfit
- netIncomeFromContinuingOps

## Discussion
The Linear Regression model had very high variance in its $R^2$ value likely because it struggled to properly fit with the non-linear behavior of the market. Therefore it is believed the high $R^2$ value may be a result of some folds having higher correlation (linear behavior) than others. This is reinforced by Tree based methods extremely low variance since it can handle non linear data far better. PCA performed similarly well when applied to the Tree based method.

Neural Networks underperformed in this project likely due to the small size of the analyzed dataset in combination with the highly voltile non-linear nature of the network. Furthermore a higher learning rate became a necessity as training was not making meaningful progress with smaller learning rates. We considered using Adaptive Gradient Neural Networks as they would have helped modify learning rate to progress training faster, however it was decided not to because our dataset was determined to be too small to make a meaningful difference.

## Conclusion
When it came to our supplimentary results we found ebit, grossProfit and netIncomeFromContinuingOps to be shared common important feature.
`ebit` stands for Earnings Before Interest and Taxes. `grossProfit` is the total difference in the costs of the product a company produces and its revenue. Finally `netIncomeFromContinuingOps` stands for the after tax earnings a company has raised from its operations. 

Therefore, as per our investigation we can conclude that the profit post tax of a company relative to its operation costs are determined to be the most important factors to consider when determining the intrinsic value of a company. We have a high confidence that this result is correct as our analysis effectivly tested several machine learning models, each of which provided good error values, and all seem to produce similar conclusions as to which features are most important to determine intrinsic value given a company's stock and financial status.



## Project Design
The project is broken down into three main parts:
1. The Main Program
2. The Preprocessor
3. The Models

The Preprocessor, loads data from raw CSV files under the ``datasets`` folder, extracts the "intrinsic value" parameter from the dataset and converts the data into a numpy array for ``X`` and ``y`` as well as a list of the keys for each predictor.

The Main program loads this up from the preprocessor, imports the models and initializes each model (MLR, MLR-PCA, Lasso, Tree, Tree-Pca, NN (Neural Network)). Next it splits the data up into K-Folds using Sklearn's KFold class and iterates over each fold, splitting each X and Y into a training and testing dataset. Next the data is loaded into each model which fits on the training data, and tests on the testing data. The resulting testing R^2 error and other metrics are saved in each model's object. Once this is complete, the main program calls ``evaluate()`` and ``error()`` which returns first the metric and then the $R^2$ error.
Finally a ``log()`` function is called to log the result in a CSV, and the best performing model is printed before the program finishes.

### Running the Program
- Run ``DataLoader.py`` to run the program in its entirety.

## Documentation 

### Overview
The project structure is broken down into several parts. On the top layer of the file structure we have our `dataloader.py`. This is our defacto `main` function. Within `dataloader.py`, we start by importing our libraries. When `preprocssing.py` is imported, it will run the script which loads the data from our `datasets` folder and  converts it into our input and output datasets. `loader.extractData()` extracts the preprocessed dataset and removes any `nan` values still included. 

Next `dataloader.py` splits the data into 20 folds each of which is then split into 80% training data, 20% testing data. We standardize the data and pass it into our models.

Each model we used `MLR`, `PCR`, `Tree`, `Neural Networks` can be found in the `models` folder and are each imported by `dataloader.py`. Each model follows a standard design:

        class Model:
            def __init__(self,X_train,X_test,y_train,y_test, args**);
            def evaluate(self)
            def error(self)

Each model that uses additional hyper parameters (such as RandomForestDecision Trees located in `models.tree.tree.py`) are determined using a model.cross_validation() function. A special model.pca_cross_validation() function is made to consider pca hyperparameters in our cross_validation.
This function creates the appropriate scikit learn model given our input,output training/testing data and returns the best hyper paramters. 
`model.cross_validation()` returns these hyper_parameters, a scikit learn model is trained with these to calculate the error. The error we used was $R^2$.

The scikit learn models we used were:
|Model Name| Library model|
|----------|--------------|
|Linear Regression| sklearn.linear_model.LinearRegression|
|PCA|sklearn.decomposition.PCA|
|Random Forest| sklearn.ensemble.RandomForestRegressor|
|Neural Network| pytorch|
|Lasso| sklearn.linear_model.Lasso, LassoCV|

The Neural Network had two additional `nn.train()` and `nn.test()` functions.

Once each model's respective $R^2$ error was obtained in addition to its respective hyper-parameters, this data was returned in a dictionary through the `model.evaluate()` function. The resulting evaluations were then logged using our custom `log()` function.

`log()` would save all the data into our `logfile.csv` file, which has been provided in the table above.

### Pre-processor
Inside of `preprocessing.py` we utilized a set of python functions built for our dataset to help pre-process and generate our expected output data using several market factors. The code first took the raw stock data, analyzing quantities of a given stock being bought and sold, annual dividends, balance sheets of a company, income of a company, and cash flow, and merged them into a dataframe, where each row represidents a quarter within a year between 2010 and 2021, and summarizes our intrinsic value output as 'marketCap'.

We then converted that into a dataframe, excluded incomplete input rows and passed that out as the completed X, y data.

#### Feature Importance
In order to analyze our feature importance, we decided to create a seperate Jupyter notebook file in the root directory called `feature_importance.ipynb`. This takes our three interpretable models `Multiple linear regression`, `Lasso Regression` and `Random Forest Descision Trees` and performs a subset selection to find the best predictors. Since `Lasso` is a shrinkage method, the best predictors were determined by checking for remaining non_zero coeficients. The indices of the predictors were then matched to their respective predictor names and printed for interpretation/analysis.



#### Neural Network Architecture
The neural network used was a three layer fully connected neural network. One input layer, one hidden layer of size $n/2$ and the ont neuron output layer containing the quantitative output prediction of intrinsic value. Since we were working with quantitative data, the `relu` activation function was used. Our loss function for this model was `MSELoss`. 

This design was made as the multi-layer structure could allow for better non-linear feature analysis. We also applied a learning rate of `.1` after several iterations of running, as we found it produced the best results. 




## References
We would like to thank the author of this Kaggle dataset for making the dataset and code implementation publicly available:
- [S&P 500 stocks price with financial statement](https://www.kaggle.com/datasets/hanseopark/sp-500-stocks-value-with-financial-statement)
- [Prediction of price for ML with finance stats](https://www.kaggle.com/code/hanseopark/prediction-of-price-for-ml-with-finance-stats)

