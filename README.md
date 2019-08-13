# Vehicle Loan Status Prediction using Random Forest Classifier
### The Training Set.csv file was used to train and validate the dataset.

The training data consisted of 109,854 rows and 36 columns. The goal of this project was to predict whether the loan of a vehicle would be approved or declined. Since we had two values to predict, this made our project a binary classification problem. The most challenging part of this project was to find a subset of features which would be highly correlated to our target, which is 'Loan Status'. The following were the steps taken to achieve a reasonably good accuracy.

## Data cleaning
### 1. Separating the categorical variables: 
10 out of the 35 features (Loan Status removed, as it is the target) were categorical. Machine learning models do not handle categorical variables as strings, which means we had to separate them from the numerical variables and then convert the strings to numbers. 

### 2. Filling the missing values: 
The missing values were filled with the maximum value counts of each column. 

### 3. Converting categorical to numerical: 
  - **One-hot encoding:** In this technique, if there are three separate categories in a particular column, then it creates three different columns, one for each category, and then fills it in with 1s or 0s. 
  - **Label Encoding:** In this technique, a new integer is assigned for every other category. No new columns are created. Ex: if a column has three categories: Red, Green and Blue. Then {Red : 0, Green : 1, Blue : 2}
  
Once we have made the numerical conversions, we can merge the new categorical clean dataset with the numerical clean dataset.
  
### 4. Selecting Features:
We used a technique called Recursive Feature Elimination, which recursively removes features, builds a model using the remaining attributes and calculates model accuracy. RFE is able to work out the combination of attributes that contribute to the prediction on the target variable (or class). Out of the 35 features, we selected the top 25 which would contribute most to the final accuracy. We also achieved a ranking of the features based on their importance. 

### 5. Model Selection
  - **Voting Ensemble:** Voting is one of the simplest ways of combining the predictions from the machine learning algorithms. It works by creating two or more standalone models (in our case we use three models Logistic Regression, SVC and Decision Tree Classifier) from the training dataset. A voting classifier can then be used to wrap the models and average the predictions of the sub-models when asked to make a prediction on the test dataset. **F1 Score = 0.811 ~ 81.1%**
  - **Random Forest Classifier:** Random Forest is a supervised learning algorithm which can be used for both classification and regression. The Random Forest comprises of a number of trees and this value can be set as a parameter when defining the object of the classifier. We set the value of number of trees (*n_estimators*) as 1000. **F1 score = 0.865 ~ 86.5%**
