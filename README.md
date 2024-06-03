#  Predicting Insurance Premium Based on Age using Machine Learning

In this project, we used an insurance premium dataset to predict the premium amount based on age using simple linear regression.

## Table of Contents
1. Introduction
2. Objectives
3. Dataset Information
4. Prerequisites
5. Steps Involved
    1. Importing Libraries
    2. Loading the Dataset
    3. Data Exploration
    4. Exploratory Data Analysis
    5. Pre-processing
    6. Splitting Data into Training and Testing Sets
    7. Building and Training the Linear Regression Model
    8. Making Predictions
    9. Evaluating Model Performance
6. Conclusion

## 1. Introduction
Simple linear regression is a technique that predicts a response variable (in this case, the insurance premium) based on a single predictor variable (in this case, age). It assumes a linear relationship between the two variables, meaning that as one variable increases, the other tends to increase (or decrease) in a straight-line manner.

## 2. Objectives
The main objectives of this project are:
- To understand the relationship between age and insurance premium.
- To build a model that can predict the insurance premium based on age.
- To evaluate the model's performance using various metrics.

## 3. Dataset Information
The dataset contains two columns:
- **Age**: The age of the individual.
- **Premium**: The insurance premium amount paid by the individual.

## 4. Prerequisites
To run this project, we'll need:
- Python installed on your computer.
- Basic understanding of Python programming.
- Libraries: pandas, matplotlib, sklearn

## 5. Steps Involved

### A. Importing Libraries

```python
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')
```

First, we need to import the necessary libraries that are essential to our data processing, model training, and assessment stages.This encompasses essential tools like **Pandas** for data manipulation and **Matplotlib** for visualization. From scikit-learn, a powerful Python ML library, we import specific functions and classes. These include **train_test_split** for dataset splitting, **LinearRegression** for model construction, and evaluation metrics such as **mean squared error, mean absolute error, and R-squared**. Importing the **warnings** module and suppressing warnings ensures a streamlined output, emphasizing the core aspects of our analysis without distractions.

### B. Loading the Dataset
Next, we load the dataset from a CSV file.

```python
df = pd.read_csv("simplelinearregression.csv")
```

### C. Data Exploration
We explore the dataset to understand its structure and check for any missing values.

```python
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe().T)
```

### D. Exploratory Data Analysis
We analyze the distribution of age and premium and the relationship between them.

#### 1. Age Distribution
We create a histogram to visualize the distribution of ages.

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=20, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.grid(True)
plt.show()
```

#### 2. Premium Distribution
We create a histogram to visualize the distribution of premiums.

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Premium"], bins=20, edgecolor="black")
plt.xlabel("Premium")
plt.ylabel("Frequency")
plt.title("Distribution of Premium")
plt.grid(True)
plt.show()
```

#### 3. Relationship Between Age and Premium
We create a scatter plot to visualize the relationship between age and premium.

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["Age"], df["Premium"], alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Premium")
plt.title("Relationship Between Age and Premium")
plt.grid(True)
plt.show()
```

#### 4. Calculate Correlation Coefficient
We calculate the correlation coefficient to quantify the strength of the relationship between age and premium.

```python
correlation = df["Age"].corr(df["Premium"])
print("Correlation Coefficient:", correlation)
```

### E. Pre-processing
We separate the features (independent variables) from the target (dependent variable).

```python
X = df.drop('Premium', axis=1)
y = df['Premium']
```

### F. Splitting Data into Training and Testing Sets
We split the data into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

### G. Building and Training the Linear Regression Model
We create and train the linear regression model.

```python
LR = LinearRegression()
LR.fit(X_train, y_train)
```

### H. Making Predictions
We use the trained model to make predictions on the test set.

```python
y_pred = LR.predict(X_test)
```

### I. Evaluating Model Performance
We evaluate the model using various metrics.

#### 1. Mean Squared Error (MSE)
Measures the average squared difference between predicted and actual values.

```python
M = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", M)
```

#### 2. Mean Absolute Error (MAE)
Measures the average absolute difference between predicted and actual values.

```python
MAE = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", MAE)
```

#### 3. Root Mean Squared Error (RMSE)
The square root of MSE, giving error in the same units as the target variable.

```python
from math import sqrt
RMSE = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", RMSE)
```

#### 4. R² Score
Indicates the proportion of the variance in the dependent variable predictable from the independent variable.

```python
R2_score = r2_score(y_test, y_pred)
print("R² Score:", R2_score)
```

## 6. Conclusion
The project successfully demonstrates the application of simple linear regression to predict insurance premiums based on age. The evaluation metrics indicate the performance of the model, and the high correlation coefficient suggests a strong linear relationship between age and premium.

## Important Points to Remember
- Simple linear regression assumes a linear relationship between the independent and dependent variables.
- The model's performance can be evaluated using metrics such as MSE, MAE, RMSE, and R² Score.
- Understanding the data and checking for assumptions (linearity, independence, homoscedasticity, and normality) is crucial for building a reliable model.

This project is a practical introduction to simple linear regression, providing insights into how machine learning can be used to make predictions based on data.
