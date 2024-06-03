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

## 5. Steps Involved in data preprocessing and model development

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

```python
df = pd.read_csv("simplelinearregression.csv")
```

In the next step, we load our dataset from a CSV (Comma Separated Values) file using the Pandas library.
`pd.read_csv("simplelinearregression.csv")`: This part of the code uses the Pandas library (`pd`) to read the contents of a CSV file named "simplelinearregression.csv". The `read_csv()` function is a Pandas function specifically designed to read data from CSV files and create a DataFrame, which is a two-dimensional labeled data structure with rows and columns similar to a spreadsheet or SQL table.

By executing this line of code, we're essentially bringing our dataset into our Python environment, allowing us to manipulate, analyze, and visualize the data using the powerful tools provided by the Pandas library.

### C. Data Exploration

```python
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe().T)
```

To explore our dataset and gain insights into its structure, we utilize various functionalities provided by the Pandas library. This includes:

- Printing the first few rows of the dataset to get an initial understanding of the data using `print(df.head())`.
- Checking the dimensions of the dataset (number of rows and columns) using `print(df.shape)`.
- Identifying and counting any missing values in the dataset using `print(df.isnull().sum())`.
- Generating descriptive statistics for numerical columns in the dataset to understand its distribution, central tendency, and dispersion using `print(df.describe().T)`.

By executing these commands, we can comprehensively explore our dataset, uncovering its characteristics and potential areas that may require preprocessing before model training. This foundational step sets the stage for further analysis and model development.

### D. Exploratory Data Analysis
We analyze the distribution of age and premium and the relationship between them.

#### 1. Age Distribution

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=20, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.grid(True)
plt.show()
```
We create a histogram to visually represent the distribution of ages in our dataset. The histogram reveals how frequently different age groups occur within the data. This distribution provides insights into the demographic composition of the dataset, aiding in further analysis and modeling decisions.

#### 2. Premium Distribution

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Premium"], bins=20, edgecolor="black")
plt.xlabel("Premium")
plt.ylabel("Frequency")
plt.title("Distribution of Premium")
plt.grid(True)
plt.show()
```

We utilize a histogram to illustrate the distribution of premiums within our dataset. This visualization offers insights into how premiums are distributed across different ranges.  This understanding of premium distribution is crucial for analyzing the variability and trends within the dataset, facilitating informed decision-making in subsequent stages of our analysis.

#### 3. Relationship Between Age and Premium

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["Age"], df["Premium"], alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Premium")
plt.title("Relationship Between Age and Premium")
plt.grid(True)
plt.show()
```
We used a scatter plot to explore the relationship between age and premium within our dataset. This visualization allows us to observe any potential patterns or trends between these two variables. This visualization serves as a preliminary examination of the association between age and premium, providing a foundation for further analysis and modeling.

#### 4. Calculate Correlation Coefficient

```python
correlation = df["Age"].corr(df["Premium"])
print("Correlation Coefficient:", correlation)
```

We calculate the correlation coefficient to quantify the strength of the relationship between age and premium within our dataset. The correlation coefficient measures the degree of linear association between two variables, ranging from -1 to 1. A coefficient close to 1 indicates a strong positive linear relationship, while a coefficient close to -1 suggests a strong negative linear relationship. A coefficient near 0 implies a weak linear relationship.

### E. Pre-processing

```python
X = df.drop('Premium', axis=1)
y = df['Premium']
```
We split our dataset into features (X) and the target variable (y). The features, denoted by X, consist of all columns except the "Premium" column, which is dropped using the `drop` function along the column axis (axis=1). 
The target variable, denoted by y, is assigned the "Premium" column, representing the variable we aim to predict. This separation allows us to train our machine learning model using the features to predict the target variable.

### F. Splitting Data into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```
We partition our dataset into training and testing subsets using the train_test_split function from scikit-learn.
Allocating 70% of the data for training and 30% for testing. This ensures that our model can learn patterns from the training data and then evaluate its performance on unseen test data, helping us assess its generalization ability.

### G. Building and Training the Linear Regression Model

```python
LR = LinearRegression()
LR.fit(X_train, y_train)
```
We create a linear regression model and train it using the training data. This involves initializing a linear regression model object and fitting it to the training features (X_train) along with their corresponding target values (y_train). Through this process, the model learns the relationship between age and premium in the training dataset.

### H. Making Predictions

```python
y_pred = LR.predict(X_test)
```
We utilize the trained linear regression model to predict premium values for the test set. By calling the `predict` method on the linear regression model (`LR`) with the test features (`X_test`), we generate predicted premium values (`y_pred`).

### I. Evaluating Model Performance
We evaluate the model using various metrics.

#### 1. Mean Squared Error (MSE)

```python
M = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", M)
```
We calculate the Mean Squared Error (MSE), which quantifies the average squared difference between predicted and actual premium values.

#### 2. Mean Absolute Error (MAE)

```python
MAE = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", MAE)
```
We calculate the Mean Absolute Error (MAE), which measures the average absolute difference between predicted and actual premium values.

#### 3. Root Mean Squared Error (RMSE)

```python
from math import sqrt
RMSE = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", RMSE)
```

We compute the Root Mean Squared Error (RMSE), which is the square root of the Mean Squared Error (MSE), providing error in the same units as the target variable.

#### 4. R² Score

```python
R2_score = r2_score(y_test, y_pred)
print("R² Score:", R2_score)
```

The R² score indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. It is a measure of how well the regression model fits the observed data, with values ranging from 0 to 1. A higher R² score signifies a better fit.


## 6. Conclusion
- The simple linear regression model demonstrated exceptional performance in predicting premium amounts based on age.
- Mean Squared Error (MSE) was close to zero (8.41e-30), indicating high accuracy.
- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) were also close to zero (2.37e-15 and 2.90e-15 respectively), signifying excellent model performance.
- With an R² score of 1.0, the model explains all the variability in premiums solely based on age, indicating a perfect fit.
- The strong positive correlation between age and premium suggests that as individuals age, their premiums tend to increase.
- This analysis underscores the effectiveness of simple linear regression in capturing and predicting linear relationships between variables.
- The model can be utilized to make accurate premium predictions for individuals based on their age, providing valuable insights for insurance companies or other related domains.
- 
In conclusion, the simple linear regression model performed exceedingly well in predicting premiums based on age, with an R² score of 1.0, indicating a perfect fit of the model to the data.

## Important Points to Remember
- Simple linear regression assumes a linear relationship between the independent and dependent variables.
- The model's performance can be evaluated using metrics such as MSE, MAE, RMSE, and R² Score.
- Understanding the data and checking for assumptions (linearity, independence, homoscedasticity, and normality) is crucial for building a reliable model.

This project is a practical introduction to simple linear regression, providing insights into how machine learning can be used to make predictions based on data.
