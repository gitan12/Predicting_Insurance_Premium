# Predicting Insurance Premium Based on Age using Machine Learning

This project demonstrates how to use simple linear regression to predict insurance premiums based on age. The project includes a Python script and a Streamlit application to interactively predict insurance premiums and save predictions to a CSV file.

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Dataset Information](#dataset-information)
4. [Prerequisites](#prerequisites)
5. [Steps Involved](#steps-involved)
    1. [Importing Libraries](#importing-libraries)
    2. [Loading the Dataset](#loading-the-dataset)
    3. [Data Exploration](#data-exploration)
    4. [Exploratory Data Analysis](#exploratory-data-analysis)
    5. [Pre-processing](#pre-processing)
    6. [Splitting Data](#splitting-data)
    7. [Building and Training the Model](#building-and-training-the-model)
    8. [Making Predictions](#making-predictions)
    9. [Evaluating Model Performance](#evaluating-model-performance)
6. [Streamlit Application](#streamlit-application)
7. [Conclusion](#conclusion)
8. [Important Points](#important-points)

## Introduction

Simple linear regression predicts a response variable (insurance premium) based on a single predictor variable (age). It assumes a linear relationship between the two variables.

## Objectives

- To understand the relationship between age and insurance premium.
- To build a model that predicts the insurance premium based on age.
- To evaluate the model's performance using various metrics.

## Dataset Information

The dataset contains two columns:
- **Age**: The age of the individual.
- **Premium**: The insurance premium amount paid by the individual.

## Prerequisites

To run this project, you'll need:
- Python installed on your computer.
- Basic understanding of Python programming.
- Libraries: pandas, matplotlib, scikit-learn, streamlit

## Steps Involved

### Importing Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```

### Loading the Dataset

```python
df = pd.read_csv("simplelinearregression.csv")
```

### Data Exploration

```python
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.describe().T)
```

### Exploratory Data Analysis

#### Age Distribution

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=20, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.grid(True)
plt.show()
```

#### Premium Distribution

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Premium"], bins=20, edgecolor="black")
plt.xlabel("Premium")
plt.ylabel("Frequency")
plt.title("Distribution of Premium")
plt.grid(True)
plt.show()
```

#### Relationship Between Age and Premium

```python
plt.figure(figsize=(8, 5))
plt.scatter(df["Age"], df["Premium"], alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Premium")
plt.title("Relationship Between Age and Premium")
plt.grid(True)
plt.show()
```

#### Calculate Correlation Coefficient

```python
correlation = df["Age"].corr(df["Premium"])
print("Correlation Coefficient:", correlation)
```

### Pre-processing

```python
X = df[['Age']]
y = df['Premium']
```

### Splitting Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Building and Training the Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Making Predictions

```python
y_pred = model.predict(X_test)
```

### Evaluating Model Performance

#### Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

#### Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
```

#### Root Mean Squared Error (RMSE)

```python
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)
```

#### R² Score

```python
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)
```

## Streamlit Application

The project includes a Streamlit app for interactive predictions.

```python
import streamlit as st

st.title("Insurance Premium Prediction")
st.write("Enter the age to predict the insurance premium")

# User input
age_input = st.number_input("Age", min_value=0, max_value=100, value=25)

# Make prediction
prediction = model.predict([[age_input]])[0]

# Display the prediction
st.write(f"The predicted premium for age {age_input} is: {prediction:.2f}")

# Store the input and prediction in a CSV file
prediction_data = {'Age': [age_input], 'Predicted Premium': [prediction]}
prediction_df = pd.DataFrame(prediction_data)

# Define the file path
file_path = "predictions.csv"

# Check if the file exists
if os.path.exists(file_path):
    # If it exists, append without writing the header
    prediction_df.to_csv(file_path, mode='a', header=False, index=False)
else:
    # If it doesn't exist, create it and write the header
    prediction_df.to_csv(file_path, mode='w', header=True, index=False)

st.write("Prediction has been saved to the file.")
```

### Streamlit App Output

![Streamlit App Output](./Result.jpeg)

## Conclusion

- The simple linear regression model performed well in predicting premiums based on age.
- Metrics such as Mean Squared Error, Mean Absolute Error, Root Mean Squared Error, and R² Score were used to evaluate the model.
- The model showed a strong positive correlation between age and premium, indicating that premiums increase with age.

## Important Points

- Simple linear regression assumes a linear relationship between the independent and dependent variables.
- Understanding the data and checking for assumptions is crucial for building a reliable model.
- This project is a practical introduction to simple linear regression and its application in predicting insurance premiums.

Feel free to explore the code, modify it, and use it for your own purposes. If you have any questions or suggestions, please open an issue or contact me directly.
