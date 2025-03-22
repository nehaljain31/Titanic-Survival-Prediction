# Logistic Regression Classification Model

This repository contains a machine learning model using Logistic Regression for binary classification.

## Performance Metrics
- Training Data Accuracy: 80.75%
- Test Data Accuracy: 78.21%

## Project Setup

### 1. Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 2. Data Preparation
- Load the dataset 
- Split the data into features (X) and target (y)
- Split the data into training and testing sets using train_test_split

### 3. Model Training
- Initialize a Logistic Regression model
- Train the model on the training data
- The model achieved 80.75% accuracy on the training data

### 4. Model Evaluation
- Make predictions on the test data
- Evaluate the model performance on test data
- The model achieved 78.21% accuracy on the test data

## Implementation Steps

1. **Data Splitting**
   - The dataset was split into features (X) and target variable (y)
   - Further split into training and testing sets

2. **Model Training**
   - A LogisticRegression model was instantiated
   - The model was fit to the training data using `model.fit(X_train, y_train)`

3. **Making Predictions**
   - Predictions were made on both training data: `x_train_prediction = model.predict(X_train)`
   - And test data: `x_test_prediction = model.predict(X_test)`

4. **Evaluating Model Performance**
   - Accuracy was calculated using sklearn's accuracy_score function
   - Training accuracy: 80.75%
   - Testing accuracy: 78.21%

## Analysis
The model shows good performance with relatively close training and test accuracy scores (difference of approximately 2.5%), suggesting that the model is not severely overfitting.

## Future Improvements
- Feature selection or engineering to improve model performance
- Hyperparameter tuning to optimize the logistic regression model
- Trying different classification algorithms for comparison
- Implementing cross-validation for more robust evaluation
