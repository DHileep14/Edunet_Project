import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
burnoutDF = pd.read_csv("employee_burnout_analysis.csv")

# Convert "Date of Joining" to datetime
burnoutDF["Date of Joining"] = pd.to_datetime(burnoutDF["Date of Joining"])

# Drop irrelevant column
burnoutDF = burnoutDF.drop(['Employee ID'], axis=1)

# Handle missing values
burnoutDF['Resource Allocation'].fillna(burnoutDF['Resource Allocation'].mean(), inplace=True)
burnoutDF["Mental Fatigue Score"].fillna(burnoutDF['Mental Fatigue Score'].mean(), inplace=True)
burnoutDF['Burn Rate'].fillna(burnoutDF['Burn Rate'].mean(), inplace=True)

# Label Encoding
label_encode = preprocessing.LabelEncoder()
burnoutDF['GenderLabel'] = label_encode.fit_transform(burnoutDF['Gender'].values)
burnoutDF['Company_TypeLabel'] = label_encode.fit_transform(burnoutDF['Company Type'].values)
burnoutDF['WFH_Setup_AvailableLabel'] = label_encode.fit_transform(burnoutDF['WFH Setup Available'].values)

# Feature selection
columns = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'GenderLabel', 'Company_TypeLabel', 'WFH_Setup_AvailableLabel']
X = burnoutDF[columns]
y = burnoutDF['Burn Rate']

# Implementing PCA
pca = PCA(0.95)
X_pca = pca.fit_transform(X)

# Data splitting
X_train_pca, X_test, Y_train, Y_test = train_test_split(X_pca, y, test_size=0.25, random_state=10)

# Model Implementation
rf_model = RandomForestRegressor()
rf_model.fit(X_train_pca, Y_train)

abr_model = AdaBoostRegressor()
abr_model.fit(X_train_pca, Y_train)

# Streamlit App
st.title("Employee Burnout Analysis App")

# Sidebar with dataset exploration options
st.sidebar.title("Dataset Exploration")
if st.sidebar.checkbox("Show Dataset"):
    st.write(burnoutDF)

if st.sidebar.checkbox("Show Data Info"):
    st.write("General Information:")
    st.write(burnoutDF.info())

# Data Visualization
st.sidebar.title("Data Visualization")
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.write("Correlation Heatmap:")
    numeric_columns = burnoutDF.select_dtypes(include=[np.number]).columns
    corr = burnoutDF[numeric_columns].corr()
    sns.set(rc={'figure.figsize': (14, 12)})
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

# Model Evaluation
st.sidebar.title("Model Evaluation")
if st.sidebar.checkbox("Evaluate Random Forest Model"):
    st.write("Random Forest Model Evaluation:")
    train_pred_rf = rf_model.predict(X_train_pca)
    test_pred_rf = rf_model.predict(X_test)
    train_r2_rf = r2_score(Y_train, train_pred_rf)
    test_r2_rf = r2_score(Y_test, test_pred_rf)
    st.write("Accuracy score of train data:", round(100 * train_r2_rf, 4), "%")
    st.write("Accuracy score of test data:", round(100 * test_r2_rf, 4), "%")

if st.sidebar.checkbox("Evaluate AdaBoost Model"):
    st.write("AdaBoost Model Evaluation:")
    train_pred_adaboost = abr_model.predict(X_train_pca)
    test_pred_adaboost = abr_model.predict(X_test)
    train_r2_adaboost = r2_score(Y_train, train_pred_adaboost)
    test_r2_adaboost = r2_score(Y_test, test_pred_adaboost)
    st.write("Accuracy score of train data:", round(100 * train_r2_adaboost, 4), "%")
    st.write("Accuracy score of test data:", round(100 * test_r2_adaboost, 4), "%")
