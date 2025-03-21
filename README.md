# AIDE_Heart_Disease
# Group : Doha Krayani and Fatima Hamade
## Project Description: 
Create an ensemble of classifiers to predict heart disease risk. 
Generate patient-specific explanations using LIME and SHAP, designing visualizations 
that could help both doctors and patients understand individual risk factors and 
potential interventions. 

## Features
Data Preprocessing: Handles missing values, scales numerical features, and one-hot encodes categorical features.
Ensemble Learning:
Bagging with Decision Trees
Boosting with AdaBoost
Random Forest
Model Interpretability: Uses SHAP to explain model predictions and LIME for local interpretability.

## Dataset
File: heart.csv .
Description: Contains features related to heart health. Typical attributes include age, sex, cholesterol levels, and other clinical measurements. The target variable indicates the presence of heart disease.

## Dependencies
Python 3.7+
Libraries:pip install pandas numpy shap matplotlib scikit-learn lime

## Code Structure
### 1. Preprocessing Pipeline:
Numerical features: Imputed with mean and standardized.
Categorical features: Imputed with mode and one-hot encoded.
Combined using ColumnTransformer.
### 2. Ensemble Models:
Bagging: 5 Decision Trees trained on bootstrapped samples (Test Accuracy: 97.07%).
AdaBoost: 50 estimators with Decision Tree base learners (Test Accuracy: 98.54%).
Random Forest: 100 trees with max depth 5 (Test Accuracy: 85.37%).
### 3. Visualizations:
We use LIME and SHAP for model interpretability.
LIME explains individual predictions by approximating the model’s behavior locally.
SHAP calculates feature contributions using Shapley values to measure the impact of each feature.

## Usage
Place heart.csv in the working directory.
Run all cells in the notebook to:
Preprocess data.
Train and evaluate ensemble models.
Visualize SHAP and Lime.

## Results
Best Model: AdaBoost achieved the highest test accuracy (98.54%).
SHAP Insights: Identifies key features driving predictions.
Lime: It outputs a local explanation for a single instance, displaying feature importance and how each feature influences the model’s prediction.
