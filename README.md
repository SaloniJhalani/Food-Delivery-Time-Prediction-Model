# Food Delivery Time Prediction Model

![Image Description](img/food-delivery.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Website Link](#website-link)
- [Implementation Details](#implementation-details)
    - [Methods Used](#methods-used)
    - [Technologies](#technologies)
    - [Python Packages Used](#python-packages-used)
- [Steps Followed](#steps-followed)
- [Results and Evaluation Criterion](#results-and-evaluation-criterion)
- [Future Improvements](#future-improvements)

  
## Project Overview
This project focuses on developing a food delivery time prediction model. The goal is to estimate the time it takes for food to be delivered to customers accurately. By accurately predicting delivery times, food delivery platforms can enhance customer experience, optimize delivery logistics, and improve overall operational efficiency.

## Data Source
The dataset used for this project can be obtained from [here](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset).

It contains relevant information such as order details, location, city, delivery person information, weather conditions and actual delivery times.

## Website Link

A web-based demonstration of the food delivery time prediction model can be accessed from this [link](https://food-delivery-time-prediction.streamlit.app).

## Implementation Details

### Methods Used
* Machine Learning
* Data Cleaning
* Feature Engineering
* Regression Algorithms

### Technologies
* Python
* Jupyter
* streamlit

### Python Packages Used
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* xgboost

## Steps Followed

1. Data Collection: Gathered the food delivery dataset from the provided data source.
2. Data Preprocessing: Performed data cleaning to handle missing values, outliers, and inconsistencies in the dataset. Conducted feature engineering to extract relevant features for the prediction model.
3. Model Development: Utilized regression algorithms to train a food delivery time prediction model. Explored different models such as linear regression, decision trees, random forests, xgboost to identify the best-performing model.
4. Model Evaluation: Evaluated the performance of the models using appropriate metrics such as mean squared error (MSE),root mean squared error (RMSE) and R2 score.
5. Deployment: Deployed the food delivery time prediction model as a standalone application for real-time predictions.

## Results and Evaluation Criterion

Based on the evaluation results, the best-performing model was **XGBoost** with R2 score of **0.82**

## Future Improvements

Here are some potential areas for future improvements in the project:

* Incorporate more features related to delivery partners, weather conditions, or traffic patterns to enhance prediction accuracy.
* Conduct more comprehensive data analysis to identify additional patterns or correlations that can contribute to better predictions.
* Fine-tune the model parameters to potentially improve performance.





