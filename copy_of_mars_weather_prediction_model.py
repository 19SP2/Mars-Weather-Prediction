# -*- coding: utf-8 -*-
"""Copy of Mars Weather Prediction Model

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qL5KenBvJq4leLxc6WgPXfMTtJC21ovg

## Mars weather data from NASA’s InSight Mars lander
NASA’s InSight Mars lander takes continuous weather measurements (temperature, wind, pressure) on the surface of Mars at Elysium Planitia, a flat, smooth plain near Mars’ equator.
This API provides per-Sol summary data for each of the last seven available Sols (Martian Days).

The API doc: https://api.nasa.gov/assets/insight/InSight%20Weather%20API%20Documentation.pdf

Check out NASA's Open APIs: https://api.nasa.gov/?ref=freepublicapis.com

## Problem Understanding and Definition

If we are to send humans on Mars we must know its weather conditions.

Predicting weather on Mars has many advantages, it helps in  understanding the planets atmosphere and its significance in the solar system. It is very crucial to understand the weather of the planet for future exploration. It also helps in planning robotic missions as well.

NASA's InSight Lander launched on May 5, 2018 collected data from the planets atmosphere. This Machine Learning Model uses supervised learning to predict data collected from the lander.

Due to the challenges of the mission, very little data was collected by the lander, therefore creating a model with this data will result in a rather inaccurate model. But with more data the model will become accurate.

## Data Collection and Preparation
"""

# importing all necessary libraries
import requests
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# connecting api
api_endpoint='https://api.nasa.gov/insight_weather/?api_key=xX2zNrbsj5op885F4cjhsdUfyPVhA0Di8hMG3XfF&feedtype=json&ver=1.0'
api_endpoint

"""The rate limits for the DEMO_KEY are:

Hourly Limit: 30 requests per IP address per hour

Daily Limit: 50 requests per IP address per day
"""

# collecting api data into a variable in json format
json_data = requests.get(api_endpoint).json()
mars_data = json_data

# finding the how many sols' data is available in the api
a = mars_data["validity_checks"]["sols_checked"]
no_of_sol = list(map(int, a))
# in the validity checks, sols checked, the first and last sol have no avialble data, therefore will be deleted
del no_of_sol[0]
del no_of_sol[-1]
print(f"Sols in the dataset: \n{no_of_sol}")

# atmospheric temperature, Minimum data sample over the sol
print("Minimum Atmospheric Temperature data sample over the sol °F:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["AT"]["mn"]
        print(f"{x} °F")
        break

# atmospheric temperature, Maximum data sample over the sol
print("Maximum Atmospheric temperature data sample over the sol °F:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["AT"]["mx"]
        print(f"{x} °F")
        break

# minimum horizontal wind speed
print("Minimum Horizontal Wind Speed metres per second:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["HWS"]["mn"]
        print(f"{x} m/s")
        break

# maximum horizontal wind speed
print("Maximum Horizontal Wind Speed metres per second:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["HWS"]["mx"]
        print(f"{x} m/s")
        break

# minimum atmosphereic pressure Pascals
print("Minimum Atmosphereic Pressure Pascals:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["PRE"]["mn"]
        print(f"{x} Pa")
        break

# maximum atmosphereic pressure Pascals
print("Maximum Atmosphereic Pressure Pascals:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["PRE"]["mx"]
        print(f"{x} Pa")
        break

# Time of first datum, of any sensor, for the Sol (UTC; YYYY-MM-DDTHH:MM:SSZ)
print("Time of first datum:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["First_UTC"]
        print(x)
        break

# Time of last datum, of any sensor, for the Sol (UTC; YYYY-MM-DDTHH:MM:SSZ)
print("Time of last datum:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["Last_UTC"]
        print(x)
        break

# Season on Northern Mars
print("Season on Northern Mars:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["Northern_season"]
        print(x)
        break

# Season on Southern Mars
print("Season on Southern Mars:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["Southern_season"]
        print(x)
        break

# Month ordinal
print("Month Ordinal:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["Month_ordinal"]
        print(x)
        break

# maximum atmosphereic pressure Pascals
print("Wind direction data for most common compass point:")
for i in no_of_sol:
    while True:
        x = mars_data[f"{i}"]["WD"]["most_common"]["compass_point"]
        print(x)
        break

# all the data in the api will be stored in a tabular form using PrettyTable library
data_table = PrettyTable()
for data in mars_data:
    for i in no_of_sol:
      # creating a table with the data collected from API
        at_mx = mars_data[f"{i}"]["AT"]["mx"]
        at_mn = mars_data[f"{i}"]["AT"]["mn"]
        hws_mx = mars_data[f"{i}"]["HWS"]["mx"]
        hws_mn = mars_data[f"{i}"]["HWS"]["mn"]
        pre_mx = mars_data[f"{i}"]["PRE"]["mx"]
        pre_mn = mars_data[f"{i}"]["PRE"]["mn"]
        first = mars_data[f"{i}"]["First_UTC"]
        last = mars_data[f"{i}"]["Last_UTC"]
        ns = mars_data[f"{i}"]["Northern_season"]
        ss = mars_data[f"{i}"]["Southern_season"]
        mo = mars_data[f"{i}"]["Month_ordinal"]
        wd = mars_data[f"{i}"]["WD"]["most_common"]["compass_point"]
        data_table.add_row((i, f"{at_mn:.3f}", f"{at_mx:.3f}", f"{hws_mn:.3f}",  f"{hws_mx:.3f}", f"{pre_mx:.3f}", f"{pre_mn:.3f}", first, last, f"{ns.upper()}", f"{ss .upper()}", mo, f"{wd .upper()}"))
    break

data_table.field_names = ['Sol', 'Min AT °F', 'Max AT °F', 'Min HWS m/s', 'Max HWS m/s', 'Max Pressure Pa', 'Min Pressure Pa', 'First UTC', 'Last UTC', 'Northern Season', 'Southern Season', 'Month Ordinal', 'Wind Direction']
print(f"This is the final data table created using PrettyTable library:\n{data_table}")

# from the table above, I filtered the data to show only relevent columns and rows
# I converted the table into a csv file to make a DataFrame of it
with open('mars_weather_data.csv', 'w', newline='') as file_output:
    file_output.write(data_table.get_csv_string())

data = pd.read_csv('mars_weather_data.csv')

data = data.drop(columns=['First UTC', 'Last UTC', 'Wind Direction', 'Northern Season', 'Southern Season'])
#reording the columns to have Min AT °F	and Max AT °F at the last
data = data.iloc[:,[0, 3,4,5,6,7,1,2]]
data = pd.DataFrame(data)
data

"""## Exploratory Data Analysis (EDA)"""

data.info()

data.describe()

# finding the co-relation between the rows and columns
data.corr()

# heatmap to find the relation between columns
plt.figure(figsize = (8,5))
sns.heatmap(data.corr(), cbar = False, annot = True)

"""Evaluation of the heat map

1.   Sol and Month Ordinal influence the min and max atmospheric temperature the most, since in the heat map they have positive values.
1.   Min and max atmospheric temperature influence each other as well, since in the heat map they have positive values.
"""

# to find out any null or missing values in the data
data.isnull()

"""The data has no null or missing values."""

# histograms to show the value or count of the data
# number of rows and columns to represent the histogram
n_rows=2
n_cols=3

# Creating the subplots
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
fig.set_size_inches(10, 5)
for i, column in enumerate(data.iloc[:, :-2].columns):
    sns.histplot(data[column], ax=axes[i//n_cols, i % n_cols], kde=True, color='red')
plt.tight_layout()

# calculating unique values in the dataset
for i in data.columns:
  unique_values = data[i].unique()
  print(f'The Column {i} has {len(unique_values)} unique values.')

"""The data does not have a categorical nature, all the values are different and can not be classified into certain categories. This reduces the accuracy of the model.

## Feature Engineering
"""

# creating functions to make 2 predictions
def printPredictions(y_true,y_pred, count):
  print('Predictions:')
  print(y_true.assign(
      Y1_pred = y_pred[:,0], # predicting min AT
      Y2_pred = y_pred[:,1] # predicting max AT
  ).head(count).to_markdown(index = False))

# r2 is the Coefficient of determination, it greater the number the better the model
def showResults(y_true, y_pred, count = 1):
  print('R2 Score: ',r2_score(y_true,y_pred))
  print('Mean Squared Error: ',mean_squared_error(y_true,y_pred))
  print('Mean Absolute Error: ',mean_absolute_error(y_true,y_pred))
  print("Root Mean Square Error",mean_squared_error(y_true=y_test,y_pred=y_pred))
  mape = mean_absolute_percentage_error(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
  print("Mean Absolute Percentage Error", mape)
  printPredictions(y_true,y_pred, count)
  # plot original vs predicted values
  plt.scatter(y_test, y_pred)
  plt.xlabel('Actual Values')
  plt.ylabel('Predicted Values')
  plt.title('Actual vs Predicted')
  plt.show()
  # This plot shows the residuals (differences between the predicted and actual petal widths) against the predicted values.
  residuals = y_test - y_pred
  plt.scatter(y_pred, residuals)
  plt.xlabel('Predicted')
  plt.ylabel('Residuals')
  plt.title('Residual Plot')
  plt.axhline(y=0, color='r', linestyle='--')
  plt.show()

# splitting data into training and testing with 20% test size
X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,:-2], data.iloc[:,-2:], test_size = 0.2, random_state = 42)
print(X_train.shape,X_test.shape)
print(y_train.shape, y_test.shape)

# using normalization for feature scaling
# the initial distribution of max horizontal wind speed
age = X_train['Max HWS m/s']
plt.figure(figsize=(5,2))
plt.hist(age, bins=30, alpha=0.5, label='Original', color='blue')
plt.legend(prop={'size': 16})
plt.title('Histogram with Original Max Horizontal Wind Speed');
plt.xlabel('Max Horizontal Wind Speed'); plt.ylabel('Count');
plt.show()

# Normalize Max Horizontal Wind Speed on the training set
normalizer = MinMaxScaler()
X_train['Max HWS m/s'] = normalizer.fit_transform(X_train['Max HWS m/s'].values.reshape(-1,1))
X_test['Max HWS m/s'] = normalizer.transform(X_test['Max HWS m/s'].values.reshape(-1,1))

# histogram to plot the normalized values
plt.figure(figsize=(5,2))
plt.hist(X_test['Max HWS m/s'], bins=30, alpha=0.5, label='Normalized', color = 'indigo')
plt.legend(prop={'size': 16})
plt.title('Histogram with Normalized Max Horizontal Wind Speed')
plt.xlabel('Normalized Max Horizontal Wind Speed')
plt.ylabel('Count')
plt.show()

# the initial distribution of Min Pressure Pa
age = X_train['Min Pressure Pa']
plt.figure(figsize=(5,2))
plt.hist(age, bins=30, alpha=0.5, label='Original', color = 'orange')
plt.legend(prop={'size': 16})
plt.title('Histogram with Original Min Pressure Pa');
plt.xlabel('Min Pressure Pa'); plt.ylabel('Count');
plt.show()

normalizer = MinMaxScaler()
X_train['Min Pressure Pa'] = normalizer.fit_transform(X_train['Min Pressure Pa'].values.reshape(-1,1))
X_test['Min Pressure Pa'] = normalizer.transform(X_test['Min Pressure Pa'].values.reshape(-1,1))

# histogram to plot the normalized values
plt.figure(figsize=(5,2))
plt.hist(X_test['Min Pressure Pa'], bins=30, alpha=0.5, label='Normalized', color = 'red')
plt.legend(prop={'size': 16})
plt.title('Histogram with Normalized Min Pressure Pa')
plt.xlabel('Normalized Min Pressure Pa')
plt.ylabel('Count')
plt.show()

# the initial distribution of Max Pressure Pa
age = X_train['Max Pressure Pa']
plt.figure(figsize=(5,2))
plt.hist(age, bins=30, alpha=0.5, label='Original', color = 'yellow')
plt.legend(prop={'size': 16})
plt.title('Histogram with Original Max Pressure Pa');
plt.xlabel('Max Pressure Pa'); plt.ylabel('Count');
plt.show()

normalizer = MinMaxScaler()
X_train['Max Pressure Pa'] = normalizer.fit_transform(X_train['Max Pressure Pa'].values.reshape(-1,1))
X_test['Max Pressure Pa'] = normalizer.transform(X_test['Max Pressure Pa'].values.reshape(-1,1))

plt.figure(figsize=(5,2))
plt.hist(X_test['Max Pressure Pa'], bins=30, alpha=0.5, label='Normalized', color = 'gold')
plt.legend(prop={'size': 16})
plt.title('Histogram with Normalized Max Pressure Pa')
plt.xlabel('Normalized Max Pressure Pa')
plt.ylabel('Count')
plt.show()

"""## Model Training and Evaluation

Trying different supervised learning models.
"""

# Linear Regression model
from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(X_train,y_train)
print('The results of the Linear Regression model:')
showResults(y_test,linear.predict(X_test))

# Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor

rdf = RandomForestRegressor()
rdf.fit(X_train,y_train)
print('The results of the Random Forest Regression model:')
showResults(y_test,rdf.predict(X_test))

# KN Regression model
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_train,y_train)
print('The results of the KN Regression model:')
showResults(y_test,knn.predict(X_test))

# Multiple Output Regression and Support Vector Machine model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

svm_multi = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
svm_multi.fit(X_train,y_train)
print('The results of the Multiple Output Regression model:')
showResults(y_test,svm_multi.predict(X_test))

# Regressor Chain model to link all the models
from sklearn.multioutput import RegressorChain

# Defining the chained multioutput model
svm_chain = RegressorChain(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))

svm_chain.fit(X_train,y_train)
print('The results of the SVM Chain Regression model:')
showResults(y_test,svm_chain.predict(X_test))

"""The r2 score of the last model is the lowest as it combines all the models together to give the most accurate results.

---

## Model Deployment
"""

import gradio as gr

def display(Sol, Min_Horizontal_Wind_Speed_ms, Max_Horizontal_Wind_Speed_ms, Min_Pressure_Pa, Max_Pressure_Pa, Month_Ordinal):
  features = np.array([[Sol, Min_Horizontal_Wind_Speed_ms, Max_Horizontal_Wind_Speed_ms, Min_Pressure_Pa, Max_Pressure_Pa, Month_Ordinal]])
  prediction = svm_chain.predict(features)
  return prediction[0,0].round(2), prediction[0,1].round(2)

inter_face = gr.Interface(fn = display,
                          title = "Mars Weather Prediction",
                          description = "The is a Supervised Machine Learning Model created to predict the weather on Mars. The data is collected from NASA's Open AI. The model predicts Maximum and Minimum Atmospheric Tempertaure °F",
                          theme = "allenai/gradio-theme", #theme template from hugging face theme gallery
                          inputs = [gr.Number(label='Sol (Try 700)'), gr.Number(label='Min Horizontal Wind Speed ms (Try 0.12)'), gr.Number(label='Max Horizontal Wind Speed ms (Try 19)'), gr.Number(label='Min Pressure Pa (Try 717)'), gr.Number(label='Max Pressure Pa (Try 763)'), gr.Number(label='Month Ordinal (Try 14)')],
                          outputs = [gr.Number(label='Max Atmospheric Temperature °F'), gr.Number(label='Min Atmospheric Temperature °F')],
                          flagging_mode="never")

inter_face.launch(debug=True)
