# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import Validators as vld
import os
import pandas as pd
import perceptron as prc
import numpy as np
import MLP
import MLfunctions as mlf
import math


###DATA PREPROCESING###
#creatig a path variable to work wit the right directory
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

dataset = pd.read_csv(os.path.join(__location__, 'housing.csv')) #we insert the data from the dsv in a dataframe
dataset = dataset.dropna() #we drop the rows with NaN values
dataset = dataset.sample(frac=1, random_state=1).reset_index() #we shuffle the dataset before every session

datanum = dataset[dataset.columns.difference(['ocean_proximity'])]
datacat = dataset[['ocean_proximity']]

enc = OneHotEncoder(handle_unknown='ignore') #a HOTencoder object to encode categorical data
scaler = StandardScaler() #a normalizer

scaler.fit(datanum) # we fit the encoder to the numerical data
enc.fit(datacat) # we fit the encoder to the categorical data

# we create the new dataframes with categorical and numerical data separately
num = pd.DataFrame(scaler.transform(datanum), columns = datanum.columns)
cat = pd.DataFrame(enc.transform(datacat).toarray())

#then we concatenate the numeric with the hotencoded categorical one in a single Dataframe
temp = pd.concat([num, cat], axis=1)

#and we finish the preprocessing by create the NDarray with the inputs and the target outputs
X = temp.iloc[:, temp.columns != "median_house_value"].values #we convert it to a vector (input data)

#Perceptron preprocessing#
y_norm = temp.iloc[:, temp.columns == "median_house_value"].values #we convert it to a vector (target data for perceptron)

#MLP preprocesing#
y_raw = dataset.iloc[:, dataset.columns == "median_house_value"].values
grouper = mlf.sturgesrule(y_raw)
y_grouped = mlf.sturges_grouper(y_raw, grouper.step) # we categorize raw data according to sturges rule
y_groups = mlf.one_hot(y_grouped)

###PERCEPTRON###
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Read the CSV file into a pandas DataFrame, skipping the header row
df = pd.read_csv(os.path.join(__location__, 'train.csv'))

# Rename the columns if needed
# Assuming you have a list of column names, replace ['First_Column', 'Column2', 'Column3', ...] with your actual column names
column_names = ['First_Column'] + [f'Column{i}' for i in range(2, len(df.columns) + 1)]
df.columns = column_names

# Convert the first column to integers
df['First_Column'] = df['First_Column'].astype(int)

# Convert other columns to integers
df.iloc[:, 1:] =  df.iloc[:, 1:].astype(int)

layer1 = prc.layer.spawn(784)
layer2 = prc.layer.spawn(10)
layer3 = prc.layer.spawn(10)
layer4 = prc.layer.spawn(10)
ml2 = prc.mlp(layer1, layer3, [layer2])

one_hot = mlf.one_hot_encoder(df['First_Column'])

for i in range(df.shape[0] - 40000):
    example = pd.DataFrame(ml2.propagate_forward(df.iloc[i, 1:]))
    guess = pd.DataFrame(ml2.guess())
    ml2.propagate_backward(one_hot.encoded[i])