# Nicholas Schafer and Joe Down
# Function library for Mod 1 Project
# King's county housing prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# A function to test autoreloading in Jupyter notebooks
def hello():
    print("hello joe")

# Data loading
def load_kc_data(filename='kc_house_data.csv'):
    df = pd.read_csv(filename)
    print(df.info())
    print(df.head())
    return df

# Data cleaning (removing columns and rows)
# Check how many NA values we have in the various columns of the dataframe
def check_for_na_values(df):
    na_columns = []
    na_values = df.isna().sum()
    for column in df.columns:
        if na_values[column] != 0:
            print(f"{column} has {na_values[column]} NA values, which corresponds to {na_values[column]/len(df)*100:3.2f}% of the data.")
            na_columns.append(column)
    if na_values.sum() == 0:
        print(f"There are no NA values in the dataframe.")
    
    return na_columns

# Convert columns to a specific type
def convert_columns_with_function(df, columns, functions):
    if type(columns) == str:
        df[columns] = functions(df[columns])
    elif type(columns) == list:
        for column_i, column in enumerate(columns):
            df[column] = functions[column_i](df[column])
    return df

def convert_columns_with_types(df, columns, types, fill_na=False):
    if type(columns) == str:
        if fill_na:
            df[columns] = df[columns].fillna(0.0).astype(types)
        else:
            df[columns] = df[columns].astype(types)
    elif type(columns) == list:
        for column_i, column in enumerate(columns):
            if fill_na:
                df[column] = df[column].fillna(0.0).astype(types[column_i])
            else:
                df[column] = df[column].astype(types[column_i])
    return df

def replace_with_year(df, column):
    df = convert_columns_with_function(df, column, pd.to_datetime)
    df[column] = df[column].map(lambda x: x.year)
    return df

# Drop columns
def drop_columns(df, columns):
    df.drop(columns, inplace=True, axis=1)
    print(df.columns)
    return df

# Drop NA rows
def drop_na_rows(df, columns):
    for column in columns:
        df = df[df[column].isna() == False]
    return df

def drop_rows_with_value(df, column, value):
    df = df[df[column] != value]
    return df

# Check for non-numeric columns
def check_for_nonnumeric_columns(df, print_num_unique_values=5):
    from pandas.api.types import is_numeric_dtype
    nonnumeric_columns = []
    for column_i, column in enumerate(df.columns):
        if not is_numeric_dtype(df[column]):
            print(f"{column} is a nonnumeric column ({df.dtypes[column_i]}).")
            nonnumeric_columns.append(column)

    for column in nonnumeric_columns:
        print(f"\n{column} information:")
        print(f"{len(df[column].unique())} unique values.")
        top_values = df[column].value_counts()[:print_num_unique_values]
        print(f"Top {print_num_unique_values} unique values")
        for value_i, (key, value) in enumerate(top_values.items()):
            print(f"{value_i+1}. {key} has {value} entries, which is {value/len(df)*100: .2f}% of the data.")
    
    if len(nonnumeric_columns) == 0:
        print("There are no nonnumeric columns.")
    return nonnumeric_columns

# Check for zeros
def check_for_zeros(df):
    from pandas.api.types import is_numeric_dtype
    columns_with_zeros = []
    for column_i, column in enumerate(df.columns):
        if not is_numeric_dtype(df[column]):
            continue
        num_zeros = len(df)-len(df[column].nonzero()[0])
        if num_zeros > 0:
            columns_with_zeros.append(column)
            print(f"{column} has {num_zeros} zeros, which is {num_zeros/len(df)*100: .2f}% of the data.")
    return columns_with_zeros

# Data imputation
def replace_values_with_another_column(df, values_to_replace, column_to_replace, column_to_use, replace_na=False):
    for value in values_to_replace:
        df.loc[df[column_to_replace] == value,column_to_replace] = df.loc[df[column_to_replace] == value,column_to_use]
    if replace_na:
        df.loc[df[column_to_replace].isna(),column_to_replace] = df.loc[df[column_to_replace].isna(),column_to_use]
    return df

def replace_values_with_value(df, values_to_replace, column_to_replace, value_to_replace_with, replace_na=False):
    for value in values_to_replace:
        df.loc[df[column_to_replace] == value,column_to_replace] = value_to_replace_with
    if replace_na:
        df.loc[df[column_to_replace].isna(),column_to_replace] = value_to_replace_with
    return df

# Adding new types of data by combining columns

# Data transformations 
def log_transform_column(df, column, replace_only_when_improved=True, verbose=True):
    from scipy.stats import normaltest
    pretransformation_statistic, _ = normaltest(df[column])
    transformed_column = df[column].apply(lambda x: np.log(x))
    posttransformation_statistic, _ = normaltest(transformed_column)
    if replace_only_when_improved:
        if posttransformation_statistic < pretransformation_statistic:
            df[column] = transformed_column
            print(f"Tranformed column: {column}")
            if verbose:
                print(f"Replaced {column} with log({column}) because scipy's normaltest indicated an improvement in normality.")
        else:
            print(f"Not tranformed column: {column}")
            if verbose:
                print(f"Did not replace {column} with log({column}) because scipy's normaltest did not indicate an improvement in normality.")
    else:
        df[column] = transformed_column
        print(f"Tranformed column: {column}")
        if posttransformation_statistic < pretransformation_statistic:
            if verbose:
                print(f"Replaced {column} with log({column}); scipy's normaltest indicated an improvement in normality.")
        else:
            if verbose:
                print(f"WARNING: Replaced {column} with log({column}); scipy's normaltest did not indicate an improvement in normality.")
    
    return df

def log_transform_columns(df, columns, replace_only_when_improved=True, verbose=False):
    for column in columns:
        df = log_transform_column(df, column, replace_only_when_improved=replace_only_when_improved, verbose=verbose)
    return df

# Data scaling/normalization
#

# Tests for data normality, heteroscedasticity, etc.
# Correlation and covariance
# Linearity
# Normality
# Heteroscedasticity

# Data visualization

# Feature selection

# Model building

# Cross validation

# Results
# Rsquared
# Adjusted Rsquared

# Results visualization


def location_value_map(longitude, lattidude, price, precision):

    #will use nested loops to move through each house location and find the values of nearby homes according to the precision
    #precision is the percentage of homes to consider as "nearby"
    print('the lattitude is length: '+str(len(lattidude)))
    print('the longitude is length: '+str(len(longitude)))
    
    #initialize an empty list to fill with location values, the mean price of nearby homes
    location_value_list = [0]*len(longitude)
        
    #loop through all the longitude/lattitude pairings
    for house in range(len(longitude)):
        price_holding_list = [0]*int(len(longitude)*(precision/100))
        distance_holding_list=[0]*len(longitude)
        sorted_distance_list=[]
        closest_houses_list=[]
        inner_price_holding_list= [0]*int(len(longitude))
        for nearby_house in range(len(longitude)):

            #initialize holding lists and variable
            
            distance_hold = 0
            #calculate the distance using distance formula between selected house
            #and all the other potentially nearby houses in the set of long/lat
            distance_hold = np.sqrt((lattidude[nearby_house] - lattidude[house]) ** 2 
                + (longitude[nearby_house] - longitude[house]) ** 2)
            distance_holding_list[nearby_house] = distance_hold
            inner_price_holding_list[nearby_house] = price[nearby_house]
        #for each house now make a sorted value ranking list of the nearby houses
        #according to the precision percentage
        zipped_distance_price_list = zip(distance_holding_list, inner_price_holding_list)

        sorted_distance_list = sorted(zipped_distance_price_list, reverse=True, key=lambda x: x[1])
        stop_index = int(len(sorted_distance_list)*precision/100)



        reduced_distance_list = sorted_distance_list[:stop_index]
      

        #for each house, populate a list with the prices of the reduced list of nearby homes
        for i in range(len(reduced_distance_list)):
            price_holding_list[i]= reduced_distance_list[1]

            # print('The price holding list for house number: ')
            # print(str(house))
            # print(' is; ')
            # print(price_holding_list)
            
        price_holding_array=np.array(price_holding_list)
        location_value_list[house]= np.mean(price_holding_array)

    #take the average of those prices and insert into the location_value_list
    #price_holding_array=np.array(price_holding_list)
    #location_value_list[house]= np.mean(price_holding_array)


    return location_value_list


longitude=[3, 7, 9, 5, 3, 8, 1,7]
lattitude=[5, 9, 1, 3, 4, 8, 1, 3]
price=[100, 200, 300, 700, 200, 550, 850, 950]

location_value_map(longitude,lattitude, price, 50)