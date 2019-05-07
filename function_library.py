# Nicholas Schafer and Joe Down
# Function library for Mod 1 Project
# King's county housing prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### TO DO
# work on visualizations H
# work on presentation H
# Build final jupyter notebook H
# Review project/presentation description H
# Interpretation of coefficients H

# power transform L
# neighborhood price/score L
# work on feature selection L
# stepwise selection L
# check for outliers L

# A function to test autoreloading in Jupyter notebooks
def hello():
    print("hello joe")

# Data loading
def load_kc_data(filename='kc_house_data.csv', verbose=True):
    df = pd.read_csv(filename)
    if verbose:
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

# Adding new types of data
# THIS IS UNTESTED
def transform_column_to_categorical_columns(df, column, bins, prefix=None, drop_original_column=True):
    if prefix == None:
        prefix = column
    bins_column = pd.cut(df[column], bins)
    bins_column = bins_column.cat.as_unordered()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df.drop(labels=[column], inplace=True, axis=1)
    return df

def binarize_column(df, column, new_column_name=None, drop_original_column=True):
    if new_column_name == None:
        df[column] = (np.abs(df[column]) > 0).astype(int)
    else:
        df[new_column_name] = (np.abs(df[column]) > 0).astype(int)
        df.drop(labels=[column], inplace=True, axis=1)
    return df

def add_dummy_zipcodes(df, drop_zipcode_column=True):
    dummy_zipcodes = pd.get_dummies(df["zipcode"])
    df = pd.concat([df, dummy_zipcodes], axis=1)
    if drop_zipcode_column:
        df.drop(labels=["zipcode"], axis=1, inplace=True)
    return df

# Data transformations 
def log_transform_column(df, column, replace_only_when_improved=True, verbose=True, mute=False):
    from scipy.stats import normaltest
    pretransformation_statistic, _ = normaltest(df[column])
    transformed_column = df[column].apply(lambda x: np.log(x))
    posttransformation_statistic, _ = normaltest(transformed_column)
    transformed = False
    if replace_only_when_improved:
        if posttransformation_statistic < pretransformation_statistic:
            df[column] = transformed_column
            transformed = True
            if not mute:
                print(f"Tranformed column: {column}")
            if verbose:
                print(f"Replaced {column} with log({column}) because scipy's normaltest indicated an improvement in normality.")
        else:
            if not mute:
                print(f"Not tranformed column: {column}")
            if verbose:
                print(f"Did not replace {column} with log({column}) because scipy's normaltest did not indicate an improvement in normality.")
    else:
        df[column] = transformed_column
        transformed = True
        if not mute:
            print(f"Tranformed column: {column}")
        if posttransformation_statistic < pretransformation_statistic:
            if verbose:
                print(f"Replaced {column} with log({column}); scipy's normaltest indicated an improvement in normality.")
        else:
            if verbose:
                print(f"WARNING: Replaced {column} with log({column}); scipy's normaltest did not indicate an improvement in normality.")
    
    return df, transformed

def log_transform_columns(df, columns, replace_only_when_improved=True, verbose=False, mute=False):
    transformed_columns = []
    for column in columns:
        df, transformed = log_transform_column(df, column, replace_only_when_improved=replace_only_when_improved, verbose=verbose, mute=mute)
        if transformed:
            transformed_columns.append(column)
    return df, transformed_columns

# Data scaling/normalization
def scale_and_normalize_columns(df, columns, scaling_functions, fit_df=None):
    for column_i, column in enumerate(columns):
        if not type(fit_df) == None:
            df[column] = scaling_functions[column_i](df[column], fit_column=fit_df[column])
        else:
            df[column] = scaling_functions[column_i](df[column])
    return df

def min_max(column, fit_column=None):
    from sklearn.preprocessing import MinMaxScaler
    column = np.array(column).reshape(-1,1)
    scaler = MinMaxScaler()
    if not type(fit_column) == None:
        fit_column = np.array(fit_column).reshape(-1,1)
        scaler.fit(fit_column)
    else:
        scaler.fit(column)
    column = scaler.transform(column)
    return column

def standardization(column, fit_column=None):
    from sklearn.preprocessing import StandardScaler
    column = np.array(column).reshape(-1,1).astype(float)
    scaler = StandardScaler()
    if not type(fit_column) == None:
        fit_column = np.array(fit_column).reshape(-1,1).astype(float)
        scaler.fit(fit_column)
    else:
        scaler.fit(column)
    column = scaler.transform(column)
    return column

def mean_normalization(column, fit_column=None):
    column = np.array(column)
    if not type(fit_column) == None:
        fit_column = np.array(fit_column)
        column_mean = np.mean(fit_column)
        column_max = np.max(fit_column)
        column_min = np.min(fit_column)
    else:
        column_mean = np.mean(column)
        column_max = np.max(column)
        column_min = np.min(column)
    column = (column-column_mean)/(column_max-column_min)
    return column

def unit_vector_transformation(column, fit_column=None):
    column = np.array(column)
    if not type(fit_column) == None:
        fit_column = np.array(fit_column)
        column /= np.linalg.norm(fit_column)
    else:
        column /= np.linalg.norm(column)
    return column

# Tests on the data
# Linearity
def check_for_linear_correlations(df, threshold=0.7, target=None):
    #print(df.corr())
    #print(df.corr()>threshold)
    correlations = df.corr()
    
    if not target == None:
        print(f"Correlations with target ({target}):")
        target_correlations = correlations[target]
        for feature, correlation_coefficient in target_correlations.items():
            if feature == target:
                continue
            print(f"{feature}: {correlation_coefficient: .2f}")
        for feature, correlation_coefficient in target_correlations.items():
            if feature == target:
                continue
            if correlation_coefficient > threshold:
                print(f"{feature} has a significant correlation ({correlation_coefficient: .2f}) with {target} according to a threshold of {threshold}.")
    
    for column_i, column in enumerate(correlations.columns):
        correlated_columns = [x for x in np.array(correlations.columns[correlations[column]>threshold]) if not x == column]
        num_correlated_columns = len(correlated_columns)
        if num_correlated_columns > 0:
            print(f"{column} is correlated with {num_correlated_columns} according to a threshold of {threshold} ({correlated_columns}).")

# Normality
def check_for_normality(df, threshold=0.0005, verbose=False):
    from scipy.stats import normaltest
    for column in df.columns:
        statistic, pvalue = normaltest(df[column])
        if verbose:
            print(f"{column} statistic: {statistic: .2f}, pvalue: {pvalue: e}")
        if pvalue < threshold:
            print(f"{column} is not normal according to a pvalue threshold of {threshold}")

# Heteroscedasticity

# Data visualization

# Feature selection
def recursive_feature_elimination_get_features(df, target, n_features_to_select=None):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    predictors = df.drop(labels=[target], axis=1)

    linreg = LinearRegression()
    selector = RFE(linreg, n_features_to_select = n_features_to_select)
    selector = selector.fit(predictors, df[target])
    return list(predictors.columns[selector.support_])

def stepwise_selection_sklearn(df, target,
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    from sklearn.feature_selection import f_regression
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    X = df.drop(labels=[target], axis=1)
    y = df[target]
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column_i, new_column in enumerate(excluded):
            new_pval[new_column] = f_regression(X, y)[1][new_column_i]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        pvalues = f_regression(X, y)[1]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# Model building and prediction
def build_linear_model_sklearn(df, target):
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    y = df[target]
    predictors = df.drop(labels=[target], axis=1)
    linreg.fit(predictors, y)
    return linreg

def get_predictions_from_linear_model_sklearn(df, linreg, target):
    predictors = df.drop(labels=[target], axis=1)
    return linreg.predict(predictors)

def get_residuals_from_linear_model_sklearn(df, linreg, target):
    return get_predictions_from_linear_model_sklearn(df, linreg, target) - df[target]

def get_RMSE_from_linear_model_sklearn(df, linreg, target):
    return np.sqrt(np.mean(get_residuals_from_linear_model_sklearn(df, linreg, target)**2))

# Results
# Rsquared
def get_rsq_sklearn(df, linreg, target):
    from sklearn.metrics import r2_score
    predictions = get_predictions_from_linear_model_sklearn(df, linreg, target)
    reality = df[target]
    return r2_score(reality, predictions)

# Adjusted Rsquared
def get_adjusted_rsq_sklearn(df, linreg, target):
    rsq = get_rsq_sklearn(df, linreg, target)
    n = len(df)
    p = len(df.columns)-1
    return 1-(1-rsq)*((n-1)/(n-p-1))

# Cross validation
def perform_cross_validation_sklearn(df, target, cv=5):
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    y = df[target]
    predictors = df.drop(labels=[target], axis=1)
    scores = cross_val_score(linreg, predictors, y, cv=cv)
    return scores

# Results visualization
def plot_predictions_vs_reality(df, target, predictions):
    plt.figure()
    plt.scatter(df[target], predictions)


def location_value_map(longitude, lattidude, price, precision):

    #will use nested loops to move through each house location and find the values of nearby homes according to the precision
    #precision is the percentage of homes to consider as "nearby"
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



#pass in a dataframe, a zipcode series, and a price series and a new dataframe will be returned with a column of zipcode rankings added
def add_zipcode_ranking_feature(df, zipcode, price):
    
    #Set up a temporary database grouped by zipcode
    by_zipcode = df.groupby(zipcode)[price].describe()
    by_zipcode.head()
    by_zipcode[zipcode]= by_zipcode.index
    by_zipcode.sort_values('mean', ascending=False, inplace = True)

    #loop through the grouped data base and create a zipcode rank list
    zipcode_rank_list = [0]*len(by_zipcode.zipcode)
    for index, zipcode in enumerate(by_zipcode.zipcode):
        zipcode_rank = [0,0]
        zipcode_rank[0]= zipcode
        zipcode_rank[1]= index**2
        zipcode_rank_list[index]=zipcode_rank

    updated_df = df
    updated_df['zipcode_rank'] = [0]*len(df.index)
    
    #Add the ranking values to a new column in the updated dataframe
    rank_holding_list = [0]*len(updated_df.zipcode)
    for index, zipcode in enumerate(updated_df.zipcode):
        for rank in zipcode_rank_list:
            if zipcode==rank[0]:
                rank_holding_list[index]=rank[1]


    updated_df['zipcode_rank']=rank_holding_list

    return updated_df


# df = pd.read_csv('https://raw.githubusercontent.com/learn-co-students/houston-ds-042219-mod1-project/master/kc_house_data.csv')
# add_zipcode_ranking_feature(df, 'zipcode', 'price')

def add_time_since_renovated_feature(df, date_of_purchase, year_renovated):
    df['time_since_renovated']= df[date_of_purchase]-df[year_renovated]
    return df

def add_cluster_label_feature(df, n_clusters=10, fit_df=None, drop_original_column=True):
    from sklearn.cluster import KMeans
    X = list(zip(df['lat'].tolist(), df['long'].tolist()))
    if type(fit_df) == None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit(X)
        df['cluster'] = kmeans.labels_
    else:
        fit_X = list(zip(fit_df['lat'].tolist(), fit_df['long'].tolist()))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(fit_X)
        labels = kmeans.fit_predict(X)
        df['cluster'] = labels

    dummies = pd.get_dummies(df['cluster'], prefix='cluster')
    df = pd.concat([df, dummies], axis=1)
    if drop_original_column:
        df.drop(labels=['cluster'], inplace=True, axis=1)
    return df

def fill_na_with_zero(df, column):
    updated_df=df
    updated_df[column]= df[column].fillna(0)
    return updated_df

def fill_na_with_median(df, column):
    updated_df=df
    updated_df[column]=df[column].fillna(df[column].median())
    return updated_df

def convert_float_to_datetime(df, column):
    updated_df = df
    updated_df[column] = df[column].astype(int)
    holding_list_for_series = [0]*len(df)
    holding_string_for_date = '-01-01'

    for index, house in  enumerate(updated_df.column):
        holding_string_for_year = ''
        holding_string_together = ''
        if updated_df[column][index] != 0:
            holding_string_for_year = str(updated_df[column][index])
            holding_string_together = holding_string_for_year + holding_string_for_date
            #print(df.column[index].join(' '))
            holding_list_for_series[index] = holding_string_together
        else: 
            holding_string_for_year = str(updated_df.yr_built[index])
            holding_string_together = holding_string_for_year + holding_string_for_date
            #print(df.column[index].join(' '))
            holding_list_for_series[index] = holding_string_together
    
    updated_df = df
    updated_df[column] = holding_list_for_series
    return updated_df

def date_to_datetime(df, column):
    updated_df = df
    updated_df[column] =  pd.to_datetime(df[column]) #, format='%d%b%Y:%H:%M:%S.%f')
    return df

def time_since_rennovated_added_in_days(df, date, rennovated):
    difference_list = len(date)
    for i, date in enumerate(date):
        difference_list[i] = date - rennovated[i]

    updated_df = df  
    updated_df['days_since_rennovated']=difference_list
    return updated_df
    
