# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:44:56 2023

@author: gianl
"""

from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt





# Specify the path to your ARFF file
arff_file_path = r'C:\Users\gianl\Downloads\ACSIncome_state_number.arff'

# Load and read the ARFF file
data, meta = arff.loadarff(arff_file_path)
    
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

df.head()

df.shape

df

df = df.drop(columns=['ST'])

df['POBP'].unique()  

df['POBP'].max()  


# Assuming 'SEX', 'RAC1P', and 'POBP' are the column names in your DataFrame
columns_to_plot = ['SEX', 'RAC1P', 'POBP']

# Sex, Race and Native Country

for column in columns_to_plot:
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=20, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

#Distribution of education level and occupation

for column in columns_to_plot:
    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=20, edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
# Create a new column 'PINCP_binary' with the binary labels
df['PINCP_binary'] = df['PINCP'].apply(lambda x: '>50k' if x > 50000 else '<=50k')

# Drop the original 'PINCP' column if needed
df = df.drop(columns=['PINCP'])    
    
df.head()

df['PINCP_binary'].unique()

df['PINCP_binary'].value_counts()

#Distribution of salary among the general population
# Create a histogram of the 'PINCP_binary' column
plt.figure(figsize=(6, 4))
df['PINCP_binary'].value_counts().plot(kind='bar', edgecolor='black')
plt.title('Income Distribution in the General Population')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.show()


# Get unique gender values
unique_genders = df['SEX'].unique()

# Create a separate histogram for each gender
for gender in unique_genders:
    gender_data = df[df['SEX'] == gender]['PINCP_binary']
    gender_label = 'Male' if gender == 1 else 'Female'

    plt.figure(figsize=(8, 6))
    plt.hist(gender_data, bins=20, edgecolor='black', alpha=0.5)
    plt.title(f'Income Distribution - {gender_label}')
    plt.xlabel('Income')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()



# Mapping of race codes to descriptions
race_mapping = {
    1: 'White alone',
    2: 'Black or African American alone',
    3: 'American Indian alone',
    4: 'Alaska Native alone',
    5: 'American Indian and Alaska native tribes specified; or American Indian or Alaska Native, not specified and no other races',
    6: 'Asian alone',
    7: 'Native Hawaiian and Other Pacific Islander alone',
    8: 'Some Other Race alone',
    9: 'Two or More races'
}

# Get unique race codes
unique_race_codes = df['RAC1P'].unique()

# Create a separate histogram for each race
for race_code in unique_race_codes:
    race_data = df[df['RAC1P'] == race_code]['PINCP_binary']
    race_label = race_mapping.get(race_code, f'Race {race_code}')

    plt.figure(figsize=(8, 6))
    plt.hist(race_data, bins=20, edgecolor='black', alpha=0.5)
    plt.title(f'Income Distribution - {race_label}')
    plt.xlabel('Income')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Create a histogram of hours worked per week
plt.figure(figsize=(10, 6))
plt.hist(df['WKHP'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Hours Worked per Week')
plt.xlabel('Hours per Week')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Create a new dataset to be modified 
dfv2 = df.copy()
dfv2.head()

# Check for missing values in each column
dfv2.isnull().sum()

# Normalize continuous features
continuous_features = ['AGEP', 'WKHP']
X = dfv2[continuous_features]
dfv2[continuous_features] = (X - np.mean(X))/ np.std(X)
dfv2.head()


# Create a new column 'US_or_NotUS' and initialize it with 0
dfv2['US_or_NotUS'] = 0

# Set the 'US_or_NotUS' column to 1 where 'POBP' is not in the range of US state codes (1-56)
us_state_codes = set(range(1, 57))
dfv2.loc[~dfv2['POBP'].isin(us_state_codes), 'US_or_NotUS'] = 1

# Drop the 'POBP' column
dfv2.drop('POBP', axis=1, inplace=True)

# Rename the 'US_or_NotUS' column to 'POBP'
dfv2.rename(columns={'US_or_NotUS': 'POBP'}, inplace=True)

# Now, dfv2 will have a 'US_or_NotUS' column where 0 indicates US and 1 indicates non-US.

dfv2['POBP'].unique()
dfv2.head()
dfv2

dfv2['POBP']


# Define a mapping for replacement
replacement_mapping = {1: 'couple', 2: 'single', 3: 'single', 4: 'single', 5: 'single'}

# Replace values in the 'MAR' column using the mapping
dfv2['MAR'] = dfv2['MAR'].replace(replacement_mapping)

# Now, the 'MAR' column will have 'couple' for 1 and 'single' for 2 to 5.

# Define a mapping for the final replacement
final_mapping = {'single': 1, 'couple': 0}

# Replace values in the 'MAR' column using the final mapping
dfv2['MAR'] = dfv2['MAR'].replace(final_mapping)

# Now, the 'MAR' column will have 1 for 'single' and 0 for 'couple'.

dfv2.columns


# {' Unmarried':0,' Husband or wife':1,' Not-in-family':2,' Own-child':3,' Other-relative':4, ' Other-nonrelative':5, ' Reference person':6}

# Define the updated mapping
relationship_mapping = {
    0.0: 6,       # Mapping 0.0 to 6
    13.0: 0,      # Mapping 13.0 to 0
    1.0: 1,       # Mapping 1.0 to 1
    11.0: 2,      # Mapping 11.0 to 2
    12.0: 2,      # Mapping 12.0 to 2
    14.0: 2,      # Mapping 14.0 to 2
    2.0: 3,       # Mapping 2.0 to 3
    3.0: 4,       # Mapping 3.0 to 4
    4.0: 4,       # Mapping 4.0 to 4
    5.0: 4,       # Mapping 5.0 to 4
    6.0: 4,       # Mapping 6.0 to 4
    7.0: 4,       # Mapping 7.0 to 4
    8.0: 4,       # Mapping 8.0 to 4
    9.0: 4,       # Mapping 9.0 to 4
    10.0: 4,      # Mapping 10.0 to 4
    15.0: 5,      # Mapping 15.0 to 5
    16.0: 5,      # Mapping 16.0 to 5
    17.0: 5       # Mapping 17.0 to 5
}

# Map the values in the RELP column using the updated mapping
dfv2['RELP'] = dfv2['RELP'].map(relationship_mapping)

# Check the updated DataFrame
print(dfv2['RELP'])

# Perform one-hot encoding on the 'RELP' column
dfv2_encoded = pd.get_dummies(dfv2, columns=['RELP'], prefix='RELP')

# Check the updated DataFrame with one-hot encoding
print(dfv2_encoded)



# Define the mapping for COW column
cow_mapping = {
    1: 1, 2: 1,    # Mapping 1 and 2 to 1
    3: 2, 4: 2, 5: 2,  # Mapping 3, 4, and 5 to 2
    6: 3, 7: 3,    # Mapping 6 and 7 to 3
    8: 4, 9: 4     # Mapping 8 and 9 to 4
}

# Map the values in the COW column using the mapping
dfv2['COW'] = dfv2['COW'].map(cow_mapping)

# Check the updated DataFrame
print(dfv2['COW'])


# Perform one-hot encoding on the 'COW' column
dfv2_encoded = pd.get_dummies(dfv2, columns=['COW'], prefix='COW')

# Check the updated DataFrame with one-hot encoding
print(dfv2_encoded)




# Train a machine learning algorithm on the data
# Multi-layer Perceptron classifier with default parameters

FEMALE_LABEL, MALE_LABEL = (2, 1)
HIGH_SALARY_LABEL, LOW_SALARY_LABEL = (0, 1)
dfv2['PINCP_binary'] = dfv2['PINCP_binary'].map({'>50k':HIGH_SALARY_LABEL,'<=50k':LOW_SALARY_LABEL})

dfv2['SEX'].unique()
dfv2['PINCP_binary'].unique()
dfv2




from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Assuming dfv2 contains your preprocessed dataset
X = dfv2.drop(['PINCP_binary'], axis=1)
y = dfv2['PINCP_binary']

# Define the desired sample size
sample_size = 50000

# Perform stratified sampling
x_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)
sampled_df = pd.concat([x_sample, y_sample], axis=1)

sampled_df.head()

sampled_df

def get_naive_dataset(dataset):
    data_shuffled = dataset.sample(frac=1).reset_index(drop=True)
    X = data_shuffled.drop(['PINCP_binary'], axis=1)
    y = data_shuffled['PINCP_binary']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return (x_train, y_train), (x_test, y_test)

MLP_MAX_ITER=10000

datav2 = sampled_df.copy()

(x_train, y_train), (x_test, y_test) = get_naive_dataset(datav2)
model = MLPClassifier(max_iter=MLP_MAX_ITER)
model.fit(x_train,y_train)
prediction = model.predict(x_test)

x_test.head()

test_df = x_test.copy()
test_df['PINCP_binary'] = y_test
test_df['pred'] = pd.Series(prediction, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['PINCP_binary'])
test_df.head()

test_df.columns

"Accuracy: ", test_df.accurate.mean()
    





# Understanding gender bias in machine learning predictions

def evaluate_gender_performance(results_df, print_stats=False):
    """
    TODO: 
        - method documentation
        - DRYing
    """
    def printline():
        _print('-------------------------------------------------')
        
    def _print(*args, **kwargs):
        if print_stats:
            print (args, kwargs)
    
    summaries = {}
    overall_accuracy = results_df.accurate.mean()
    summaries['accuracy_overall'] = overall_accuracy
    printline()
    _print("\n1.Overall accuracy: ", overall_accuracy)
    
    printline()
    
    # Accuracy accross gender
    _print("\n2.Accuracy accross gender \n ")
    printline
    for gender in [(FEMALE_LABEL, "Female"), (MALE_LABEL, "Male")]:
        rows = results_df[results_df.SEX==gender[0]]
        accuracy_for_gender = rows.accurate.mean();
        summaries['accuracy_'+gender[1]] = accuracy_for_gender
        _print("P(((high, HIGH) or (low, LOW)) |", gender[1], "): ", accuracy_for_gender)
    printline()
    
    _print("\n3.Positive Rates: \n")
    # High income rate given gender
    for gender in [(FEMALE_LABEL, "Female"), (MALE_LABEL, "Male")]:
        rows = results_df[results_df.SEX==gender[0]]
        positive_rate_for_gender = (rows['pred'] ==HIGH_SALARY_LABEL).mean()
        summaries['positive_rate_'+gender[1]] = positive_rate_for_gender
        _print("P(high|", gender[1], "): ", positive_rate_for_gender)
    printline()
    
    _print("\n4. Negative Rates: \n")
    
    # Low income rate given gender
    # High income rate given gender
    for gender in [(FEMALE_LABEL, "Female"), (MALE_LABEL, "Male")]:
        rows = results_df[results_df.SEX==gender[0]]
        positive_rate_for_gender = (rows['pred'] ==LOW_SALARY_LABEL).mean()
        summaries['negative_rate_'+gender[1]] = positive_rate_for_gender
        _print("P(low|", gender[1], "): ", positive_rate_for_gender)
    printline()
    
    _print("\n4. True positive and True negative rates")
    
    printline()
    
    for index, gender in enumerate([(FEMALE_LABEL, "Female"), (MALE_LABEL, "Male")]):
        
        _print("\n4."+("i")*(1+index), " True positive and negative rates on SEX="+gender[1], "\n")
        rows = results_df[results_df.SEX==gender[0]]
        
        high_income = rows[rows.PINCP_binary== HIGH_SALARY_LABEL]
        low_income=rows[rows.PINCP_binary == LOW_SALARY_LABEL]
        if high_income.shape[0] > 0:
            assert high_income.PINCP_binary.mean() == HIGH_SALARY_LABEL, "high_mean: " + str(high_income.PINCP_binary.mean())
        if low_income.shape[0] > 0:
            assert low_income.PINCP_binary.mean() == LOW_SALARY_LABEL, "low_mean: " + str(low_income.PINCP_binary.mean())
        
        high_pred = rows[rows.pred == HIGH_SALARY_LABEL]
        low_pred = rows[rows.pred == LOW_SALARY_LABEL]
        if high_pred.shape[0] > 0:
            assert high_pred.pred.mean() == HIGH_SALARY_LABEL, "high_pred_mean: " + str(high_pred.pred.mean())
        if low_pred.shape[0] > 0:
            assert low_pred.pred.mean() == LOW_SALARY_LABEL, "low_pred_mean: " + str(low_pred.pred.mean())
        
        printline() 
        true_positive_rate = high_income.accurate.mean()
        true_negative_rate = low_income.accurate.mean()
        summaries['true_positive_rate_'+gender[1]] = true_positive_rate
        summaries['true_negative_rate_'+gender[1]] = true_negative_rate
        
        _print(str.format("P((high, HIGH)| HIGH,{0})", gender[1]), ": ",true_positive_rate)
        _print(str.format("P((low, LOW)| LOW,{0})", gender[1]), ":",true_negative_rate)
        
        printline()
        #true_positive_rate_on_positive_predictions = high_pred.accurate.mean()
        #true_negative_rate_on_negative_predictions = low_pred.accurate.mean()
        #summaries['true_positive_rate_on_positive_predictions_'+gender[1]] = true_positive_rate_on_positive_predictions
        #summaries['true_negative_rate_on_negative_predictions_'+gender[1]] = true_negative_rate_on_negative_predictions
        #_print(str.format("P((high,HIGH) | high, {0})", gender[1]), ": ",true_positive_rate_on_positive_predictions)
        #_print(str.format("P((low,LOW) |low,{0})", gender[1]), ":", true_negative_rate_on_negative_predictions)
        
        
    return summaries

def plot_performance_per_group(accuracy_results, title, fignum=1, rotation='horizontal', labels=["Male", "Female"]):
    
    """
    Plot results for 2 groups stacked together
    """
    assert isinstance(accuracy_results, list), "Accuracy results must be a list"
    
    
    indices = [0]
    colors = ['red', 'blue']
    fig, ax = plt.subplots()
    
    for index in indices:
        ax.scatter(index, accuracy_results[0][index], c=colors[0], label=labels[0] if labels and index ==0 else None)
        ax.scatter(index, accuracy_results[1][index], c=colors[1], label=labels[1] if labels and index ==0 else None)
        
    if labels:
        ax.legend()
        
    #plt.xticks(indices, approaches, rotation=rotation)
    plt.title(title)
    
    plt.show()
    
    
def plot_comparisons_groups(approaches, accuracy_results, title, fignum=1, rotation='horizontal', labels=["Male", "Female"]):    
    """
    Plot results for 2 groups stacked together
    """
    assert isinstance(accuracy_results, list), "Accuracy results must be a list"
    
    
    indices = list(range(len(approaches)))
    colors = ['red', 'blue']
    fig, ax = plt.subplots()
    
    for index in indices:
        ax.scatter(index, accuracy_results[0][index], c=colors[0], label=labels[0] if labels and index ==0 else None)
        ax.scatter(index, accuracy_results[1][index], c=colors[1], label=labels[1] if labels and index ==0 else None)
        
    if labels:
        ax.legend()
        
    plt.xticks(indices, approaches, rotation=rotation)
    plt.title(title)
    
    plt.show()
    
def plot_model_gender_metrics(_feature, _summaries, _modelNames, _title, rotation='vertical'):
    gender_metrics = [[summary[_feature+'_Male'] for summary in _summaries], 
                         [summary[_feature+'_Female'] for summary in _summaries]
                        ]
    plot_comparisons_groups(_modelNames,gender_metrics, _title, rotation=rotation)

def model_summary(model_name, title, summary):
    summaries = []
    model_names = []
    
    for key in ["accuracy", "positive_rate", "negative_rate", "true_positive_rate", "true_negative_rate"]:
        new_summary = {"accuracy_Male": summary[key+"_Male"], "accuracy_Female": summary[key+"_Female"]}
        summaries.append(new_summary)
        model_names.append(key)   
    plot_model_gender_metrics("accuracy", summaries, model_names, model_name)
    #plot_model_gender_metrics(key, [summary], [model_name], "Model="+model_name+", Metric="+key, rotation="horizontal")



original_approach = evaluate_gender_performance(test_df)
model_summary("MLP_no_debias", "", original_approach)


default_training_sizes = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, lrxtrain, lrytrain, train_sizes, title=None):
    _train_sizes = []
    for size in train_sizes:
        if size <= lrxtrain.shape[0]*.65:
            _train_sizes.append(size)
        else:
            break
    train_sizes, train_scores, validation_scores = learning_curve(
                                                 estimator, lrxtrain, lrytrain, train_sizes = _train_sizes, scoring = 'neg_log_loss')
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

    plt.ylabel('NLL', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()


#plot_learning_curve(MLPClassifier(), X, y, default_training_sizes, "Learning curve, Naive approach")


# Mitigation through unawareness

def get_unawareness_dataset(dataset):
    (x_train, y_train), (x_test, y_test) = get_naive_dataset(dataset)
    testdata = x_test.copy()
    assert "SEX" in list(testdata.columns), ("columns: ", list(testdata.columns))
    
    x_train, x_test = [v.drop(['SEX'], axis=1) for v in (x_train, x_test)]
    return (x_train, y_train), (x_test, y_test), testdata


predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
(x_train, ytrain), (x_test, y_test), testdata = get_unawareness_dataset(datav2)
predictor.fit(x_train, y_train)


def evaluate_predictor_performance(predictions, x_test, y_test):
    """
    Returns summary statistics for the predictor's performance
    
    Input:
        - predictions: model's predictions on x_test
        - x_test: test input
        - y_test: test labels
        
    Requires:
        - predictor must have been fitted on x_train and y_train from the same dataset
    
    Check method evaluate_gender_performance for more on the produced summary statistics
    """
    testdata = x_test.copy()
    testdata['PINCP_binary'] = y_test
    testdata['pred'] = pd.Series(predictions, index=x_test.index)
    testdata['accurate'] = (testdata['pred'] == testdata['PINCP_binary'])
    return evaluate_gender_performance(testdata)


predictions = predictor.predict(x_test)
approach_1 = evaluate_predictor_performance(predictions, testdata, y_test)
model_summary("MLP, gender unaware", "", approach_1)


# TODO: Change X, Y
#plot_learning_curve(MLPClassifier(), X, y, default_training_sizes, 'Learning curves, Unawareness')

# Mitigation through dataset balancing

def get_gender_balanced_dataset(dataset, test_size=0.25):
    """
    Returns (x_train, y_train), (x_test, y_test) with equal number of samples for each gender
    """
    males, females = dataset[dataset.SEX == MALE_LABEL], dataset[dataset.SEX==FEMALE_LABEL]
    sampled_males = males.sample(n=int(min(females.shape[0], males.shape[0]))).reset_index(drop=True)
    combined = pd.concat([sampled_males, females]).sample(frac=1).reset_index(drop=True)
    Xvals=combined.drop(["PINCP_binary"], axis=1)
    Yvals = combined["PINCP_binary"]
    x_train, x_test, y_train, y_test = train_test_split(Xvals, Yvals, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)


datav3 = datav2.copy()
datav3.head()


(x_train, y_train), (x_test, y_test) = get_gender_balanced_dataset(datav3)
x_train.shape, x_test.shape


predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train, y_train)
approach_2 = evaluate_predictor_performance(predictor.predict(x_test), x_test, y_test)
model_summary("MLP, equal_datapoints", "", approach_2)


# TODO: Set to correct values
#plot_learning_curve(MLPClassifier(), Xvals, Yvals, default_training_sizes, 'Learning curve, 3.2')


predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train.drop(['SEX'], axis=1), y_train)
approach_2_blind = evaluate_predictor_performance(predictor.predict(x_test.drop(['SEX'], axis=1)), x_test, y_test)
model_summary("MLP, equal_datapoints_blind", "", approach_2_blind)



# Equal number of datapoints per demographic in each category

# TODO: Implement random sampling
def get_gender_category_balanced_dataset(dataset, test_size=0.25):
    """
    Equal number of datapoints per category. Limited by the smallest number of points
    """
    # Old distribution categories
    males = dataset[(dataset.SEX==MALE_LABEL)]
    females = dataset[(dataset.SEX==FEMALE_LABEL)]
    male_high = males[(males.PINCP_binary == HIGH_SALARY_LABEL)]
    male_low = males[(males.PINCP_binary == LOW_SALARY_LABEL)]
    female_high = females[(females.PINCP_binary == HIGH_SALARY_LABEL)]
    female_low = females[(females.PINCP_binary == LOW_SALARY_LABEL)]
    
    # Smallest is the bottleneck
    smallest = min((x.shape[0] for x in [male_high, male_low, female_high, female_low]))
    
    # New distribution categories
    _male_high = male_high.sample(n=smallest).reset_index(drop=True)
    _male_low = male_low.sample(n=smallest).reset_index(drop=True)
    _female_high = female_high.sample(n=smallest).reset_index(drop=True)
    _female_low = female_low.sample(n=smallest).reset_index(drop=True)
    _combined = pd.concat([_male_high, _male_low, _female_high, _female_low]).sample(frac=1).reset_index(drop=True)
    
    Xvals=_combined.drop(["PINCP_binary"], axis=1)
    Yvals = _combined["PINCP_binary"]
    x_train, x_test, y_train, y_test = train_test_split(Xvals, Yvals, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = get_gender_category_balanced_dataset(datav3)

predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train, y_train)
predictions = predictor.predict(x_test)


approach_3 = evaluate_predictor_performance(predictions, x_test, y_test)
model_summary("MLP, equal_datapoints_per_category", "", approach_3)



# Equal datapoints, gender_unaware

(x_train, y_train), (x_test, y_test) = get_gender_category_balanced_dataset(datav3)

predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train.drop(['SEX'], axis=1), y_train)
predictions = predictor.predict(x_test.drop(['SEX'], axis=1))
approach_3_blind = evaluate_predictor_performance(predictions, x_test, y_test)
model_summary("MLP, equal_datapoints_per_category_blind", "", approach_3_blind) 



# Equal ratios instead of equal number of datapoints

def get_gender_category_ratio_balanced_dataset(dataset):
    
    """
    Ratio of (male_high, male_row) = Ratio of (female_high, female_low), maximize number of real datapoints
    """
    
    # Old distribution categories
    males = dataset[(dataset.SEX==MALE_LABEL)]
    females = dataset[(dataset.SEX==FEMALE_LABEL)]
    assert males.shape[0] > 0 and females.shape[0] > 0, "Empty males or females"
    
    male_high = males[(males.PINCP_binary == HIGH_SALARY_LABEL)]
    male_low = males[(males.PINCP_binary == LOW_SALARY_LABEL)]
    
    assert male_high.shape[0] > 0 and male_low.shape[0] > 0, " empty male high or low"
    
    female_high = females[(females.PINCP_binary == HIGH_SALARY_LABEL)]
    female_low = females[(females.PINCP_binary == LOW_SALARY_LABEL)]
    
    assert female_high.shape[0] > 0 and female_low.shape[0] > 0, "empty female high or low"
    
    
    print("shapes mh, ml, fh, fl: ", [x.shape[0] for x in [male_high, male_low, female_high, female_low]])
    
    ratio = float(male_high.shape[0]) / float(male_low.shape[0])
    assert ratio > 0, " ratio must be greater than 0"
    
    print ("Ratio is ", ratio)
    n_female_high = female_high.shape[0]
    n_female_low = int(n_female_high / ratio)

    _male_low = male_low.copy()
    _male_high = male_high.copy()
    _female_high = female_high.copy()
    _female_low = female_low.sample(n=n_female_low).reset_index(drop=True)
    _combined = pd.concat([_male_high, _male_low, _female_high, _female_low]).sample(frac=1).reset_index(drop=True)
    
    Xvals=_combined.drop(["PINCP_binary"], axis=1)
    Yvals = _combined["PINCP_binary"]
    x_train, x_test, y_train, y_test = train_test_split(Xvals, Yvals, test_size=0.25)
    
    return (x_train, y_train), (x_test, y_test)


datav3.shape


(x_train, y_train), (x_test, y_test) = get_gender_category_ratio_balanced_dataset(datav3)
predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train, y_train)
predictions = predictor.predict(x_test)


approach_4 = evaluate_predictor_performance(predictions, x_test, y_test)
model_summary("MLP equal_ratios", "", approach_4)


predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train.drop(['SEX'], axis=1), y_train)
predictions = predictor.predict(x_test.drop(['SEX'], axis=1))
approach_4_blind = evaluate_predictor_performance(predictions, x_test, y_test)
model_summary("MLP, equal_ratios_blind", "", approach_4_blind)



# Bias mitigation through data augmentation

def with_gender_counterfacts(df):
    df_out = df.copy()
    df_out['SEX'] = df_out['SEX'].apply(lambda value: 1-value)
    result = pd.concat([df.copy(), df_out])
    return result


ctf_gender_augmented = with_gender_counterfacts(datav2)
(x_train, y_train), (x_test, y_test) = get_naive_dataset(ctf_gender_augmented)

predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train, y_train)
ctf_1 = evaluate_predictor_performance(predictor.predict(x_test), x_test, y_test)
model_summary("counterfactual_augmentation", "", ctf_1)


predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train.drop(['SEX'], axis=1), y_train)
              

ctf_blind = evaluate_predictor_performance(predictor.predict(x_test.drop(['SEX'], axis=1)), x_test, y_test)
model_summary("counterfactual_augmentation_blind", "", ctf_blind)


# Comparing bias mitigation approaches

def plot_comparisons(approach_names, accuracy_results, title, fignum=1, rotation='horizontal'):
    """
    Args:
        - summary: Dictionary describing the approach's gender performance
        - approach_name: The name of the technique, to be displayed
    """
    assert isinstance(accuracy_results, list) and not isinstance(accuracy_results[0], list), accuracy_results
    
    
    indices = list(range(len(approach_names)))
    colors = cm.rainbow(np.linspace(0, 1, len(indices)))
    plt.figure(fignum)
    
    for index in indices:
        plt.scatter(index, accuracy_results[index], color=colors[index])
        
    plt.xticks(indices, approach_names, rotation=rotation)
    
    plt.title(title)
    plt.show()
    
def plot_comparisons_groups(approaches, accuracy_results, title, fignum=1, rotation='horizontal', labels=["Male", "Female"]):    
    """
    Plot results for 2 groups stacked together
    """
    assert isinstance(accuracy_results, list), "Accuracy results must be a list"
    
    
    indices = list(range(len(approaches)))
    colors = ['red', 'blue']
    fig, ax = plt.subplots()
    
    for index in indices:
        ax.scatter(index, accuracy_results[0][index], c=colors[0], label=labels[0] if labels and index ==0 else None)
        ax.scatter(index, accuracy_results[1][index], c=colors[1], label=labels[1] if labels and index ==0 else None)
        
    if labels:
        ax.legend()
        
    plt.xticks(indices, approaches, rotation=rotation)
    plt.title(title)
    
    plt.show()


approaches = ['no_debias', 'gender_unaware', 'equal_|data|_per_gender', 'if_gender_blind', 'equal_data_per_(gender, category)', 'if_gender_blind' 'equal_data_ratio_per_gender', 'if_blind', 'ctf', 'ctf_blind']
summaries = [original_approach, approach_1, approach_2, approach_2_blind, approach_3, approach_3_blind, approach_4, approach_4_blind, ctf_1, ctf_blind]


# Comparing overall accuracies

accuracy_results = [summary['accuracy_overall'] for summary in summaries]
plot_comparisons(approaches, accuracy_results, 'Comparisons of overall accuracy', rotation='vertical')


def plot_model_gender_metrics(_feature, _summaries, _modelNames, _title, rotation='vertical'):
    gender_metrics = [[summary[_feature+'_Male'] for summary in _summaries], 
                         [summary[_feature+'_Female'] for summary in _summaries]
                        ]
    plot_comparisons_groups(_modelNames,gender_metrics, _title, rotation=rotation)


# Comparing overall accuracy accross gender

plot_model_gender_metrics('accuracy', summaries, approaches, "Accuracy on Male vs Accuracy on Female")


# Positive and negative rates accross gender


plot_model_gender_metrics('positive_rate', summaries, approaches, "Positive rates accross gender")
plot_model_gender_metrics('negative_rate', summaries, approaches, "Negative rates accross gender")


# True positive and True negative rates accross gender

plot_model_gender_metrics('true_positive_rate', summaries, approaches, "True positive rates accross gender")
plot_model_gender_metrics('true_negative_rate', summaries, approaches, "True negative rates accross gender")


#True positive rate on positive predictions, and true negative rate on negative predictions

#plot_model_gender_metrics('true_positive_rate_on_positive_predictions', summaries, approaches, "True positive rates on positive predictions")
#plot_model_gender_metrics('true_negative_rate_on_negative_predictions', summaries, approaches, "True negative rates on negative predictions")

from sklearn import svm, metrics
clf = MLPClassifier(max_iter=MLP_MAX_ITER)
clf.fit(x_train, y_train)
# obtain the learned decision function and evaluate it on the held-out data



def plot_categs(df, category, fignum=1, title="Histogram of number of datapoints"):
    plt.figure(fignum)
    uniques= list(sorted(df[category].unique()))
    counts = [df[df[category] == value].shape[0] for value in uniques]
    size = len(uniques)
    xcoords = list(range(1, size+1))
    plt.bar(xcoords, counts)
    plt.xticks(xcoords, uniques, rotation='vertical' if size >= 5 else 'horizontal')
    plt.title((title if title is not None else ''))
    plt.tight_layout()



def plot_roc_curve(trained_predictor, X_test_list=None, Y_test_list=None, label_list = None, fignum=None):
    """
    Trained predictor must have .decision_function attribute
    """
    if fignum is not None:
        plt.figure(fignum)
    for index in range(len(X_test_list)):
        X_test = X_test_list[index]
        Y_test = Y_test_list[index]
        assert X_test is not None and Y_test is not None, "X_test and Y_test cannot be None"
        y_pred_scores = trained_predictor.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred_scores) # obtain false positive and true positive rates
        area_under_curve = metrics.auc(fpr, tpr)
        label = "for gender = "+  label_list[index] if label_list is not None else ''
        #plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f) %s' % (area_under_curve, label)) # plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f) %s' % (area_under_curve, label))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")



plot_roc_curve(clf, X_test_list = [x_test], Y_test_list = [y_test])


predictor = MLPClassifier(max_iter=MLP_MAX_ITER)
predictor.fit(x_train, y_train)




def plot_gender_roc_curves(trained_predictor, X_test, Y_test):
    """
    Plots the ROC curve for each gender demographic
    """
    combined = pd.concat([X_test, Y_test], axis=1)
    x_test_list, y_test_list, gender_labels = [], [], []
    for gender, gender_label in (("FEMALE", FEMALE_LABEL), ("MALE", MALE_LABEL)):
        with_gender = combined[combined['SEX'] == gender_label]
        x_test = with_gender.drop(['PINCP_binary'], axis=1)
        y_test = with_gender['PINCP_binary']
        x_test_list.append(x_test)
        y_test_list.append(y_test)
        gender_labels.append(gender)
    plot_roc_curve(predictor, X_test_list=x_test_list, Y_test_list=y_test_list,label_list=gender_labels)


plot_gender_roc_curves(predictor, x_test, y_test)


# Bias mitigation through fair model selection
#how model choice affects different metrics

from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score

LR_MAX_ITER=1000

(x_train, y_train), (x_test, y_test) = get_naive_dataset(datav3)

lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1, max_iter=LR_MAX_ITER) # GLM
rf = RandomForestClassifier(n_estimators=50, random_state=1) # Random Forest
gnb = GaussianNB() # GLM
mlp = MLPClassifier(max_iter=MLP_MAX_ITER)  # ANN
svc = svm.SVC() # SVM
knc = KNeighborsClassifier(n_neighbors=5)
for model in [lr, rf, gnb, mlp, svc, knc]:
    model.fit(x_train, y_train)
    

for model_name, model in [('LR', lr), ('RF', rf), ('GNB', gnb), ('MLP', mlp), ('svc', svc), ('knc', knc)]:
    print(model_name, ' accuracy: ', accuracy_score(y_test, model.predict(x_test)))


# Debiasing through multi-model architecture

# A multi-model architecture combines different machine learning models, and makes a prediction by taking into account the predictions of multiple models.

from sklearn.ensemble import VotingClassifier

def default_voting_classifier(voting='hard'):
    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1, max_iter=LR_MAX_ITER)
    rf = RandomForestClassifier(n_estimators=50, random_state=1)
    gnb = GaussianNB()
    mlp = MLPClassifier(max_iter=MLP_MAX_ITER)
    svc = svm.SVC(probability = voting != 'hard')
    knc = KNeighborsClassifier(n_neighbors=5)
    voter = VotingClassifier(estimators=[('LR', lr), ('RF', rf), ('GNB', gnb), ('MLP', mlp), ('svc', svc)], voting=voting)
    
    return voter

(x_train, y_train), (x_test, y_test) = get_naive_dataset(datav3)

hardvoter = default_voting_classifier(voting='hard')
softvoter = default_voting_classifier(voting='soft')
for model in [hardvoter, softvoter]:
    model.fit(x_train, y_train)


print('Hard voting accuracy ', accuracy_score(y_test, hardvoter.predict(x_test)))
print('Soft voting accuracy ', accuracy_score(y_test, softvoter.predict(x_test)))



# Model persistence
import pickle

class Persistence:
    """
    Implements model persistence functionality
    """
    def __init__(self):
        pass
    
    @staticmethod 
    def storeObject(_object, filename):
        pickle_out = open(filename,"wb")
        pickle.dump(_object, pickle_out)
        pickle_out.close()
    
    @staticmethod
    def loadObjects(filenames=None):
        result = []
        for filename in filenames:
            result.append(pickle.load(open(filename, 'rb')))
        return result
    
    @staticmethod
    def storeOrLoad(store=False, load=False, names=None, objects=None):
        """
        Returns file names if storing, returns objects if reading
        """
        if store or load:
            assert store != load, 'Cannot store and load'
        if store:
            for _object, name in zip(objects, names):
                Persistence.storeObject(_object, name)
            return 'Stored'
        if load:
            read = Persistence.loadObjects(filenames=names)
            return read


modelNames = ['LR', 'RF', 'GNB', 'MLP', 'SVC', 'hard_voter', 'soft_voter']
models = [lr, rf, gnb, mlp, svc, hardvoter, softvoter]


summaries = []
for model, modelname in zip(models, modelNames):
    summaries.append(evaluate_predictor_performance(model.predict(x_test), x_test, y_test))


# Comparing model performance for 1 training session
# Overall Accuracy

overall_accuracies = [summary['accuracy_overall'] for summary in summaries]
plot_comparisons(modelNames, overall_accuracies, "Overall Accuracy", rotation='vertical')

gender_accuracies = [[summary['accuracy_Male'] for summary in summaries], [summary['accuracy_Female'] for summary in summaries]]
plot_comparisons_groups(modelNames, gender_accuracies, "Gender accuracies", rotation='vertical')


# Positive and negative rates accross each demographic


plot_model_gender_metrics('positive_rate', summaries, modelNames, "Positive rates", rotation='vertical')
plot_model_gender_metrics('negative_rate', summaries, modelNames, "Negative rates", rotation='vertical')


# True positive and true negative rates across each demographic

plot_model_gender_metrics('true_positive_rate', summaries, modelNames, "True positive rates", rotation='vertical')
plot_model_gender_metrics('true_negative_rate', summaries, modelNames, "True negative rates", rotation='vertical')


# Comparing model performance over multiple training sessions

def get_model_class_summaries(model_class, dataset, training_sessions, *args, **kwargs):
    """
    Repeatedly sample from the dataset, train, test and return summary statistics
    """
    assert training_sessions >= 1, "Must train at least once"
    
    Xvals, Yvals = dataset
    summaries = []
    for session in range(training_sessions):
        x_train, x_test, y_train, y_test = train_test_split(Xvals, Yvals, test_size=.25)
        model = model_class(*args, **kwargs)
        model.fit(x_train, y_train)
        
        evaluation = evaluate_predictor_performance(model.predict(x_test), x_test, y_test)
        summaries.append(evaluation)
        
    assert len(summaries) == training_sessions
    return summaries


dataset = datav3.copy()
Xvals=dataset.drop(["salary"], axis=1)
Yvals = dataset["salary"]
some_summaries = get_model_class_summaries(MLPClassifier, (Xvals, Yvals), 2, max_iter=MLP_MAX_ITER)
some_summaries[:2]


single_model_name_classes_args_kwargs = [
    ['LR', LogisticRegression,     [], {'solver': 'lbfgs', 'multi_class': 'multinomial', 'random_state':1, 'max_iter':LR_MAX_ITER}], 
    ['RF', RandomForestClassifier, [], {'n_estimators':50, 'random_state':1}], 
    ['GNB', GaussianNB,             [], {}],
    ['MLPC', MLPClassifier,          [], {'max_iter':MLP_MAX_ITER}], 
    ['SVC', svm.SVC,                [], {}]
]


single_model_summaries = []


runLoop = True # simply set runLoop to False to avoid iterations
if runLoop:
    for name, model_class, args, kwargs in single_model_name_classes_args_kwargs:
        model_class_summaries = get_model_class_summaries(model_class, (Xvals, Yvals), 5, *args, **kwargs)
        single_model_summaries.append((name, model_class_summaries))



store = True # Change the flag to store the summaries
if store:
    Persistence.storeOrLoad(store=True, names=['single_model_summaries'], objects = [single_model_summaries])


voting_model_name_classes_args_kwargs = [
    ['hard_voting', default_voting_classifier, [], {'voting':'hard'}],
    ['soft_voting', default_voting_classifier, [], {'voting':'soft'}]
]



# Aggregate results for multi-model architectures

voting_model_summaries = []


runLoop = True
if runLoop:
    for name, model_class, args, kwargs in voting_model_name_classes_args_kwargs:
        model_class_summaries = get_model_class_summaries(model_class, (Xvals, Yvals), 5, *args, **kwargs)
        voting_model_summaries.append((name, model_class_summaries))


assert len(voting_model_summaries) > 0
Persistence.storeOrLoad(store=True, names=['voting_model_summaries'], objects=[voting_model_summaries])


def extract_treatment_differences(summaries):
    """
    Extract treatment difference(male-female) from a performance summary
    """
    differences = []
    for summary_dict in summaries:
        gender_attrs = set()
        for gender_key in summary_dict:
            if '_Male' in gender_key:
                gender_attrs.add(gender_key[:gender_key.rindex('_')])
        value_dict = {}
        for gender_attr in gender_attrs:
            value_dict[gender_attr] = summary_dict[gender_attr+"_Male"] - summary_dict[gender_attr+ "_Female"]
        differences.append(value_dict)
    return differences


all_model_summaries = single_model_summaries + voting_model_summaries


store = True # set to True to store all model summaries
if store:
    Persistence.storeOrLoad(store=True, names=['all_model_summaries'], objects=[all_model_summaries])


load = True # Set to true to load stored models
if load:
    all_model_summaries = Persistence.storeOrLoad(load=load, names=['all_model_summaries'])[0]


all_model_summaries[-2:]


all_model_differences = []



for model_name, model_summaries in all_model_summaries:
    differences = extract_treatment_differences(model_summaries)
    all_model_differences.append((model_name, differences))
    
    
all_model_differences[:1]



# Plotting model performance differences

def get_model_values_for_feature(feature, nsp, abs_val=False):
    """
    Inputs: 
        nsp = name summary pairs
    """
    model_names = [model_summary[0] for model_summary in nsp]
    model_summary_lists = [model_summary[1] for model_summary in nsp] # Each element is a list of dicts
    model_y_values = []
    for model_summary_list in model_summary_lists:
        values = [abs(model_summary[feature]) if abs_val else model_summary[feature] for model_summary in model_summary_list]
        model_y_values.append(values)
    return model_names, model_y_values

def plot_model_values_for_feature(model_names, model_y_values, title, rotation='vertical'):

    indices = list(range(len(model_names)))
    
    colors = cm.rainbow(np.linspace(0, 1, len(indices)))
    
    fig, ax = plt.subplots()
    
    for index in indices:
        for y_value in model_y_values[index]:
            ax.scatter(index, abs(y_value), color=colors[index], label=model_names[index])
    plt.xticks(indices, model_names, rotation=rotation)
    #plt.yticks([0])
    plt.title(title)
    
    plt.show()
    
def plot_feature_differences(feature_name, model_differences, title):
    model_names, model_y_values = get_model_values_for_feature(feature_name, all_model_differences)
    plot_model_values_for_feature(model_names, model_y_values, title)    
    


# Accuracy difference comparison

plot_feature_differences('accuracy', all_model_differences, 'Accuracy Difference Comparison')


# Positive and negative rate difference comparison

pl = [('positive_rate', 'Positive Rate'), ('negative_rate', 'Negative Rate')]
for feature, title in pl:
    plot_feature_differences(feature, all_model_differences, title+ ' Difference Comparison')


# True positive and true negative rate difference comparison


pl = [('true_positive_rate', 'True Positive Rate'), ('true_negative_rate', 'True Negative Rate')]
for feature, title in pl:
    plot_feature_differences(feature, all_model_differences, title+ ' Difference Comparison')
    
    
    
    
    
    
