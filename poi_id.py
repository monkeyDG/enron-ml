#!/usr/bin/python -W ignore::DeprecationWarning

import sys
import pickle

import numpy
sys.path.append("../tools/")
from tester import dump_classifier_and_data
import pandas as pd
import sys
import pickle
import csv
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
#from poi_data import *
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score, precision_score, recall_score


#%%

#% ## Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

############# Task 1: Select what features you'll use.############

target_label = 'poi'

email_features_list = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
    
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'salary',
    'total_payments'
]

stock_features_list = [
    'total_stock_value',
    'restricted_stock',
    'restricted_stock_deferred',
    'exercised_stock_options',
] # we split this list out in order to make sure that total payments can be calculated separately from stock value

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
### 1.1.0 Explore csv file 
def make_csv(data_dict):
    """ generates a csv file from a data set"""
    fieldnames = ['name'] + data_dict.itervalues().next().keys()
    with open('data.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data_dict:
            person = data_dict[record]
            person['name'] = record
            assert set(person.keys()) == set(fieldnames)
            writer.writerow(person)

### 1.1.1 Dataset Exploration
print('# Exploratory Data Analysis #')
data_dict.keys()
numdata = len(data_dict.keys())
print(f"Total number of data points: {numdata}")
num_poi = 0
for name in data_dict.keys():
    if data_dict[name]['poi'] == True:
        num_poi += 1
print('Number of Persons of Interest: %d' % num_poi)
print('Number of people without Person of Interest label: %d' % (len(data_dict.keys()) - num_poi))

df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=float)
df.head()
df.tail()

# creating a new dataframe with the sum of null values for each column in the dataset
df_null = df.isnull().sum()
df_null

# transforming the data boolean to numeric.
df['poi'] = df['poi'].map({True: 1,False: 0})

# replacing all NAN values to zero.
df = df.replace(to_replace=np.nan, value=0)

# Checking if we still have null values.
df.isnull().sum()
df.tail(n=1)

### Initial Data Visualization
print('# Initial Data Visualization #')

sns.set(font_scale=1.25)

fig, ax = plt.subplots()
fig.set_size_inches(9, 6)

sns.set(style="darkgrid")
ax = sns.countplot(x="poi", data=df).set_title("Numbers of NON-POI's vs POI")
plt.ylabel("Qtd.")

# feature correlation for financial data
plt.figure(figsize=(12,10))
cor = df[financial_features_list].corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)

# salary to bonus scatterplots
ax1 = sns.scatterplot(x=df[df['poi'] == True].salary, y=df[df['poi'] == True].bonus, color = 'r')
ax2 = sns.scatterplot(x=df[df['poi'] == False].salary, y=df[df['poi'] == False].bonus, color = 'b')

plt1 = ax1.get_figure()
plt2 = ax2.get_figure()

###1.1.2 Feature Exploration
all_features = data_dict['ALLEN PHILLIP K'].keys()
print('Each person has %d features available' %  len(all_features))
### Evaluate dataset for completeness
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in all_features:
        if data_dict[person][feature] == 'NaN':
            missing_values[feature] += 1
            if feature in financial_features_list or feature in stock_features_list:
                data_dict[person][feature] = 0 # According to the documentation for the financial data, values of NaN represent 0, so we replace numpy.NaN with 0
        else:
            records += 1

# We can use the total_payments column to check for outliers due to addition errors of financial and stock data
people_with_errors = []
for person in data_dict.keys():
    sum_financial_data = 0
    sum_stock_data = 0
    for feature in all_features:
        if feature in financial_features_list and feature != 'total_payments':
            sum_financial_data = sum_financial_data + data_dict[person][feature]
        if feature in stock_features_list and feature != 'total_stock_value':
            sum_stock_data = sum_stock_data + data_dict[person][feature]
    if sum_financial_data != data_dict[person]['total_payments'] or sum_stock_data != data_dict[person]['total_stock_value']:
        people_with_errors.append(person)
        print(f"{person} has an error in financial totals. Sum = {sum_financial_data}, expected = {data_dict[person]['total_payments']}")
        print(f"{person} has an error in total_stock_value. Sum = {sum_stock_data}, expected = {data_dict[person]['total_stock_value']}")
    
# here we can manually correct the issues for Robert Belfer and Sanjay Bhatnagarby looking at the source pdf and entering the correct values:
data_dict['BELFER ROBERT']["deferred_income"] = -102500
data_dict['BELFER ROBERT']["deferral_payments"] = 0
data_dict['BELFER ROBERT']["expenses"] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']["exercised_stock_options"] = 0
data_dict['BELFER ROBERT']["restricted_stock"] = 44093
data_dict['BELFER ROBERT']["restricted_stock_deferred"] = -44093
data_dict['BELFER ROBERT']["total_stock_value"] = 0

data_dict['BHATNAGAR SANJAY']["other"] = 0
data_dict['BHATNAGAR SANJAY']["expenses"] = 137864
data_dict['BHATNAGAR SANJAY']["director_fees"] = 0
data_dict['BHATNAGAR SANJAY']["total_payments"] = 137864
data_dict['BHATNAGAR SANJAY']["exercised_stock_options"] = 15456290
data_dict['BHATNAGAR SANJAY']["restricted_stock"] = 2604490
data_dict['BHATNAGAR SANJAY']["restricted_stock_deferred"] = -2604490
data_dict['BHATNAGAR SANJAY']["total_stock_value"] = 15456290

### Print results of completeness analysis
print('\nNumber of Missing Values for Each Feature (features with [*] are marked for deletion):')
for feature in all_features:
    if (missing_values[feature] / numdata) > 0.8:
        print(f"[*] {feature}: {missing_values[feature]} ({round(missing_values[feature] / numdata * 100, 1)}%)")
        del feature # here we delete features that have too many nulls to be useful predictors with over 80%
    else:
        print(f"{feature}: {missing_values[feature]} ({round(missing_values[feature] / numdata * 100, 1)}%)")


#%%
################# Task 2: Remove outliers #####################
def PlotOutlier(data_dict, feature_x, feature_y):
    """ Plot with flag = True in Red """
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

# 2.1 Visualise outliers
print(PlotOutlier(data_dict, 'total_payments', 'total_stock_value'))
print(PlotOutlier(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(PlotOutlier(data_dict, 'salary', 'bonus'))
#Remove outlier TOTAL line in pickle file.
data_dict.pop( 'TOTAL', 0 )

# Salary outliers
print('# Salary Analysis #')
have_quantified_salary = 0
for name in data_dict.keys():
    if data_dict[name]['salary']== 'NaN':
       have_quantified_salary += 1
print('Have Quantified Salary: %d' % have_quantified_salary)

feature1 = ["salary"]
data_dict.pop('TOTAL', 0)
salary = featureFormat(data_dict, feature1)
counts, bins = np.histogram(salary)
plt.hist(bins[:-1], bins, weights=counts)
plt.title("Salary")
plt.show()

#Find out the outliers and who is receiving such salaries
for x in salary:
    if x > 1000000 :
        print("Salary Outliers", x[0])

print('- People with Salary Outliers -')
for person in data_dict:
    if data_dict[person]["salary"] == 1072321:
        print("Salary Outliers 1:", person)
    if data_dict[person]["salary"] == 1111258:
        print("Salary Outliers 2:",person)
    if data_dict[person]["salary"] == 1060932:
        print("Salary Outliers 3:",person)

# Bonus outliers
print('# Bonus Analysis #')
from matplotlib import pyplot as plt 
import numpy as np  
feature2 = ["bonus"]
data_dict.pop('TOTAL', 0)
bonus = featureFormat(data_dict, feature2)
counts, bins = np.histogram(bonus)
plt.hist(bins[:-1], bins, weights=counts)
plt.title("Bonus")
plt.show()

#Find out the outliers and who is receiving such bonus
for x in bonus:
    if x > 5000000 :
        print("Bonus Outliers:", x[0])

print('- People with Bonus Outliers -')
for person in data_dict:
    if data_dict[person]["bonus"] == 8000000:
        print("Bonus Outliers 1:", person)
    if data_dict[person]["bonus"] == 7000000:
        print("Bonus Outliers 2:",person)
    if data_dict[person]["bonus"] == 5249999:
        print("Bonus Outliers 3:",person)
    if data_dict[person]["bonus"] == 5600000:
        print("Bonus Outliers 4:",person)   

## Lay and Skilling have been known for their fraud
## We get additional 2 persons from Bonus Outliers
#  Are these POIs as well?

##Salary- Bonus - POI Analysis
print('# Salary - Bonus - POI Analysis #')
if data_dict["LAVORATO JOHN J"]["poi"] == 1:
    print("John J Lavorato is a POI")
if data_dict["LAY KENNETH L"]["poi"] == 1:
    print("Kenneth L Lay is a POI")
if data_dict["BELDEN TIMOTHY N"]["poi"] == 1:
    print("BELDEN TIMOTHY N is a POI")
if data_dict["SKILLING JEFFREY K"]["poi"] == 1:
    print("SKILLING JEFFREY K is a POI")
if data_dict["FREVERT MARK A"]["poi"] == 1:
    print("FREVERT MARK A is a POI")
if data_dict["LOCKHART EUGENE E"]["poi"] == 1:
    print("LOCKHART EUGENE E is a POI")
if data_dict["BAXTER JOHN C"]["poi"] == 1:
    print("BAXTER JOHN C is a POI")
if data_dict["WHALLEY LAWRENCE G"]["poi"] == 1:
    print("WHALLEY LAWRENCE G is a POI")

# 2.2 Function to remove outliers
def remove_outlier(dict_object, keys):
    """ removes list of outliers keys from dict object """
    for key in keys:
        dict_object.pop(key, 0)

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E', 'BAXTER JOHN C', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G']
remove_outlier(data_dict, outliers)

# in addition to removing outliers, we can also impute NaN values with the mean of the column, for email features (remember, we set NaN = 0 for financial features)
# Fill in the NaN email data with the mean of column grouped by poi versus non_poi, since this is likely our strongest separator flag for our data

# first, get the mean for each numerical feature in the email features, for data imputation
avg_poi = {}
avg_non_poi = {}
for feature in email_features_list:
    missing_values[feature] = 0
    avg_poi[feature] = [0, 0]  # [a, b], where a is the sum of values, and b number of values
    avg_non_poi[feature] = [0, 0]
for person in data_dict.keys():
    for feature in email_features_list:
        if type(data_dict[person][feature]) == int or type(data_dict[person][feature]) == float and data_dict[person][feature] != numpy.nan:
            if data_dict[person]['poi'] == True:
                avg_poi[feature] = [avg_poi[feature][0] + data_dict[person][feature], avg_poi[feature][1] + 1]
            elif data_dict[person]['poi'] == False:
                avg_non_poi[feature] = [avg_non_poi[feature][0] + data_dict[person][feature], avg_non_poi[feature][1] + 1]

for feature in avg_poi.keys():
    avg_poi[feature] = round(avg_poi[feature][0] / avg_poi[feature][1], 3)

for feature in avg_non_poi.keys():
    avg_non_poi[feature] = round(avg_non_poi[feature][0] / avg_non_poi[feature][1], 3)

for person in data_dict.keys(): # impute values
    for feature in email_features_list:
        if data_dict[person][feature] == "NaN":
            if data_dict[person]['poi'] == True:
                data_dict[person][feature] = avg_poi[feature]
            elif data_dict[person]['poi'] == False:
                data_dict[person][feature] = avg_non_poi[feature]
#%%
################ Task 3: Create new feature(s) ####################

# 3.1 create new copies of dataset for grading
my_dataset = data_dict

## 3.2 add new features to dataset
def compute_fraction(x, y):
    """ return fraction of one variable compared to another"""    
    if x == 'NaN' or y == 'NaN' or y == 0:
        return 0.
    fraction = x / y
    return fraction

for name in my_dataset:
    data_point = my_dataset[name]

    # Create the new email features and add to the dataset
    data_point["fraction_from_poi"] = compute_fraction(data_point["from_poi_to_this_person"], data_point["to_messages"])
    data_point["fraction_to_poi"] = compute_fraction(data_point["from_this_person_to_poi"], data_point["from_messages"])
    data_point["fraction_shared_poi"] = compute_fraction(data_point["shared_receipt_with_poi"], data_point["to_messages"])

    # Create the new financial features and add to the dataset
    data_point["bonus_to_salary"] = compute_fraction(data_point["bonus"], data_point["salary"])
    data_point["bonus_to_total"] = compute_fraction(data_point["bonus"], data_point["total_payments"])

# 3.3 create feature list for grading
financial_features_list = financial_features_list + ['bonus_to_salary'] + ['bonus_to_total']
email_features_list = email_features_list + ['fraction_from_poi'] + ['fraction_to_poi'] + ["fraction_shared_poi"]
features_list = financial_features_list + stock_features_list + email_features_list

#plotting POI emails using our new fraction from poi and fraction to poi data points
features_to_plot = ['poi', 'fraction_from_poi', 'fraction_to_poi']
data_subset = featureFormat(data_dict, features_to_plot)

for person in data_subset:
    x = person[1]
    y = person[2]
    if person[0] == 0:
        plt.scatter(x, y, color='black', alpha=0.5)
    else:
        plt.scatter(x, y, color='r')
        
plt.xlabel('Fraction of emails received from POIs')
plt.ylabel('Fraction of emails sent to POIs')

#3.3.1
#PCA to reduce dimensionality:
pca = PCA(n_components=1)
df = pd.DataFrame.from_dict(data_dict, orient='index')  # creating a new pandas dataframe to run PCA
pca.fit(df[financial_features_list])
pcaComponents = pca.fit_transform(df[financial_features_list])
df['pca_financial'] = pcaComponents
data_point["pca_financial"] = pcaComponents # append new data to the dataframe

for index, name in enumerate(my_dataset):
    data_point = my_dataset[name]
    data_point["pca_financial"] = pcaComponents[index][0] # append new data to the data dictionary

features_list.append('pca_financial')
financial_features_list.append('pca_financial')

# 3.4 get K-best features
num_features = 20

# 3.5 function using SelectKBest
def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    print(scores)
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    #print ("{0} best features: {1}\n".format(k, k_best_features.keys(), scores))
    return k_best_features

best_features = get_k_best(my_dataset, features_list, num_features)

features_list = [target_label] + list(set(best_features.keys()))
features_list = [target_label] + ['bonus', 'fraction_from_poi', 'salary', 'total_stock_value', 'exercised_stock_options', 'bonus_to_total', 'deferred_income', 'bonus_to_salary', 'fraction_shared_poi', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'long_term_incentive', 'total_payments', 'restricted_stock', 'other', 'loan_advances', 'expenses', 'from_this_person_to_poi', 'fraction_to_poi']


# 3.6 print features
print ("{0} selected features: {1}\n".format(len(features_list) - 1, features_list[1:]))

# 3.7 extract the features specified in features_list
data = featureFormat(my_dataset, features_list,sort_keys = True)
# split into labels and features
labels, features = targetFeatureSplit(data)

# 3.8 scale features via min-max
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#%%
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

##########################Task 4: Using algorithm########################

###4.1  Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
g_clf = GaussianNB()

###4.2  Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

l_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1e-08, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, 
max_iter=100, multi_class='ovr', penalty='l2', random_state=42, solver='liblinear', tol=0.001, verbose=0))])

###4.3  K-means Clustering
from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

###4.4 Support Vector Machine Classifier
from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')

###4.5 Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

###4.6 Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,random_state = 42)

###4.7 Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=None, min_samples_split = 20)

###4.8 AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), learning_rate=1, n_estimators = 70)

###4.7 evaluate function
def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.2):
    print (clf)
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print ("done.\n")
    # print ("precision: {}".format(np.mean(precision)))
    # print ("recall:    {}".format(np.mean(recall)))  # we use tester.py for better testing, so we ignore the print statements here
    return np.mean(precision), np.mean(recall)


### 4.8 Evaluate all functions
evaluate_clf(g_clf, features, labels)
evaluate_clf(l_clf, features, labels)
evaluate_clf(k_clf, features, labels)
evaluate_clf(s_clf, features, labels)
evaluate_clf(rf_clf, features, labels)
evaluate_clf(gb_clf, features, labels)
evaluate_clf(dt_clf, features, labels)


#%%
### Task 5: Tune your classifier to achieve better than .42 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# dump your classifier, dataset and features_list so
# anyone can run/check your results
pickle.dump(dt_clf, open("../final_project/my_classifier.pkl", "wb"))
pickle.dump(my_dataset, open("../final_project/my_dataset.pkl", "wb"))
pickle.dump(features_list, open("../final_project/my_feature_list.pkl", "wb"))

#%%
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(dt_clf, my_dataset, features_list)
