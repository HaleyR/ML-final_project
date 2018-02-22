#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
### 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
###'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
### 'restricted_stock', 'director_fees'

### 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
### 'from_this_person_to_poi', 'shared_receipt_with_poi'

features_list = ['poi','total_stock_value',
                 'salary']
 # You will need to use more features
clean_feature_list=['salary', 'exercised_stock_options', 'total_stock_value']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_cutoff=0.5 #percent of relavent features that are missing before removing the datapoint    
delete_list=[]
    
### Task 3: Create new feature(s)
for person, value in data_dict.items():
    
    missing_vals=0
    for i in clean_feature_list:
        if value[i]=="NaN":
            value[i]=0
            missing_vals+=1
            print(i)
  
    value['salary_over_stock_options']=(value['salary'])/(
            value['exercised_stock_options']+0.01)    
    
    if float(missing_vals)/len(features_list)>data_cutoff:
        delete_list.append(person) #TODO need to figure out if deleting this is good or not
    
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.svm import SVC
cValue=1000
kernelType='rbf'

clf = SVC(C=cValue, kernel=kernelType)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

num_train_poi=sum(labels_train)
num_test_poi=sum(labels_test)
num_train_Npoi=len(labels_train)-num_train_poi
num_test_Npoi=len(labels_test)-num_test_poi

print("\nIn the training set there are {} POI and {} Non POI".format(
        num_train_poi, num_train_Npoi))

print("In the testing set there are {} POI and {} Non POI\n".format(
        num_test_poi, num_test_Npoi))
    
clf.fit(features_train,labels_train)    
pred=clf.predict(features_test)    

    
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

pre=precision_score(labels_test,pred)
print("The precision of the ML is {:.3f}".format(pre))

rec=recall_score(labels_test,pred)
print("The recall of the ML is {:.3f}".format(rec))

acc=accuracy_score(labels_test,pred)
print("The accuracy of the ML is {:.2f}%".format(acc*100))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

from tester import test_classifier
test_classifier(clf, my_dataset, features_list, folds = 1000)

from test_plotter import testPlotter
i=0
viewFeature=[]
while i<len(features):
    viewFeature.append([features[i][0], features[i][1]])
    i=i+1
testPlotter(features_list[1:3], viewFeature, labels)