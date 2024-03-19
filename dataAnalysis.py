import numpy as np
import pandas as pd
import os
import csv

def realpath_folder():
    curfile = os.path.realpath(__file__)
    index = curfile.rfind("\\")
    curfolder = curfile[0:index]
    return curfolder


train = pd.read_csv(realpath_folder() + '\\train.csv')

print("number of columns: ", train.shape[1])
print("number of rows: ", train.shape[0], '\n')


print("Mostly void deleted columns")
for col in train.columns:
    percentage_of_lacking_data = train[col].isna().sum()/train.shape[0]
    if (percentage_of_lacking_data > 0.1):
        print("\t", col, ":", percentage_of_lacking_data)
        train.drop(columns = [col], axis = 1, inplace = True)

print('\n')
print('Mostly homogenous categorical deleted columns, that should be deleted')
for col in train.columns:
    if len(train[col].unique()) < 20 or train[col].dtype.char == 'O':
        max_percentage_of_occurances = train[col].value_counts().values.max() / train.shape[0]
        if max_percentage_of_occurances > 0.8:
            print("\t", col)
            # print('Analysing column {0}:'.format(col))
            # print(train[col].value_counts(dropna=False))
            # print('\n')
            train.drop(columns = [col], inplace = True)


print('\n')
i = 0 # This variable is only here to secure id for each hot-encoded categorical column
print('Categorical columns, that are one hot-encoded')
nonnumeric_cols = []
for col in train.columns:
    if len(train[col].unique()) < 20 or train[col].dtype.char == 'O':
        print("\t", col) 
#        print('Analysing column {0}:'.format(col))
        one_hot = pd.get_dummies(train[col], prefix = 'Dummy_{0}'.format(i), drop_first = True)
        train = train.join(one_hot)
        i += 1
#         in case we need to work on which columns shoud be deleted
#         print('\t mode is {0}'.format(train[col].mode()[0]))
#         mode_category = train[col].mode()[0]
#         cols_to_dummie = train[col].unique()
#         np.delete(cols_to_dummie, np.where(cols_to_dummie == mode_category))
#         print(train[col].value_counts(dropna=False))
#         print('\n')

print("\n")
print("number of columns: ", train.shape[1])
