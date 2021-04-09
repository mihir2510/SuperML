from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from auto_machine_learning.utils import get_features
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#---------------------------------------------------------------------------------------------------------------------#

def preprocess_data(dataset, label, task='classification'):
    '''
    Implements all the preprocessing steps: remove null, label encode, oversampling, data split, correlation matrix

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe
                    task (string) : type of task default is classification

            Returns:
                    dataset(dataframe) : processed data to be used for training model
    '''

    try:
        dataset = remove_null(dataset,label)
    except Exception as e:
        raise type(e)("Error at remove_null. Check the input data and label")
        
    try:
        dataset = label_encode(dataset,label) 
    except Exception as e:
        raise type(e)("Error at label_encode. Check the input data and label")

    if task == 'classification':
        dataset = oversampling(dataset,label)

    correlation_matrix(dataset,label)

    print('Data Preprocessing Complete')
    return dataset

#---------------------------------------------------------------------------------------------------------------------#

def remove_null(dataset, label):
    '''
    Removes the null rows and features.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe

            Returns:
                    dataset(dataframe) : processed data to be used for training model
    
    '''
    #get features
    features = get_features(dataset,label)

    # Removing Columns with more than 50% null data
    for feature in features:
        if (dataset[feature].isnull().sum()/len(dataset) > 0.5):
            dataset.drop([feature], axis=1, inplace=True)
            features.remove(feature)

    # Removing rows having null values
    dataset.dropna(inplace=True)

    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(['Unnamed: 0'],axis=1,inplace=True)

    return dataset

#---------------------------------------------------------------------------------------------------------------------#

def label_encode(dataset, label):
    '''
    Label encode the data

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe

            Returns:
                    dataset(dataframe) : processed data to be used for training model
    
    '''
    #get features
    features = get_features(dataset,label)

    #for features
    for feature in features:
        if dataset[feature].dtype == object:

            encoder = LabelEncoder()
            encoder.fit(dataset[feature])
            dataset[feature] = encoder.transform(dataset[feature])
    #for label
    if dataset[label].dtype == object:
        dataset[label] = LabelEncoder().fit_transform(dataset[label])

    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(['Unnamed: 0'],axis=1,inplace=True)

    return dataset

#---------------------------------------------------------------------------------------------------------------------#

def oversampling(dataset, label):
    '''
    Oversamples the data to get rid of skewness from the dataset.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe

            Returns:
                    dataset(dataframe) : processed data to be used for training model
    '''
    
    #get features
    features = get_features(dataset,label)
    new_dataset = None

    if True:
        oversampler=RandomOverSampler()
        X,Y=oversampler.fit_resample(dataset[features],dataset[label])

        new_dataset = pd.DataFrame(X,columns=features)
        new_dataset[label] = pd.DataFrame(Y)
    else:
        new_dataset = dataset

    if 'Unnamed: 0' in new_dataset.columns:
        new_dataset.drop(['Unnamed: 0'],axis=1,inplace=True)

    return new_dataset

#---------------------------------------------------------------------------------------------------------------------#

def dataset_split(dataset, label, test_size=0.3, random_state = 1):
    '''
    Splits the dataset in train and test data on the basis of given fraction.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe
                    test_size (float) : fraction of data to be alloted for testing
                    random_state (int) : Random state for the dataset

            Returns:
                    dataset(dataframe) : processed data to be used for training model
    '''
    #get features
    features = get_features(dataset, label)

    try:
        X, Y = dataset[features], dataset[label]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    except Exception as e:
        print(e+' '+"Check data and test_size")

    return X_train, X_test, Y_train, Y_test

#---------------------------------------------------------------------------------------------------------------------#

def correlation_matrix(dataset, label):
    '''
    Plots the heat map for correlations between the features and labels.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe


    
    '''
    #get features
    features = get_features(dataset,label)
    
    try:
        correlation = dataset[features].corr().abs()
        f, ax = plt.subplots(figsize=(20,20))
        sns.heatmap(correlation,linecolor='black',linewidths=2, annot=True, ax=ax)
    except Exception as e:
        print(e+' '+"Please check the data and label")
    
    #plot the correlation matrix
    #plt.show()
    print('Correlation Matrix Saved')
    plt.savefig('correlation_matrix.png')

#---------------------------------------------------------------------------------------------------------------------#