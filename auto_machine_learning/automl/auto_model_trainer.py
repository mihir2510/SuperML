from auto_machine_learning.data_preprocessing.preprocessing import *
from auto_machine_learning.feature_engineering.select_from_model import select_from_model
from auto_machine_learning.feature_engineering.anova import anova_classifier, anova_regressor
from auto_machine_learning.feature_engineering.correlation import correlation
from auto_machine_learning.feature_engineering.pca import pca
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from auto_machine_learning.hyperparameter_optimization.hpo_methods import *
from auto_machine_learning.hyperparameter_optimization.hpo import get_trained_model
from auto_machine_learning.utils import *
from auto_machine_learning.metrics.metrics import get_model_metrics
import pandas as pd

default_feature_engineering_methods = ['all_features','anova', 'correlation', 'pca', 'select_from_model']
default_hyperparamter_methods = ['standard','grid_search', 'random_search', 'bayesian_tpe']
models = list(map_model.keys())

name_holder = {
    'LinearRegression' : 'LiR',
    'Ridge' : "RR",
    'Lasso' : "LaR",
    'DecisionTreeRegressor' : 'DTR',
    'RandomForestRegressor' : 'RFR',
    'AdaBoostRegressor' : 'ABR',
    'ExtraTreesRegressor' : 'ETR',
    'BaggingRegressor' : 'BR',
    'GradientBoostingRegressor' : 'GBR',
    'LogisticRegression' : 'LoR',
    'RandomForestClassifier' : 'RFC',
    'AdaBoostClassifier' : 'ABC',
    'BaggingClassifier' : 'BC',
    'GradientBoostingClassifier' : 'GBC',
    'ExtraTreesClassifier' : 'ETC',
    'DecisionTreeClassifier' : 'DTC',
    'standard':'No HPO',
    'grid_search':'GS',
    'random_search':'RS',
    'bayesian_tpe':'BO',
    'all_features' : 'No FE',
    'anova_regressor' : 'ANOVA',
    'anova_classifier' : 'ANOVA',
    'correlation' : 'CoRR',
    'pca' : 'PCA',
    'select_from_model' : 'SFM'
}

techniques_dict = {
    mod.__name__ : mod for mod in [anova_classifier, anova_regressor, correlation, pca, select_from_model, grid_search, random_search, bayesian_tpe, bayesian_gp]
}


def train(dataset, label, task, feature_engineering_method='all_features', hpo_method='standard', model_name=None, threshold=0.9, max_evals=500, test_size=0.3, random_state=1, download_model=None):
    '''
    Implements a pipeline for training the machine learning model. Consists of the stages: Datapreprocessing -> Feature Engineering -> training -> Hpo.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe
                    task (string) : type of task
                    feature_engineering_methods(string): feature engineering method to be implemented
                    hpo_methods(string): list of hpo methods to be implemented
                    model_name(string) : Name of the model to be trained
                    threshold(float) : to decide the useful features to be kept
                    max_evals(int) : max number of evaluations to be done
                    test_size(float) : fraction of data to be used for testing
                    random_state(int) : random state to follow
                    download_model(string) : name of the file to be used for saving model


            Returns:
                    model (mode object): trained model
    '''

    # Preprocessing
    dataset = preprocess_data(dataset, label, task)

    # Feature Engineering
    if feature_engineering_method == 'all_features':
        pass
    elif feature_engineering_method == 'anova':
        if task == 'prediction':
            dataset = anova_regressor(dataset, label)
        else:
            dataset = anova_classifier(dataset, label)
    elif feature_engineering_method == 'correlation':
        dataset = correlation(dataset, label, threshold)
    elif feature_engineering_method == 'pca':
        dataset = pca(dataset, label)
    elif feature_engineering_method == 'select_from_model':
        if task == 'prediction':
            dataset = select_from_model(dataset, label, RandomForestRegressor)
        else:
            dataset = select_from_model(dataset, label, RandomForestClassifier)
    print('Feature Engineering Complete')

    # Hyperparameter optimisation and training the model
    trained_model = get_trained_model(dataset, label, model_name, task, hpo_method, max_evals, test_size, random_state)
    print('Model trained')

    #download model
    if download_model:
        pickle_model(trained_model, download_model)
        print('Pickle file generated.')
    return trained_model

#---------------------------------------------------------------------------------------------------------------------#

def auto_trainer(dataset,label,task,feature_engineering_methods=default_feature_engineering_methods, hpo_methods=default_hyperparamter_methods, models=[] ,anova_estimator=None, sortby=None, download_model = None,excel_file=None, threshold=0.9, max_evals=500, test_size=0.3, random_state=1):
    '''
    Implements the whole automl pipeline. Consists of the stages: Datapreprocessing -> Feature Engineering -> HPO.
    Helps in generating results for various combinations of Feature Engineering methods, HPO methods and models

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe
                    task (string) : type of task
                    feature_engineering_methods(list): list of feature engineering methods to be implemented
                    hpo_methods(list): list of hpo methods to be implemented
                    models(list): list of models to be used for generating results
                    anova_estimator (class of model) : model to be used as base for featureengineering technique
                    sortby(string) : sort the result as per metric
                    download_model(string) : name of the file to be used for saving model
                    excel_file(string) : name of the file to be used for saving stats
                    threshold(float) : to decide the useful features to be kept
                    max_evals(int) : max number of evaluations to be done
                    test_size(float) : fraction of data to be used for testing
                    random_state(int) : random state to follow
            Returns:
                    stats (dictionary): contains the metrics for given data
                    model (mode object): trained model

    '''

    # dataset = preprocess_data(dataset,label)
    stats = []
    if task=='prediction':
        if models == []:
            models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'BaggingRegressor', 'GradientBoostingRegressor']
        notallowed=['LogisticRegression','RandomForestClassifier', 'AdaBoostClassifier', 'BaggingClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'DecisionTreeClassifier']
        print(task)
        print(models)
        for model in models:
            if model in notallowed:
                raise Exception("Input valid model list for the given task")

    else:
        if models == []:
            models = ['LogisticRegression','RandomForestClassifier', 'AdaBoostClassifier', 'BaggingClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'DecisionTreeClassifier']
        notallowed=['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'BaggingRegressor', 'GradientBoostingRegressor']
        for model in models:
            if model in notallowed:
                raise Exception("Input valid model list for the given task")
    try:
        dataset = remove_null(dataset,label) #Change to cumulated function
        dataset = label_encode(dataset,label)
        correlation_matrix(dataset,label)
    except Exception as e:
        raise type(e)("Please check the data and the label given exists in the dataset")

    #Feature Engineering, Hyperparameter Optimization, Model Training
    original_dataset = dataset.copy()
    for feature_engineering_method in feature_engineering_methods:
        if feature_engineering_method == 'anova':
            feature_engineering_method = 'anova_regressor' if task == 'prediction' else 'anova_classifier'
        for hpo_method in hpo_methods:
            if hpo_method == 'standard':
                hpo_method='standard'
            for model_name in models:
                print(feature_engineering_method, hpo_method, model_name)
                if task != 'prediction' and feature_engineering_method != 'pca':
                    try:
                        dataset = oversampling(dataset,label)
                    except Exception as e:
                        raise type(e)("Please check the data and label given properly")
                if feature_engineering_method.startswith('anova') and anova_estimator:
                    try:
                        dataset = techniques_dict[feature_engineering_method](dataset, label, anova_estimator)
                    except Exception as e:
                        raise type(e)("Please check the data, label and the anova_estimator provided properly")
                elif feature_engineering_method == 'correlation':
                    dataset = techniques_dict[feature_engineering_method](dataset, label, threshold)
                elif feature_engineering_method == 'select_from_model':
                    dataset = techniques_dict[feature_engineering_method](dataset, label, map_model[model_name])
                elif feature_engineering_method == 'all_features':
                    pass
                else:
                    dataset = techniques_dict[feature_engineering_method](dataset, label)

                #Model Training
                if model_name=='LogisticRegression' and len(dataset[label].unique())>2:
                    print("The logistic regression requires the output to be binary classification problem")
                else:
                    model = get_trained_model(dataset, label, model_name, task,hpo_method, max_evals, test_size, random_state)
                    features = get_features(dataset, label)
                    X, Y = dataset[features], dataset[label]
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=random_state)
                    model_metrics = get_model_metrics(model,label,task,X_test, Y_test)
                    column_names = ['Estimator', 'Feature Engineering Method', 'Hyperparameter Optimization Method']
                    column_names.extend(list(model_metrics.keys()))
                    model_metrics = list(map(lambda value : round(value, 4), model_metrics.values()))
                    #change names
                    stats.append([model,name_holder[model_name],name_holder[feature_engineering_method],name_holder[hpo_method]]+list(model_metrics))


                dataset = original_dataset.copy()
    
    print(stats)
    #To sort on basis of metric provided
    if sortby:
        index = column_names.index(sortby) + 1
        stats.sort(key= lambda x: x[index],reverse=True)

    # To download model
    if download_model:
        pickle_model(stats[0][0],download_model)
    
    pd_stats = pd.DataFrame(stats)
    pd_stats.drop(pd_stats.columns[0], axis=1,inplace=True)
    pd_stats.columns = column_names

    #Download excelsheet
    if excel_file:
        get_csv(pd_stats,excel_file)

    #Return statistics in form of dataframe and model
    return pd_stats,stats[0][0]
