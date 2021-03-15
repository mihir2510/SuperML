from MyAutoMLLibrary.data_preprocessing.preprocessing import *
from MyAutoMLLibrary.feature_engineering.select_from_model import select_from_model
from MyAutoMLLibrary.feature_engineering.anova import anova_classifier, anova_regressor
from MyAutoMLLibrary.feature_engineering.correlation import correlation
from MyAutoMLLibrary.feature_engineering.pca import pca
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from MyAutoMLLibrary.hyperparameter_optimization.hpo_methods import *
from MyAutoMLLibrary.hyperparameter_optimization.hpo import get_trained_model
from MyAutoMLLibrary.utils import *
from MyAutoMLLibrary.metrics.metrics import get_model_metrics
import pandas as pd

default_feature_engineering_methods = ['all_features','anova', 'correlation', 'pca', 'select_from_model']
default_hyperparamter_methods = ['standard','grid_search', 'random_search', 'bayesian_tpe', 'bayesian_gp']
models = list(map_model.keys())


techniques_dict = {
    mod.__name__ : mod for mod in [anova_classifier, anova_regressor, correlation, pca, select_from_model, grid_search, random_search, bayesian_tpe, bayesian_gp]
}

def auto_train(dataset,label,task,feature_engineering_methods=default_feature_engineering_methods, hpo_methods=default_hyperparamter_methods, models=models ,modelClass=None, sortby=None, download_model = None,excel_file=None, threshold=0.9, max_evals=500, test_size=0.3, random_state=1):
    '''
    Implements the whole automl pipeline. Consists of the stages: Datapreprocessing -> Feature Engineering -> HPO -> Ensembling.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe  
                    task (string) : type of task 
                    feature_engineering_methods(list): list of feature engineering methods to be implemented
                    hpo_methods(list): list of hpo methods to be implemented
                    models(list): list of models to be used for generating results
                    modelClass (class of model) : model to be used as base for featureengineering technique
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
    dataset = remove_null(dataset,label)
    dataset = label_encode(dataset,label)
    correlation_matrix(dataset,label)
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
                    dataset = oversampling(dataset,label)
                if feature_engineering_method.startswith('anova') and modelClass:
                    dataset = techniques_dict[feature_engineering_method](dataset, label, modelClass)
                elif feature_engineering_method == 'correlation':
                    dataset = techniques_dict[feature_engineering_method](dataset, label, threshold)
                elif feature_engineering_method == 'select_from_model':
                    dataset = techniques_dict[feature_engineering_method](dataset, label, map_model[model_name])
                elif feature_engineering_method == 'all_features':
                    pass
                else:
                    dataset = techniques_dict[feature_engineering_method](dataset, label)
                model = get_trained_model(dataset, label, model_name, task,hpo_method, max_evals, test_size, random_state)
                features = get_features(dataset, label)
                X, Y = dataset[features], dataset[label]
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=random_state)
                model_metrics = get_model_metrics(model,label,task,X_test, Y_test)
                column_names = ['Estimator', 'Feature Engineering Method', 'Hyperparameter Optimisation Method']
                column_names.extend(list(model_metrics.keys()))
                model_metrics = list(map(lambda value : round(value, 4), model_metrics.values()))
                stats.append([model,model_name,feature_engineering_method,hpo_method]+list(model_metrics))

                dataset = original_dataset.copy()

    if sortby:
        index = column_names.index(sortby)
        stats.sort(key= lambda x: x[index],reverse=True)
    if download_model:
        pickle_model(stats[0][0],download_model)       
    pd_stats = pd.DataFrame(stats)
    pd_stats.drop(pd_stats.columns[0], axis=1,inplace=True)
    pd_stats.columns = column_names
    
    if excel_file:
        get_csv(pd_stats,excel_file)

    return pd_stats,stats[0][0]

