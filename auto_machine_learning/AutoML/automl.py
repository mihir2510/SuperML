from auto_machine_learning.data_preprocessing.preprocessing import *
from auto_machine_learning.feature_engineering import pca, anova
from auto_machine_learning.ensembling.super_learner import *
from auto_machine_learning.metrics.metrics import *
from auto_machine_learning.utils import *
from auto_machine_learning.hyperparameter_optimization.hpo import get_trained_model
import pandas as pd
import sklearn

def auto_ensemble(dataset, label, task, base_layer_models=None, meta_layer_model=None, n_splits=5, optimize=True, max_evals=100, download_model = None):   
    '''
        Implements Automated Ensembling based on the base layer and meta layer models provided.
            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe  
                    task (string) : type of task 
                    n_splits(int) : number of splits to be made for cross validation
                    optimize(boolean) : optimize the process
                    max_evals(int) :  max number of evaluations to be done
                    download_model(boolean) : save the final trained ensembleo
                    metric(string)
            Returns:     
                    ensemble(model object) : trained super learner model   
    '''

    # Data Preprocessing
    prepocessed_dataset = preprocess_data(dataset, label, task)
    print('\nData Preprocessed.\n')

    # Feature Engineering
    features = get_features(prepocessed_dataset, label)
    if (len(features) > 15):
        feature_engineered_dataset = pca.pca(prepocessed_dataset, label)
    elif task == 'prediction':
        feature_engineered_dataset = anova.anova_regressor(prepocessed_dataset, label)
    elif task == 'classification':
        feature_engineered_dataset = anova.anova_classifier(prepocessed_dataset, label)
    else:
        feature_engineered_dataset = prepocessed_dataset.copy()

    print('\nFeature Engineering performed\n')

    X_train, X_test, Y_train, Y_test = dataset_split(feature_engineered_dataset, label)

    # Ensembling
    if task == 'prediction':
        metric = 'r2' if metric == None else metric
        try:
            ensemble = SuperLearnerRegressor(base_layer_models, meta_layer_model, n_splits, optimize, max_evals)
            ensemble.fit(X_train, Y_train)
        except Exception as e:
            raise type(e)("Please check the values of base_layer_models,meta_layer_models")

    elif task == 'classification':
        metric = 'f1' if metric == None else metric
        try:
            ensemble = SuperLearnerClassifier(base_layer_models, meta_layer_model, n_splits, optimize, max_evals)
            ensemble.fit(X_train, Y_train)
        except Exception as e:
            raise type(e)("Please check the values of base_layer_models,meta_layer_models")
    print('\nEnsemble trained\n')

    #Get Statistics
    stats = get_model_metrics(ensemble, dataset[label], task, X_test, Y_test)
    print(stats)
    
    #Download model    
    if download_model:
        pickle_model(ensemble, 'automl-ensembled-file')
        print('\nPickle file generated.\n')
    
    #Return Model
    print('\nEnsemble Returned.\n')
    return ensemble


def automl_run(dataset, label, task, base_layer_models=None, meta_layer_models=None, n_splits=5, optimize=True, max_evals=100, download_model = None, metric=None, sortby=None, excel_file=None):
    '''
    Implements the whole automl pipeline. Consists of the stages: Datapreprocessing -> Feature Engineering -> HPO -> Ensembling.
    It creates a super learner using the base layer and meta layer to combine the performance of various models.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe  
                    task (string) : type of task 
                    base_layer_models (list) : list of models to be used at base layer in ensembling
                    meta_layer_models (list) : list of models to be used at meta layer in ensembling
                    n_splits(int) : number of splits to be made for cross validation
                    optimize(boolean) : optimize the process 
                    max_evals(int) : max number of evaluations to be done
                    download_model(string) : name of the file to be used for saving model
                    metric(string) : metric to select best model
                    sortby (string) : sort the result as per metric
                    excel_file(strig) : name of the file to be used for saving stats

            Returns:
                    stats (dictionary): contains the metrics for given data
                    ensemble (model object) : trained superlearner model based on given metric
    '''

    # Data Preprocessing
    prepocessed_dataset = preprocess_data(dataset, label, task)

    print('\nData Preprocessed.\n')

    # Feature Engineering
    features = get_features(prepocessed_dataset, label)
    if (len(features) > 15):
        feature_engineered_dataset = pca.pca(prepocessed_dataset, label)
    elif task == 'prediction':
        feature_engineered_dataset = anova.anova_regressor(prepocessed_dataset, label)
    elif task == 'classification':
        feature_engineered_dataset = anova.anova_classifier(prepocessed_dataset, label)
    else:
        feature_engineered_dataset = prepocessed_dataset.copy()
    print('\nFeature Engineering performed\n')

    X_train, X_test, Y_train, Y_test = dataset_split(feature_engineered_dataset, label)

    # Ensembling
    if task == 'prediction':
        metric = 'r2' if metric == None else metric
        notallowed=['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'BaggingClassifier', 'GradientBoostingClassifier']
        for model in base_layer_models:
            if model in notallowed:
                raise Exception("Invalid base_layer_models for the given task")
        for model in meta_layer_models:
            if model in notallowed:
                raise Exception('Invalid meta_layer_models for the given task')
        base_layer_models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor'] if base_layer_models == None else base_layer_models

        meta_layer_models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor'] if meta_layer_models == None else meta_layer_models

        trained_models = []
        for base_layer_model in base_layer_models:
            trained_model = get_trained_model(feature_engineered_dataset, label, base_layer_model, task, 'bayesian_tpe')
            Y_pred = trained_model.predict(X_test)
            stats = get_model_metrics_ensemble(feature_engineered_dataset[label], task, Y_test, Y_pred)
            trained_models.append([trained_model, stats[metric]])
        
        trained_models.sort(key=lambda x : x[1], reverse=True)
        column_names = ['Meta Layer Model', 'Base Layer Models']+list(stats.keys())
        stats_list = []
        
        ensemble = SuperLearnerRegressor([])
        ensemble.X_train = X_train
        ensemble.Y_train = Y_train

        for meta_layer_model in meta_layer_models:
            ensemble.set_meta_model(get_model(meta_layer_model)())
            for i in range(1, len(trained_models)+1):
                ensemble.models = [trained_model[0].__class__ for trained_model in trained_models[:i]]
                ensemble.trained_models = [trained_model[0] for trained_model in trained_models[:i]]

                ensemble.meta_X, ensemble.meta_Y = ensemble.get_out_of_fold_predictions()
                ensemble.fit_meta_model(ensemble.meta_model)

                stats = get_model_metrics(ensemble, feature_engineered_dataset[label], task, X_test, Y_test)
                temp = [ensemble, meta_layer_model]
                temp.append(', '.join([model.__name__ for model in ensemble.models]))
                temp.extend(list(stats.values()))
                stats_list.append(temp)

    elif task == 'classification':
        metric = 'f1' if metric == None else metric
        notallowed = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor']
        for model in base_layer_models:
            if model in notallowed:
                raise Exception("Invalid base_layer_models for the given task")
        for model in meta_layer_models:
            if model in notallowed:
                raise Exception('Invalid meta_layer_models for the given task')
        base_layer_models = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'BaggingClassifier', 'GradientBoostingClassifier'] if base_layer_models == None else base_layer_models

        meta_layer_models = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'BaggingClassifier', 'GradientBoostingClassifier'] if meta_layer_models == None else meta_layer_models

        trained_models = []
        for base_layer_model in base_layer_models:
            trained_model = get_trained_model(feature_engineered_dataset, label, base_layer_model, task, 'bayesian_tpe')
            Y_pred = trained_model.predict(X_test)
            stats = get_model_metrics_ensemble(feature_engineered_dataset[label], task, Y_test, Y_pred)
            trained_models.append([trained_model, stats[metric]])
        
        trained_models.sort(key=lambda x : x[1], reverse=True)
        column_names = ['Meta Layer Model', 'Base Layer Models']+list(stats.keys())
        stats_list = []
        
        ensemble = SuperLearnerRegressor([])
        ensemble.X_train = X_train
        ensemble.Y_train = Y_train

        for meta_layer_model in meta_layer_models:
            ensemble.set_meta_model(get_model(meta_layer_model)())
            for i in range(1, len(trained_models)+1):
                ensemble.models = [trained_model[0].__class__ for trained_model in trained_models[:i]]
                ensemble.trained_models = [trained_model[0] for trained_model in trained_models[:i]]

                ensemble.meta_X, ensemble.meta_Y = ensemble.get_out_of_fold_predictions()
                ensemble.fit_meta_model(ensemble.meta_model)

                stats = get_model_metrics(ensemble, feature_engineered_dataset[label], task, X_test, Y_test)
                temp = [ensemble, meta_layer_model]
                temp.append(', '.join([model.__name__ for model in ensemble.models]))
                temp.extend(list(stats.values()))
                stats_list.append(temp)
    print('\nEnsemble trained\n')
    
    #To sort on basis of metric provided
    if sortby:
        index = column_names.index(sortby)
        stats_list.sort(key= lambda x: x[index],reverse=True)

    #Download model
    if download_model:
        pickle_model(stats_list[0][0],download_model)       
    pd_stats = pd.DataFrame(stats_list)
    pd_stats.drop(pd_stats.columns[0], axis=1,inplace=True)
    pd_stats.columns = column_names
    
    #Download Excel File
    if excel_file:
        get_csv(pd_stats,excel_file)

    #Return statistics in form of dataframe and model
    return pd_stats,stats_list[0][0]