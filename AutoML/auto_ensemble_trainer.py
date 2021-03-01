from utils import pickle_model
from data_preprocessing.preprocessing import *
from feature_engineering import pca, anova
from ensembling.super_learner import *
from hyperparameter_optimization.hpo import get_trained_model
from metrics.metrics import *
import sklearn
import pandas as pd

def auto_ensemble_trainer(dataset, label, task, base_layer_models=None, meta_layer_models=None, n_splits=5, optimize=True, max_evals=100, download_model = None, metric=None, sortby=False, excel_file=None):
    
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
    # print('\nEnsemble created\n')
    
    print('\nEnsemble trained\n')
    
    if sortby:
        index = column_names.index(sortby)
        stats_list.sort(key= lambda x: x[index],reverse=True)
    if download_model:
        pickle_model(stats_list[0][0],download_model)       
    pd_stats = pd.DataFrame(stats_list)
    pd_stats.drop(pd_stats.columns[0], axis=1,inplace=True)
    pd_stats.columns = column_names
    
    if excel_file:
        pd.get_csv(pd_stats,excel_file)

    return pd_stats,stats_list[0][0]