from utils import pickle_model
from data_preprocessing.preprocessing import *
from feature_engineering import pca, anova
from ensembling.super_learner import *
from metrics.metrics import *
import sklearn

def automl(dataset, label, task, base_layer_models=None, meta_layer_model=None, n_splits=5, optimize=True, max_evals=100, generate_pickle_file=False, metric=None):
    
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
        if meta_layer_model == None:
            ensemble = SuperLearnerRegressor(base_layer_models, None, n_splits, optimize, max_evals)
            ensemble.X_train = X_train
            ensemble.Y_train = Y_train
            ensemble.meta_X, ensemble.meta_Y = ensemble.get_out_of_fold_predictions()
            ensemble.fit_base_models(X_train, Y_train)

            meta_layer_models = []
            for model in ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor']:
                meta_layer_models.append(ensemble.fit_meta_model(get_model(model)()))
            
            meta_X = ensemble.predict_base_models(X_test)
            score = None
            best_meta_model = None

            for meta_model in meta_layer_models:
                Y_pred = ensemble.predict_meta_model(meta_model, meta_X)
                scores = get_model_metrics_ensemble(task, Y_test, Y_pred)

                if score == None or score < scores[metric]:
                    score = scores[metric]
                    best_meta_model = meta_model
            
            ensemble.set_meta_model(best_meta_model)
            print(best_meta_model.__class__.__name__)

        else:
            ensemble = SuperLearnerRegressor(base_layer_models, meta_layer_model, n_splits, optimize, max_evals)
            ensemble.fit(X_train, Y_train)

    elif task == 'classification':
        ensemble = SuperLearnerClassifier(base_layer_models, meta_layer_model, n_splits, optimize, max_evals)
        ensemble.fit(X_train, Y_train)
    print('\nEnsemble created\n')
    
    print('\nEnsemble trained\n')
    stats = get_model_metrics(ensemble, task, X_test, Y_test)
    print(stats)
    
    if generate_pickle_file:
        pickle_model(ensemble, 'automl-ensembled-file')
        print('\nPickle file generated.\n')
        return None
    else:
        print('\nEnsemble Returned.\n')
        return ensemble