from MyAutoMLLibrary.utils import pickle_model
from MyAutoMLLibrary.data_preprocessing.preprocessing import *
from MyAutoMLLibrary.feature_engineering import pca, anova
from MyAutoMLLibrary.ensembling.super_learner import *
from MyAutoMLLibrary.metrics.metrics import *
import sklearn

def automl(dataset, label, task, base_layer_models=None, meta_layer_model=None, n_splits=5, optimize=True, max_evals=100, download_model = None, metric=None):
    '''
    Implements the whole automl pipeline. Consists of the stages: Datapreprocessing -> Feature Engineering -> HPO -> Ensembling.

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe  
                    task (string) : type of task 
                    base_layer_models (list) : list of models to be used at base layer in ensembling
                    meta_layer_models (list) : list of models to be used at meta layer in ensembling
                    n_splits(int) : number of splits to be made for cross validation
                    optimize(boolean) : optimize the process
                    max_evals(int) :  max number of evaluations to be done
                    download_model(boolean) : save the final trained ensemble
                    metric(string)
            Returns:
                    stats (dictionary): contains the metrics for given data
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
    stats = get_model_metrics(ensemble, dataset[label], task, X_test, Y_test)
    print(stats)
    
    if download_model:
        pickle_model(ensemble, 'automl-ensembled-file')
        print('\nPickle file generated.\n')
        return None
    else:
        print('\nEnsemble Returned.\n')
        return ensemble