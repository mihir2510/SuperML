from data_preprocessing.preprocessing import *
from feature_engineering.select_from_model import select_from_model
from feature_engineering.anova import anova_classifier, anova_regressor
from feature_engineering.correlation import correlation
from feature_engineering.pca import pca
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from hyperparameter_optimization.hpo_methods import *
from hyperparameter_optimization.hpo import get_trained_model
from utils import map_model

all_feature_engg_techs = ['anova', 'correlation', 'pca', 'select_from_model']
all_hpo_techs = ['grid_search', 'random_search', 'bayesian_tpe', 'bayesian_gp']
models = list(map_model.keys())


techniques_dict = {
    mod.__name__ : mod for mod in [anova_classifier, anova_regressor, correlation, pca, select_from_model, grid_search, random_search, bayesian_tpe, bayesian_gp]
}

def automate(dataset,label,task,feature_engg_techs=all_feature_engg_techs, hpo_techs=all_hpo_techs, models=models ,modelClass=None, threshold=0.9, max_evals=500, test_size=0.3, random_state=1):
    # dataset = preprocess_data(dataset,label)
    stats = []
    print(feature_engg_techs)
    print(hpo_techs)
    print(models)
    dataset = remove_null(dataset,label)
    dataset = label_encode(dataset,label)
    original_dataset = dataset.copy()
    for fe_tech in feature_engg_techs:
        if fe_tech == 'anova':
            fe_tech = 'anova_regressor' if task == 'prediction' else 'anova_classifier'
        for hpo_tech in hpo_techs:
            for model_name in models:
                print(fe_tech, hpo_tech, model_name)
                if task != 'prediction' and fe_tech != 'pca':
                    dataset_local = oversampling(dataset,label)
                if fe_tech.startswith('anova') and modelClass:
                    dataset_local = techniques_dict[fe_tech](dataset, label, map_model[model_name])
                elif fe_tech == 'correlation':
                    dataset_local = techniques_dict[fe_tech](dataset, label, threshold)
                elif fe_tech == 'select_from_model':
                    dataset_local = techniques_dict[fe_tech](dataset, label, map_model[model_name])
                else:
                    dataset_local = techniques_dict[fe_tech](dataset, label)
                model = get_trained_model(dataset_local, label, model_name, hpo_tech, task, max_evals, test_size, random_state)
                features = get_features(dataset_local, label)
                X, Y = dataset_local[features], dataset_local[label]
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,random_state=random_state)
                stats.append((fe_tech, hpo_tech, model_name, model.score(X_test,Y_test)))
                print(stats[-1][-1])
                dataset = original_dataset.copy()
    return stats


