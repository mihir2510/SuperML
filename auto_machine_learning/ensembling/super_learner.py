from auto_machine_learning.utils import get_model
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate
from auto_machine_learning.data_preprocessing.preprocessing import dataset_split
from auto_machine_learning.hyperparameter_optimization.hpo_methods import bayesian_tpe
import numpy as np

#---------------------------------------------------------------------------------------------------------------------#

class SuperLearnerRegressor():
    '''
    Super learner algorithm for regression tasks.

    '''

    def __init__(self, base_layer_models=None, meta_model='RandomForestRegressor', n_splits=5, optimize=True, max_evals=100):
        '''
        Initializes the object values with given arguments.

                Parameters:
                        self (object ref)
                        base_layer_model (list) : list of models to be used at base layer in ensembling
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        n_splits(int) : number of splits to be made for cross validation
                        optimize(boolean) : optimize the process 
                        max_evals(int) : max number of evaluations to be done
                        
        '''
        if base_layer_models == None:
            # base_layer_models = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor','GradientBoostingRegressor']
            base_layer_models = ['LinearRegression', 'Ridge', 'Lasso']

        self.models = [get_model(model) for model in base_layer_models]

        if meta_model == None:
            self.meta_model = None
        elif meta_model == 'AdaBoostRegressor':
            self.meta_model = get_model(meta_model)(base_estimator=get_model('RandomForestRegressor')())
        else:
            self.meta_model = get_model(meta_model)()

        self.n_splits = n_splits
        self.trained_models = []
        self.optimize = optimize
        self.max_evals = max_evals


    def add_models(self, base_layer_models):
        '''
        Initializes the base layer models of the object

                Parameters:
                        self (object ref)
                        base_layer_model (list) : list of models to be used at base layer in ensembling
                        
                        
        '''
        self.models.extend([get_model(model) for model in base_layer_models])

    def set_meta_model(self, meta_model):
        '''
            Initializes the meta layer model .
                Parameters:
                        self (object ref)
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        
                        
        '''
        self.meta_model = meta_model

    def get_out_of_fold_predictions(self):
        '''
        Returns the out of fold predictions

                Parameters:
                        self (object ref)
                        
                Returns:
                        meta_x (v_stack) : training data for meta layer
                        meta_y (array) : label for meta layer
        '''
        meta_X = list()
        meta_Y = list()
        k_folds = KFold(n_splits=self.n_splits, shuffle=True)

        for train_idx, val_idx in k_folds.split(self.X_train):
            fold_Y_pred = list()
            X_train, X_val,= self.X_train.values[train_idx], self.X_train.values[val_idx]
            Y_train, Y_val = self.Y_train.values[train_idx], self.Y_train.values[val_idx]
            meta_Y.extend(Y_val)

            Y_predictions = list()
            for model_class in self.models:
                model = model_class()
                model.fit(X_train,Y_train)
                Y_pred = model.predict(X_val)
                Y_pred = Y_pred.reshape(len(Y_pred), 1)

                train_score = model.score(X_train, Y_train)
                val_score = model.score(X_val, Y_val)

                Y_predictions.append([train_score-val_score, Y_pred])

            Y_predictions.sort(key=lambda x : x[0])
            selected_Y_pred = Y_predictions[-1][1]

            for Y_pred in Y_predictions:
                if Y_pred[0] >= 0:
                    selected_Y_pred = Y_pred[1]
                    break

            fold_Y_pred.append(selected_Y_pred)
            meta_X.append(np.hstack(fold_Y_pred))

        return np.vstack(meta_X), np.asarray(meta_Y)

    def fit_base_models(self, X_train, Y_train):
        '''
        Trains all the base layer models

                Parameters:
                        self (object ref)
                        X_train (dataframe) : training data for base layer
                        Y_train (series) : labels for base layer
                        
        '''
        for model_class in self.models:
            if self.optimize:
                X_tr, X_te, Y_tr, Y_te = train_test_split(X_train, Y_train, random_state=1, test_size=0.3)
                model = bayesian_tpe(model_class, X_tr, X_te, Y_tr, Y_te, 'prediction', self.max_evals)
            else:
                model = model_class()
                model.fit(X_train, Y_train)

            self.trained_models.append(model)

    def fit_meta_model(self, meta_model):
        '''
        Fit the model at the meta layer.

                Parameters:
                        self (object ref)
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        
        '''
        meta_model.fit(self.meta_X, self.meta_Y)
        return meta_model

    def fit(self, X, Y):
        '''
        Initializes the object values with given arguments.

                Parameters:
                        self (object ref)
                        X (dataframe) : data to train
                        Y (series) : labels for the given dataset
                        
        '''
        self.X_train = X
        self.Y_train = Y
        self.meta_X, self.meta_Y = self.get_out_of_fold_predictions()
        self.fit_base_models(self.X_train, self.Y_train)
        self.meta_model = self.fit_meta_model(self.meta_model)

    def predict_base_models(self, X):
        '''
        Generate predictions for the base layer models

                Parameters:
                        self (object ref)
                        X (dataframe) : data to generate predictions
                        
        '''
        meta_X = list()
        for model in self.trained_models:
            Y = model.predict(X)
            Y = Y.reshape(len(Y), 1)
            meta_X.append(Y)

        meta_X = np.hstack(meta_X)
        meta_X = meta_X.mean(axis=1).reshape((-1,1))
        return meta_X

    def predict_meta_model(self, meta_model, meta_X):
        '''
        Initializes the object values with given arguments.

                Parameters:
                        self (object ref)
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        meta_x (dataframe) : Data for meta model to train on
                        
        '''
        Y_pred = meta_model.predict(meta_X)
        return Y_pred

    def predict(self, X):
        '''
        Generate predictions

                Parameters:
                        self (object ref)
                        X (dataframe) : data to generate predictions
                        
        '''
        meta_X = self.predict_base_models(X)
        Y_pred = self.predict_meta_model(self.meta_model, meta_X)
        return Y_pred

#---------------------------------------------------------------------------------------------------------------------

class SuperLearnerClassifier():
    '''
    Super learner algorithm for classification tasks.

    '''

    def __init__(self, base_layer_models=None, meta_model='RandomForestClassifier', n_splits=5, optimize=True, max_evals=100):
        '''
        Initializes the object values with given arguments.

                Parameters:
                        self (object ref)
                        base_layer_model (list) : list of models to be used at base layer in ensembling
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        n_splits(int) : number of splits to be made for cross validation
                        optimize(boolean) : optimize the process 
                        max_evals(int) : max number of evaluations to be done
                        
        '''
        if base_layer_models == None:
            #base_layer_models = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier','ExtraTreesClassifier','AdaBoostClassifier']
            base_layer_models = ['LogisticRegression', 'DecisionTreeClassifier']

        self.models = [get_model(model) for model in base_layer_models]

        if meta_model == None:
            self.meta_model = None
        elif meta_model == 'AdaBoostClassifier':
            self.meta_model = get_model(meta_model)(base_estimator=get_model('RandomForestClassifier')())
        else:
            self.meta_model = get_model(meta_model)()

        self.n_splits = n_splits
        self.trained_models = []
        self.optimize = optimize
        self.max_evals = max_evals

    def add_models(self, base_layer_models):
        '''
        Initializes the base layer models of the object

                Parameters:
                        self (object ref)
                        base_layer_model (list) : list of models to be used at base layer in ensembling
                        
                        
        '''
        self.models.extend([get_model(model) for model in base_layer_models])

    def set_meta_model(self, meta_model):
        '''
            Initializes the meta layer model .
                Parameters:
                        self (object ref)
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        
                        
        '''
        self.meta_model = meta_model

    def get_out_of_fold_predictions(self):
        '''
        Returns the out of fold predictions

                Parameters:
                        self (object ref)
                        
                Returns:
                        meta_x (v_stack) : training data for meta layer
                        meta_y (array) : label for meta layer
        '''
        meta_X = list()
        meta_Y = list()
        k_folds = KFold(n_splits=self.n_splits, shuffle=True)

        for train_idx, val_idx in k_folds.split(self.X_train):
            fold_Y_pred = list()
            X_train, X_val = self.X_train.values[train_idx], self.X_train.values[val_idx]
            Y_train, Y_val = self.Y_train.values[train_idx], self.Y_train.values[val_idx]
            meta_Y.extend(Y_val)

            Y_predictions=list()
            for model_class in self.models:
                model= model_class()
                model.fit(X_train, Y_train)
                Y_pred = model.predict_proba(X_val)
                fold_Y_pred.append(Y_pred)

            meta_X.append(np.hstack(fold_Y_pred))

        return np.vstack(meta_X), np.asarray(meta_Y)

    def fit_base_models(self, X_train, Y_train):
        '''
        Trains all the base layer models

                Parameters:
                        self (object ref)
                        X_train (dataframe) : training data for base layer
                        Y_train (series) : labels for base layer
                        
        '''
        X_tr, X_te, Y_tr, Y_te = train_test_split(X_train, Y_train, test_size=0.3, random_state=1)
        for model_class in self.models:
            if self.optimize:
                model = bayesian_tpe(model_class, X_tr, X_te, Y_tr, Y_te, 'classification', self.max_evals)
            else:
                model = model_class()
                model.fit(X_train, Y_train)

            self.trained_models.append(model)

    def fit_meta_model(self, meta_model):
        '''
        Fit the model at the meta layer.

                Parameters:
                        self (object ref)
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        
        '''
        meta_model.fit(self.meta_X, self.meta_Y)
        return meta_model

    def fit(self, X, Y):
        '''
        Initializes the object values with given arguments.

                Parameters:
                        self (object ref)
                        X (dataframe) : data to train
                        Y (series) : labels for the given dataset
                        
        '''
        self.X_train = X
        self.Y_train = Y
        self.meta_X, self.meta_Y = self.get_out_of_fold_predictions()
        self.fit_base_models(self.X_train, self.Y_train)
        self.fit_meta_model(self.meta_model)

    def predict_base_models(self, X):
        '''
        Generate predictions for the base layer models

                Parameters:
                        self (object ref)
                        X (dataframe) : data to generate predictions
                        
        '''
        meta_X = list()
        for model in self.trained_models:
            Y = model.predict_proba(X)
            meta_X.append(Y)

        meta_X = np.hstack(meta_X)
        return meta_X

    def predict_meta_model(self, meta_model, meta_X):
        '''
        Initializes the object values with given arguments.

                Parameters:
                        self (object ref)
                        meta_layer_model (string) : name of model to be used at meta layer in ensembling
                        meta_x (dataframe) : Data for meta model to train on
                        
        '''
        Y_pred = meta_model.predict(meta_X)
        return Y_pred

    def predict(self, X):
        '''
        Generate predictions

                Parameters:
                        self (object ref)
                        X (dataframe) : data to generate predictions
                        
        '''
        meta_X = self.predict_base_models(X)
        Y_pred = self.predict_meta_model(self.meta_model, meta_X)
        return Y_pred

#---------------------------------------------------------------------------------------------------------------------#