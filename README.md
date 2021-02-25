# AutoML-library

### Data Preprocessing:
  get_features

    Parameters
    --------
    - dataset : Pandas Dataframe
    Dataset containing features *AND* labels
    - label : String 
    Column name of label

    Returns
    --------
    List of Features in String

  remove_null

    Parameters
    --------
    - dataset : Pandas Dataframe
    Dataset containing features *AND* labels
    - label : String 
    Column name of label

    Returns
    --------
    Modified dataset with null values removed

label_encode

    Parameters
    --------
    - dataset : Pandas Dataframe
    Dataset containing features *AND* labels
    - label : String 
    Column name of label

    Returns
    --------
    Modified dataset with Label encoded values

oversampling

    Parameters
    --------
    - dataset : Pandas Dataframe
    Dataset containing features AND labels
    - label : String 
    Column name of label

    Returns
    --------
    Modified dataset with oversampling performed


### Feature Engineering:
`TBD`

### Hyperparameter Optimization:

HPO

    Parameters
    --------
    - dataset : Pandas Dataframe
    Dataset containing features AND labels
    - label : String 
    Column name of label
    - Model name : String
    - Method name : String ['all', 'grid', 'random', 'bayesian']
    - Time taken?

    Returns
    --------
    Sklearn model of type `Model name`
