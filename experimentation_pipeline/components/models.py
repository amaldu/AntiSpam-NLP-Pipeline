from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def get_model(model_type, params=None):
    default_params = {
        'logistic': {'C': 1.0, 'solver': 'lbfgs', 'max_iter': 100},
        'naive_bayes': {'alpha': 1.0},
        'svc': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
    }
    
    if params is None:
        params = default_params.get(model_type, {})
    
    if model_type == 'logistic':
        model = LogisticRegression(**params)
    elif model_type == 'naive_bayes':
        model = MultinomialNB(**params)
    elif model_type == 'svc':
        model = SVC(**params)
    else:
        raise ValueError("Modelo no soportado. Debes elegir entre 'logistic', 'naive_bayes' o 'svc'.")
    
    return model
