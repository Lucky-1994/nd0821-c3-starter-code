import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_leaf=2, min_samples_split=3,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : GradientBoostingClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


def compute_model_metrics_on_slices(model, data, cat_features, encoder, lb):
    """Print metrics for fixed categorical features

    Parameters
    ----------
    model: trained machine learning model
    data: data used to analyze the model
    cat_features: list of categorical features
    encoder: encoder
    lb: label binarizer

    Returns
    -------
    None
    """
    for cat_feature in cat_features:
        for unique_cat in data[cat_feature].unique():
            data_slice = data[data[cat_feature] == unique_cat]
            X_slice, y_slice, encoder, lb = process_data(
                data_slice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )
            predictions_slice = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, predictions_slice)
            print(build_metrics_string(precision, recall, fbeta, prefix=f"Feature: {unique_cat} [{cat_feature}]\n\t"))


def build_metrics_string(precision, recall, fbeta, prefix=""):
    """

    Parameters
    ----------
    precision: precision score
    recall: recall score
    fbeta: f1 score
    prefix: additional string and the beginning

    Returns
    -------
    metrics_string: combination of scores into single string
    """
    return prefix + f"Precision: {round(precision, 2)} \t\tRecall: {round(recall, 2)} \t F1 Score: {round(fbeta, 2)}"


def save_model(model_path, model, encoder, lb):
    """Save a model and other necessary components in the desired path

    Parameters
    ----------
    model_path: path to save the model in
    model: trained model
    encoder: fitted encoder
    lb: fitted label binarizer

    Returns
    -------
    None
    """
    with open(model_path, 'wb') as f:
        pickle.dump([model, encoder, lb], f)


def load_model(model_path):
    """Load a saved model and other necessary components

    Parameters
    ----------
    model_path: path to the saved model

    Returns
    -------
    model: trained model
    encoder: fitted encoder
    lb: fitted label binarizer
    """
    with open(model_path, 'rb') as f:
        model, encoder, lb = pickle.load(f)
    return model, encoder, lb
