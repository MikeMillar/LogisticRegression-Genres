# External imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from datetime import datetime
import time

# Other models
from sklearn.linear_model import LogisticRegression as lr
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# Internal imports
import utils
from logreg import LogisticRegression

# Global variables
# Data variables
train_path = 'data/train/train_short.csv'        # Path to the training CSV
test_path = 'data/test/test_short.csv'           # Path to the output CSV
save_path = 'data/result/'
split_size = 0.2                                # Testing validation split size percent
variance = 0.85                                 # Percent of variance to keep when reducing dimensionality
# Gradient Ascent variables
learning_rate = 0.01                            # Learning rate used for gradient ascent
penalty = 0.01                                  # Regularization factor used for gradient ascent
epsilon = 1e-9                                  # Early termination error difference used for gradient ascent
max_iterations = int(10e4)                      # Max iterations before termination of gradient ascent

def print_params():
    print(f'train={train_path}, test={test_path}, test_size={split_size}, var={variance}, eta={learning_rate}, pen={penalty}, epsilon={epsilon}, max_iter={max_iterations}')

def load_data(filename: str) -> pd.DataFrame:
    """
    Loads the CSV file path provided into a pandas dataframe.
    
    Args:
        filename (string): Filepath of the CSV file to load

    Returns:
        (pd.Dataframe): Pandas dataframe of the CSV data
    """
    # Check if file is a CSV file
    if not filename.endswith('.csv'):
        print('Cannot load non-csv file into pandas dataframe.')
        return None
    
    return pd.read_csv(filename, index_col=0)

def prepare_data(df, extract_y=False):
    """
    Simplifies the index to just the filenames and not the full paths,
    and if extract_y = True, extracts the label from the dataframe
    and returns it as well.

    Args:
        df (DataFrame): pandas dataframe of the data
        extract_y (bool): True if you want to extract the label from the data

    Returns:
        (DataFrame): pandas dataframe of the fixed index
        (Optional) (np.array): numpy array of Y if extract_y was True
    """
    simple_index = map(utils.get_filename, df.index)
    df.index = simple_index
    if extract_y:
        Y = np.array(df['label'])
        df.drop('label', axis=1, inplace=True)
        return df, Y
    return df
    

def scale_and_reduce(X_train, X_test, Z, variance):
    """
    Trains a Standard Scaler and PCA model with the X_train
    data, then transforms X_train, X_test, and Z by scaling
    and reducing their dimensionality.

    Args:
        X_train (DataFrame): Training data to train and transform with
        X_test (DataFrame): Testing data to transform
        Z (DataFrame): Output data to transform

    Returns:
        X_train (DataFrame): Transformed training data
        X_test (DataFrame): Transformed testing data
        Z (DataFrame): Transform output data
    """
    X_train_index = X_train.index
    X_test_index = X_test.index
    Z_index = Z.index
    # Scale the data to mean = 0, std = 1
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train_index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test_index)
    Z = pd.DataFrame(scaler.transform(Z), columns=Z.columns, index=Z_index)

    # Reduce the data dimensionalit using PCA
    pca = PCA(n_components=variance)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.fit_transform(X_test)
    # Z = pca.fit_transform(Z)
    X_train = pd.DataFrame(pca.fit_transform(X_train), index=X_train_index)
    X_test = pd.DataFrame(pca.transform(X_test), index=X_test_index)
    Z = pd.DataFrame(pca.transform(Z), index=Z_index)
    return X_train, X_test, Z

def load_prep_scale(X_path, Z_path, split_size, variance):
    """
    Loads the data CSV files, cleans the data, then scales and performs
    dimensionality reduction on that data and returns the set of
    transformed data.

    Args:
        X_path (str): Path to the training CSV
        Z_path (str): Path to the output CSV
        split_size (float): Between 0 and 1, the percentage size of the validation test split
        variance (float): Between 0 and 1, the percentage of explained variance to keep

    Returns:
        X_train (DataFrame): Training data
        X_test (DataFrame): Validation testing data
        Y_Train (DataFrame): Labels for the training data
        Y_test (DataFrame): Labels for the validation data
        Z (DataFrame): The output data to be classified
    """
    print("Preparing data...")
    # Load initial data
    X = load_data(X_path)
    Z = load_data(Z_path)
    
    # Clean up some data
    X, Y = prepare_data(X, True)
    Z = prepare_data(Z)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, stratify=Y)

    # Scale and reduce the data
    X_train, X_test, Z = scale_and_reduce(X_train, X_test, Z, variance)

    # Return data
    return X_train, X_test, Y_train, Y_test, Z

def get_model_stats(Y_values, Y_true, Y_pred):
    """
    Calculates and prints the model statistics for the possible values,
    the true values, and the predicted values.

    Args:
        Y_values (array): Array of possible labels
        Y_true (array): Array of correct labels
        Y_pred (array): Array of predicted labels

    Returns:
        balanced_accuracy (float): Balanced accuracy score, domain [0, 1]
        adj_balanced_accuracy (float): Adjusted for randomness, 0 = no better than random
        precision ([float]): Array of precision values for each label
        recall ([float]): Array of recall values for each label
    """
    # Validation Statistics
    balanced_accuracy = balanced_accuracy_score(Y_true, Y_pred)
    adj_balanced_accuracy = balanced_accuracy_score(Y_true, Y_pred, adjusted=True)
    precision = precision_score(Y_true, Y_pred, average=None, labels=Y_values)
    recall = recall_score(Y_true, Y_pred, average=None, labels=Y_values)
    print(f'Model Statistics:\nBalanced Accuracy = {balanced_accuracy}\nAdjusted Balanced Accuracy = {adj_balanced_accuracy}')
    print('Precision:')
    for i in range(len(Y_values)):
        print(f'\t{Y_values[i]}: {precision[i]}')

    print('Recall:')
    for i in range(len(Y_values)):
        print(f'\t{Y_values[i]}: {recall[i]}')
    return balanced_accuracy, adj_balanced_accuracy, precision, recall

def train_predict_custom(X_train, X_test, Y_train, Y_test, Z):
    """
    Trains, tests, and saves the output of the kaggle test file to a csv file.

    Args:
        X_train (dataframe): Samples of training data
        X_test (dataframe): Samples of validation testing data
        Y_train (array): Correct labels for the training data
        Y_test (array): Correct labels for the validation data
        Z (dataframe): Samples of the kaggle test file

    Returns:
        None
    """
    # Create model
    model = LogisticRegression(learning_rate=learning_rate, penalty=penalty, epsilon=epsilon, max_iterations=max_iterations)

    start = time.time()
    # Train the model
    model.fit(X_train, Y_train)

    # Validation testing of the model
    Y_pred = model.predict(X_test)

    # Validation Statistics
    Y_values = np.unique(Y_train)
    print(f'Stats for Custom Logistic Regression({time.time() - start} s)')
    get_model_stats(Y_values, Y_test, Y_pred)

    # Run model of actual test data
    out_pred = model.predict(Z)
    Z['class'] = out_pred

    # Save the output data to a file
    # Get current datetime
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H-%M")
    Z['class'].to_csv(save_path + dt_string + '.csv', index_label='id')

def train_predict_other(X_train, X_test, Y_train, Y_test):
    """
    Trains and tests on several Sklearn ML models for result comparrison.

    Args:
        X_train (dataframe): Samples of training data
        X_test (dataframe): Samples of validation testing data
        Y_train (array): Array of correct labels for training data
        Y_test (array): Array of correct labels for validation testing data

    Returns:
        None
    """
    Y_values = np.unique(Y_train)

    # Sklearn logistic regression
    logreg = lr(penalty='l1', max_iter=max_iterations, tol=epsilon, solver='liblinear')
    start = time.time()
    logreg.fit(X_train, Y_train)
    lr_pred = logreg.predict(X_test)
    print(f'Stats for Sklearn Logistic Regression({time.time() - start} s):')
    get_model_stats(Y_values, Y_test, lr_pred)

    # Sklearn Random Forest
    rf = RandomForestClassifier(criterion='entropy', max_depth=50)
    start = time.time()
    rf.fit(X_train, Y_train)
    rf_pred = rf.predict(X_test)
    print(f'Stats for Sklearn Random Forest({time.time() - start} s):')
    get_model_stats(Y_values, Y_test, rf_pred)

    # Sklearn Gaussian Naive Bayes
    gnb = GaussianNB()
    start = time.time()
    gnb.fit(X_train, Y_train)
    gnb_pred = gnb.predict(X_test)
    print(f'Stats for Sklearn Gaussian Naive Bayes({time.time() - start} s):')
    get_model_stats(Y_values, Y_test, gnb_pred)

    # Sklearn Gradient Boosting
    gbc = GradientBoostingClassifier(max_depth=50)
    start = time.time()
    gbc.fit(X_train, Y_train)
    gbc_pred = gbc.predict(X_test)
    print(f'Stats for Sklearn Gradient Boosting({time.time() - start} s):')
    get_model_stats(Y_values, Y_test, gbc_pred)

    # Sklearn SVM
    svc = svm.SVC()
    start = time.time()
    svc.fit(X_train, Y_train)
    svc_pred = svc.predict(X_test)
    print(f'Stats for Sklearn SVM({time.time() - start} s):')
    get_model_stats(Y_values, Y_test, svc_pred)

if __name__ == "__main__":
    print_params()
    # Load and prepare the data
    X_train, X_test, Y_train, Y_test, Z = load_prep_scale(train_path, test_path, split_size, variance)

    # Test our model
    train_predict_custom(X_train, X_test, Y_train, Y_test, Z)

    # Test with other models
    train_predict_other(X_train, X_test, Y_train, Y_test)