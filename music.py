# External imports
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
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
from reportutils import random_search_demo

# Global variables
# Data variables
train_path = 'data/train/music_20_mfcc.csv'        # Path to the training CSV
test_path = 'data/test/test_20_mfcc.csv'           # Path to the output CSV
save_path = 'data/result/'
split_size = 0.2                                # Testing validation split size percent
variance = 0.85                                 # Percent of variance to keep when reducing dimensionality
save_output = False                             # Indicates that the output should be saved to file
reduce = False                                   # Set to True if you want PCA dimensionality reduction
total_runs = 1                                  # Total number of times to run each defined model
score_file = 'data/result/_scores.csv'
# Gradient Ascent variables
learning_rate = 0.003705                        # Learning rate used for gradient ascent
penalty = 0.673836                              # Regularization factor used for gradient ascent
epsilon = 1e-5                                  # Early termination error difference used for gradient ascent
max_iterations = 10000                          # Max iterations before termination of gradient ascent

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
    if reduce:
        pca = PCA(n_components=variance)
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
    Y = X['label']
    # MFCCs
    mfcc_cols = []
    for col in X.columns:
        if col.startswith('mfcc'):
            mfcc_cols.append(col)
    X = X[mfcc_cols]
    X['label'] = Y
    Z = Z[mfcc_cols]
    
    # Clean up some data
    X, Y = prepare_data(X, True)
    Z = prepare_data(Z)

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, stratify=Y, random_state=None)

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
    precision = precision_score(Y_true, Y_pred, average='weighted', labels=Y_values, zero_division=0.0)
    recall = recall_score(Y_true, Y_pred, average='weighted', labels=Y_values)
    f1 = f1_score(Y_true, Y_pred, average='weighted', labels=Y_values)
    print(f'\tBalanced Accuracy = {balanced_accuracy}\n\tAdjusted Balanced Accuracy = {adj_balanced_accuracy}')
    print(f'\tPrecision = {precision}\n\trecall = {recall}\n\tf1 = {f1}')
    # print('Precision:')
    # for i in range(len(Y_values)):
    #     print(f'\t{Y_values[i]}: {precision[i]}')
    # print('Recall:')
    # for i in range(len(Y_values)):
    #     print(f'\t{Y_values[i]}: {recall[i]}')
    # print('F1 Score:')
    # for i in range(len(Y_values)):
    #     print(f'\t{Y_values[i]}: {f1[i]}')
    return balanced_accuracy, adj_balanced_accuracy, precision, recall, f1

def train_predict_custom(model, X_train, X_test, Y_train, Y_test, Z):
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

    start = time.time()
    # Train the model
    model.fit(X_train, Y_train)

    # Validation testing of the model
    Y_pred = model.predict(X_test)

    # Run model of actual test data
    out_pred = model.predict(Z)
    Z['class'] = out_pred

    # Save the output data to a file
    # Get current datetime
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H-%M")
    Z['class'].to_csv(save_path + dt_string + '.csv', index_label='id')

    # Validation Statistics
    Y_values = np.unique(Y_train)
    print(f'Stats for Custom Logistic Regression({time.time() - start} s)')
    return get_model_stats(Y_values, Y_test, Y_pred)

def train_predict_other(model, name, X_train, X_test, Y_train, Y_test):
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
    start = time.time()
    model.fit(X_train, Y_train)
    lr_pred = model.predict(X_test)
    print(f'Stats for {name}({time.time() - start} s):')
    return get_model_stats(Y_values, Y_test, lr_pred)

def run_models(names, models, X_train, X_test, Y_train, Y_test, Z):
    bal_accs = []
    adj_bal_accs = []
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(names)):
        name = names[i]
        model = models[i]
        if name == 'My LR' and save_output:
            ba, aba, p, r, f = train_predict_custom(model, X_train, X_test, Y_train, Y_test, Z)
            bal_accs.append(ba)
            adj_bal_accs.append(aba)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        else:
            ba, aba, p, r, f = train_predict_other(model, name, X_train, X_test, Y_train, Y_test)
            bal_accs.append(ba)
            adj_bal_accs.append(aba)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
    return bal_accs, adj_bal_accs, precisions, recalls, f1s

# def random_search(X_train, X_test, Y_train, Y_test, iters):
def random_search(iters):
    # X_train, X_test, Y_train, Y_test, Z = load_prep_scale(train_path, test_path, split_size, variance)
    # Randomized hyper parameters
    etas = []
    eps = []
    pens = []
    # Metrics
    bals = []
    adj_bals = []
    precisions = []
    recalls = []
    f1s = []
    for i in range(iters):
        X_train, X_test, Y_train, Y_test, Z = load_prep_scale(train_path, test_path, split_size, variance)
        eta = np.random.uniform(0,0.01)
        pen = np.random.uniform(0,1)
        ep = np.random.uniform(1e-15, 0.001)
        model = LogisticRegression(learning_rate=eta, penalty=pen, epsilon=ep)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        metrics = get_model_stats(np.unique(Y_train), Y_test, Y_pred)
        etas.append(eta)
        eps.append(ep)
        pens.append(pen)
        bals.append(metrics[0])
        adj_bals.append(metrics[1])
        precisions.append(metrics[2])
        recalls.append(metrics[3])
        f1s.append(metrics[4])
    return {
        'learning_rate': etas,
        'tolerance': eps,
        'penalty': pens,
        'bal_acc': bals,
        'adj_bal_acc': adj_bals,
        'precision': precisions,
        'recall': recalls,
        'f1': f1s
    }

if __name__ == "__main__":
    # results = random_search(10000)
    # df = pd.DataFrame(results)
    # random_search_demo(df)
    # exit()
    
    print_params()
    # Load and prepare the data
    # X_train, X_test, Y_train, Y_test, Z = load_prep_scale(train_path, test_path, split_size, variance)

    names = [
        'My_LR', 
        'SK_LR', 
        'SK_RF', 
        'SK_GNB', 
        'SK_GB', 
        'SK_SVM'
    ]
    models = [
        LogisticRegression(learning_rate=learning_rate, penalty=penalty, epsilon=epsilon, max_iterations=max_iterations),
        lr(penalty='l1', solver='liblinear'),
        RandomForestClassifier(criterion='entropy', max_depth=50),
        GaussianNB(),
        GradientBoostingClassifier(max_depth=50),
        svm.SVC()
    ]

    score_filename = score_file
    for _ in range(total_runs):
        X_train, X_test, Y_train, Y_test, Z = load_prep_scale(train_path, test_path, split_size, variance)
        metrics = run_models(names, models, X_train, X_test, Y_train, Y_test, Z)
        score_file = open(score_filename, 'a')
        for i in range(len(names)):
            line = f'{names[i]},{metrics[0][i]},{metrics[1][i]},{metrics[2][i]},{metrics[3][i]},{metrics[4][i]}\n'
            score_file.write(line)
        score_file.close()