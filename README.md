# Project 2 - Music Classification
In this project we will be classifying audio files into several different genres of music. We are given audio files for training which are sorted by genre already. We will use these audio files to extract features from them. These features will be used to classify the test audio files by genre based on a Logistic Regression Model we implement from scratch. This model will also utilize a gradient descent algorithm which will also be written from scratch. 

# Project Members
- Michael Millar: Sole developer

# Running the Application
The next few sections will step through the setup, configuration options, to actually running the application.

### Setup
1. Have python and pip installed on your computer.
2. Setup python virtual environment: \
    2a. Run command `python -m venv .venv` \
    2b. Activate virtual environment with `source .venv/bin/activate`
3. Use pip to install required libraries with `pip install -r requirements.txt`

### Configuration and Running the Application
The configuration of this project is done by modify variables and certain lines of code within the python files themselves. If you are interested in extracting features from music files, see `extract.py` in the file manifest. If you already have extracted feature data and want to run the model(s), see `music.py` in the file manifest. All information about running the application is defined in the file manifest for both `extract.py` and `music.py`.

# File Manifest
This project contains many parts which can be run independantly, or together. The following manifest will outline the different files in the application. This will include detailed descriptions of each file, which will detail their functions and use cases.

- `extract.py`: This file is used for the preprocessing and extraction of features in the audio files. The specified features and extracted and saved into an output file for later use. This file requires manually commenting/uncommenting of code, to determine which features are extracted. It has some additional parameters which can be modified:
    - train_dir: Directory of the training data.
    - test_dir: Directory of the testing data.
    - test: Boolean to indicate testing of the extract functions, only runs on a single file specified in the main method.
    - flatten: Boolean to indicate if the vector/matrices should be flattened in the output dataframe/csv
    - hop_size: Hop sized use by librosa during audio extraction.
    - mfcc_count: How many MFCC coefficients to be extracted (13-20)

    To use `extract.py`, follow the steps below:
    1. Decide which features you want to extract and uncomment the features you want, and comment the features you don't want.
    2. Update either the train_dir or test_dir directory paths, then in the main method update which one is used. 
    3. Rename the output file at the bottom of main.
    4. Run the program with `python extract.py`
    5. Let program run, then check the output file.
- `logreg.py`: This file contains the functions for the implementation of the logistic regression model. Code from this file is run from the `music.py` file, but has 3 may ways of interacting with.
    1. Initialization and hyper-parameter setup. `lr = LogisticRegression()`. You can use the default parameter values defined in the `__init__` method, or pass them as arguments.
    2. Training of the model using the fit method. `lr.fit(X, Y)`
    3. Predicting classifications with the predict method. `pred = lr.predict(X)`

    It has several hyper-parameters that you can specify:
    - `learning_rate`: The rate at which the model progresses towards convergence during training.
    - `epsilon`: The error difference before early training termination.
    - `penalty`: The regularization pentaly used during training.
    - `max_iterations`: The maximum training iterations before termination.
- `utils.py`: This file contains a handful of useful utility functions that are used throughout the program.
    - `matrix_to_columns(matrix, label)`: Takes an numpy array matrix and turns it into a dictionary of columns which can be loaded into a pandas dataframe.
    - `trim_matrices(container, flat)`: Takes a container of matrices and trims them to be all have the same width. If the container instead contians vectors or arrays, then flatten should be set to True.
    - `get_filename(path)`: Extracts just the filename from a full path string.
- `reportutils.py`: This is a file that was used to collect and graph statistics from various data collected during the development and testing of the model. You can comment/uncomment which methods you want to see run. Some methods may need you specify directory/file paths.
- `music.py`: This is the main script of the program. It is used to run model or models for comparison. It has a variety of parameters that can be specified, and some code may need to be commented in/out depending on what portions of the application you want to run. The variables to modify are:
    - train_path: Path to the file of training and cross-validation data CSV file.
    - test_path: Path to the kaggle competition data CSV file. This file does not have label data.
    - save_path: Path to the directory that you want the output files to be saved in.
    - split_size: The percent size of the validation set from training data.
    - variance: The percent variance to keep during PCA reduction.
    - save_output: Boolean indicating if the application should produce an output file of predictions from the test data.
    - reduce: Boolean indicating if PCA reduction should be performed.
    - learning_rate: Learning rate of the logistic regression model
    - epsilon: Error difference value for early termination.
    - penalty: Regularization penalty used during training.
    - max_iterations: Max iterations that can occur during training.
    - total_runs: The total number of times to run the defined models.
    - score_file: The path and filename of a CSV file to save the results of the models.

    To run the file:
    1. Set the variables to your data path and to your liking.
    2. In the main method, define the names of the methods you want to run. There are 6 names pre-defined (My_LR, SK_LR, SK_RF, SK_GNB, SK_GB, SK_SVM).
    3. Below where you defined the names, initialize the models in the `models` array. Optionally you can specify the variables of the sklearn models or leave them as they are already defined.
    4. Caveats with the score file:
        - If you do not want the score file to be saved, comment out the lines at the bottom of main that `open`, `write` and `close` the file. The `get_model_stats` method will print the results of the models to the console either way.
        - If the score file doesn't already exist with a defined header, the created score file will be missing a header. You can edit the score file and add a header as the first line, which should be the following string: `model,bal_acc,adj_bal_acc,precision,recall,f1_score`.
        - The score file is what is used in the `model_comparison` method in `reportutils.py`.
    5. In a console run using `python music.py`
- `graphs/`: Directory containing all of the statistical graphs used during the evaluation and reporting of this project.