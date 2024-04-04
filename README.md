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

### Configuration
This application has several configuration options which are outlined below. \
TODO

### Starting the application
TODO

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
    - `max_interations`: The maximum training iterations before termination.
- `utils.py`: This file contains useful utility functions that are used throughout the program.