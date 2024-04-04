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
- `logreg.py`: This file contains the functions for the implementation of the logistic regression model.
- `utils.py`: This file contains useful utility functions that are used throughout the program.