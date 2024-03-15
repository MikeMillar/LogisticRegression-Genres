# Import required libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# Import OS to read files from directories
import os
# Imports for Audio Processing
import librosa

# Path to training data
train_dir = 'data/train/'
# Path to testing data
test_dir = 'data/test/'
# Path to single audio file for testing
test_path = 'data/train/blues/blues.00000.au'

# Configuration Variables
test = True   # If test is set to true, run on single audio file described above
hop_size = 512 # Step size of the audio, 512 ~= 23ms
mfcc_count = 13 # Total number of MFCC's to return



def get_audio_filenames(dir):
    """
    Parses a directory and it's subdirectories for all audio files and
    adds their relative path to an array.

    Args:
        dir (string): The directory to start fetching audio files from.

    Returns:
        [string]: Array of string file paths of matching audio files.
    """

    # Get all files in the directory
    dir_list = os.listdir(dir)
    # Initialize list
    filenames = []
    # Loop through all files
    for filename in dir_list:
        filepath = dir + filename
        # If file is a directory, need to recurse
        if os.path.isdir(filepath):
            ret = get_audio_filenames(filepath + '/')
            filenames += ret
            continue
        # Only care about files that end in au
        if filename.endswith('.au'):
            filenames.append(filepath)
    # Return filenames
    return filenames



def extract_labels(filenames):
    """
    Parses the relative path filenames for the genre each audio
    file is of. Appends to a list of genres.

    Args:
        filenames ([string]): Array of string file paths to audio files

    Returns:
        [string]: Array of strings representing the labeled genre of each file.
    """

    labels = []
    for filename in filenames:
        sub_filename = filename[11:]
        slash_index = sub_filename.find('/')
        if slash_index > 0:
            labels.append(sub_filename[:slash_index])
    return labels



def load_audio_files(filepaths):
    """
    For each audio file path, usings Librosa's load function to
    extract the waveform and sample rate of the audio file. 
    Combines all the audio waveforms and sample rates into
    two arrays and returns them.

    Args:
        filepaths ([string]): Array of string filepaths to load

    Returns:
        np.ndarray: audio time series
        np.ndarray: audio sample rates
    """

    waveforms = []
    sample_rates = []
    for path in filepaths:
        y, sr = librosa.load(path)
        waveforms.append(y)
        sample_rates.append(sr)
    return waveforms, sample_rates



def extract_beat(waveforms, sample_rates):
    """
    Analyzes a waveform and sample rate series to produce the 
    beats per minute and beat frames of the waveform.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates

    Returns:
        [float]: array of bpm measurements for each waveform
        np.ndarray: beat frames for each waveform
    """

    tempos = []
    beat_frame_sets = []
    for i in range(len(waveforms)):
        tempo, beat_frames = librosa.beat.beat_track(y=waveforms[i], sr=sample_rates[i])
        tempos.append(tempo)
        beat_frame_sets.append(beat_frames)
    return tempos, beat_frame_sets



def extract_mfcc(waveforms, sample_rates):
    """
    Analyzes the Mel Frequency Cepstral Coefficients (MFCC) for each
    waveform. MFCC produces a matrix of coefficients which we 
    aggregate using a number of statistics and reduce to several
    aggregated statistics. Use PCA to reduce the MFCC matrix down
    to single dimensional vectors.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates

    Returns:
        [float]: array of MFCC values summed.
        [float]: array of MFCC values means.
        [float]: array of MFCC standard deviations.
        [float]: array of MFCC minimum values.
        [float]: array of MFCC maximum values.
        [np.array]: array of MFCC matrices reduced to 1 dimensional vectors
    """
    
    mfcc_sums = []
    mfcc_means = []
    mfcc_stds = []
    mfcc_mins = []
    mfcc_maxs = []
    mfcc_vectors = []
    # Compute MFCC for all waveforms
    for i in range(len(waveforms)):
        # Compute MFCC matrix
        mfcc_matrix = librosa.feature.mfcc(y=waveforms, sr=sample_rates, hop_length=hop_size, n_mfcc=mfcc_count)
        # Flatten matrix to obtain aggregate statistics
        mfcc_flatten = mfcc_matrix.flatten()
        mfcc_sums.append(np.sum(mfcc_flatten))
        mfcc_means.append(np.mean(mfcc_flatten))
        mfcc_stds.append(np.std(mfcc_flatten))
        mfcc_mins.append(np.min(mfcc_flatten))
        mfcc_maxs.append(np.max(mfcc_flatten))
        # Perform PCA to reduce MFCC matrix to 1 dimension
        pca = PCA(n_components=1)
        mfcc_vectors.append(pca.fit_transform(mfcc_matrix).flatten())
    return mfcc_sums, mfcc_means, mfcc_stds, mfcc_mins, mfcc_maxs, mfcc_vectors



def extract_spectral_contrast(waveforms, sample_rates):
    """
    Analyzes the Spectral Constrast (SC) for each waveform.
    Spectral contrast produces a matrix of coefficients which we 
    aggregate using a number of statistics and reduce to several
    aggregated statistics. We use PCA to reduce the SC matrix down
    to 1 dimensional vectors which maintain the highest information
    possible.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates

    Returns:
        [float]: array of SC values summed.
        [float]: array of SC values means.
        [float]: array of SC standard deviations.
        [float]: array of SC minimum values.
        [float]: array of SC maximum values.
        [np.array]: array of specral contrast matrices reduced to 1 dimensional vectors.
    """

    sc_sums = []
    sc_means = []
    sc_stds = []
    sc_mins = []
    sc_maxs = []
    sc_vectors = []
    # Compute Spectral Contrast for all waveforms
    for i in range(len(waveforms)):
        # Compute Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size)
        # Flatten to aggregate statistics
        contrast_flat = contrast.flatten()
        sc_sums.append(np.sum(contrast_flat))
        sc_means.append(np.mean(contrast_flat))
        sc_stds.append(np.std(contrast_flat))
        sc_mins.append(np.min(contrast_flat))
        sc_maxs.append(np.max(contrast_flat))
        # Perform PCA to reduce SC to 1 dimension
        pca = PCA(n_components=1)
        sc_vectors.append(pca.fit_transform(contrast).flatten())
    # Return all values
    return sc_sums, sc_means, sc_stds, sc_mins, sc_maxs, sc_vectors



if __name__ == '__main__':
    # Check if testing mode is on
    if test:
        # load the audio file
        y, sr = librosa.load(test_path)
        # get tempo (bpm)
        tempo, beat_frames = extract_beat([y], [sr])
        # Get aggregate MFCC features
        # mfcc_sums, mfcc_means, mfcc_stds, mfcc_mins, mfcc_maxs = extract_mfcc([y], [sr])
        extract_mfcc(y, sr)
        extract_spectral_contrast(y, sr)
    else:
        filenames = get_audio_filenames(train_dir)
        labels = extract_labels(filenames)
        waveforms, sample_rates = load_audio_files(filenames)
        tempos, beat_frames = extract_beat(waveforms, sample_rates)
        mfcc_sums, mfcc_means, mfcc_stds, mfcc_mins, mfcc_maxs = extract_mfcc(waveforms, sample_rates)
        sc_sums, sc_means, sc_stds, sc_mins, sc_maxs = extract_spectral_contrast(waveforms, sample_rates)
        audio_data = {
            'file': filenames,
            'bpm': tempos,
            'mfcc_sum': mfcc_sums,
            'mfcc_mean': mfcc_means,
            'mfcc_std': mfcc_stds,
            'mfcc_min': mfcc_mins,
            'mfcc_max': mfcc_maxs,
            'sc_sum': sc_sums,
            'sc_mean': sc_means,
            'sc_std': sc_stds,
            'sc_min': sc_mins,
            'sc_max': sc_maxs,
            'label': labels
        }
        df = pd.DataFrame(audio_data)
        print(df)
    
