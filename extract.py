# Import required libraries
import numpy as np
import pandas as pd
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
test = False   # If test is set to true, run on single audio file described above
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
    aggregated statistics. These statistics include sum, mean,
    standard deviation, min, and max values.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates

    Returns:
        [float]: array of MFCC values summed.
        [float]: array of MFCC values means.
        [float]: array of MFCC standard deviations.
        [float]: array of MFCC minimum values.
        [float]: array of MFCC maximum values.
    """

    # initialize return arrays
    mfcc_sums = []
    mfcc_means = []
    mfcc_stds = []
    mfcc_mins = []
    mfcc_maxs = []
    # Loop through all waveforms
    for i in range(len(waveforms)):
        # Get the MFCCs
        mfcc = librosa.feature.mfcc(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size, n_mfcc=mfcc_count)
        # initialize per file arrays
        sums = []
        means = []
        stds = []
        mins = []
        maxs = []
        # aggregate statistics of each MFCC window
        for j in range(len(mfcc)):
            sums.append(np.sum(mfcc[j]))
            means.append(np.mean(mfcc[j]))
            stds.append(np.std(mfcc[j]))
            mins.append(np.min(mfcc[j]))
            maxs.append(np.max(mfcc[j]))
        # aggreate statistics of all MFCC windows
        mfcc_sums.append(np.mean(sums))
        mfcc_means.append(np.mean(means))
        mfcc_stds.append(np.mean(stds))
        mfcc_mins.append(np.min(mins))
        mfcc_maxs.append(np.max(maxs))
    # return all mfcc aggregrated statistics
    return mfcc_sums, mfcc_means, mfcc_stds, mfcc_mins, mfcc_maxs



def extract_spectral_contrast(waveforms, sample_rates):
    """
    Analyzes the Spectral Constrast (SC) for each waveform.
    Spectral contrast produces a matrix of coefficients which we 
    aggregate using a number of statistics and reduce to several
    aggregated statistics. These statistics include sum, mean,
    standard deviation, min, and max values.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates

    Returns:
        [float]: array of SC values summed.
        [float]: array of SC values means.
        [float]: array of SC standard deviations.
        [float]: array of SC minimum values.
        [float]: array of SC maximum values.
    """

    # initialize return variables
    sc_sums = []
    sc_means = []
    sc_stds = []
    sc_mins = []
    sc_maxs = []
    # loop over all wave forms
    for i in range(len(waveforms)):
        # compute spectral contrast
        contrasts = librosa.feature.spectral_contrast(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size)
        # initialize aggregator variables
        sums = []
        means = []
        stds = []
        mins = []
        maxs = []
        # Loop over constract matrix to get aggregate stats
        for j in range(len(contrasts)):
            sums.append(np.sum(contrasts[j]))
            means.append(np.mean(contrasts[j]))
            stds.append(np.std(contrasts[j]))
            mins.append(np.min(contrasts[j]))
            maxs.append(np.max(contrasts[j]))
        # accumulate results
        sc_sums.append(np.mean(sums))
        sc_means.append(np.mean(means))
        sc_stds.append(np.mean(stds))
        sc_mins.append(np.min(mins))
        sc_maxs.append(np.max(maxs))
    # return spectral contrast stats
    return sc_sums, sc_means, sc_stds, sc_mins, sc_maxs



if __name__ == '__main__':
    # Check if testing mode is on
    if test:
        # load the audio file
        y, sr = librosa.load(test_path)
        # get tempo (bpm)
        tempo, beat_frames = extract_beat([y], [sr])
        # Get aggregate MFCC features
        # mfcc_sums, mfcc_means, mfcc_stds, mfcc_mins, mfcc_maxs = extract_mfcc([y], [sr])
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
    
