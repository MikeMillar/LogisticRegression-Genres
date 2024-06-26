# Import required libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# Import OS to read files from directories
import os
# Imports for Audio Processing
import librosa
# Import project files
import utils

# Path to training data
train_dir = 'data/train/'
# Path to testing data
test_dir = 'data/test/'
# Path to single audio file for testing
test_path = 'data/train/blues/blues.00000.au'

# Configuration Variables
test = False   # If test is set to true, run on single audio file described above
flatten = True # True if we want to flatten our matrices
hop_size = 512 # Step size of the audio, 512 ~= 23ms
mfcc_count = 20 # Total number of MFCC's to return




def get_audio_filenames(dir):
    """
    Parses a directory and it's subdirectories for all audio files and
    adds their relative path to an array.

    Args:
        dir (string): The directory to start fetching audio files from.

    Returns:
        [string]: Array of string file paths of matching audio files.
    """

    print('Fetching audio file paths...')
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

    print('Fetching file labels...')
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

    print('Loading audio files...')
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

    print('Extracting tempo...')
    tempos = []
    beat_frame_sets = []
    for i in range(len(waveforms)):
        tempo, beat_frames = librosa.beat.beat_track(y=waveforms[i], sr=sample_rates[i])
        tempos.append(tempo)
        beat_frame_sets.append(beat_frames)
    return tempos, beat_frame_sets



def extract_mfcc(waveforms, sample_rates, pool_axis):
    """
    Analyzes the Mel Frequency Cepstral Coefficients (MFCC) for each
    waveform. MFCC produces a matrix of shape (mfcc_count,x), where
    x depends on the hop length. Each MFCC matrix is mean pooled on
    the columns. Function returns a dictionary of all the MFCC
    coefficient mean pooled columns for each waveform.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates
        pool_axis (0 or 1): 0 for column pooling, 1 for row pooling

    Returns:
        (dict): Dictionary of mean pooled MFCC data
    """
    print('Extracting MFCCs...')
    mfcc_matrices = []
    mfcc_vectors = []
    # Compute MFCC for all waveforms
    for i in range(len(waveforms)):
        # Compute MFCC matrix
        mfcc_matrix = librosa.feature.mfcc(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size, n_mfcc=mfcc_count)
        if flatten:
            mfcc_matrix = mfcc_matrix.flatten('F')
        mfcc_matrices.append(mfcc_matrix)
    # Ensure all matrices are of same size (truncating wider matrices)
    mfcc_trimmed_matrices = utils.trim_matrices(mfcc_matrices, flatten)
    if flatten:
        return utils.matrix_to_columns(np.array(mfcc_trimmed_matrices), 'mfcc')
    for mfcc in mfcc_trimmed_matrices:
        # Mean pool MFCC matrix
        mfcc_mp = np.mean(mfcc, axis=pool_axis)
        # Add to vectors
        mfcc_vectors.append(mfcc_mp)
    # Convert to column dictionary and return
    return utils.matrix_to_columns(np.array(mfcc_vectors), 'mfcc')



def extract_spectral_contrast(waveforms, sample_rates, pool_axis):
    """
    Analyzes the Spectral Constrast (SC) for each waveform.
    Spectral contrast produces a matrix of coefficients which
    are mean pooled on the columns. Fucntion returns a 
    dictionary of all the SC coefficient mean pooled columns
    for each waveform.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates
        pool_axis (0 or 1): 0 for column pooling, 1 for row pooling

    Returns:
        (dict): Dictionary of mean pooled SC data
    """

    print("Extracting SCs...")
    sc_matrices = []
    sc_vectors = []
    # Compute Spectral Contrast for all waveforms
    for i in range(len(waveforms)):
        # Compute Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size)
        if flatten:
            contrast = contrast.flatten('F')
        sc_matrices.append(contrast)
    # Ensure all matrices are of same size (truncating wider matrices)
    sc_trimmed_matrices = utils.trim_matrices(sc_matrices)
    if flatten:
        return utils.matrix_to_columns(sc_trimmed_matrices, 'sc')
    for m in sc_trimmed_matrices:
        # Mean pooling on the columns of the SC
        contrast_mp = np.mean(m, axis=pool_axis)
        # Add to vectors
        sc_vectors.append(contrast_mp)
    # Convert to column dictionary and return
    return utils.matrix_to_columns(sc_vectors, 'sc')



def extract_spectral_centroid(waveforms, sample_rates):
    """
    Extracts the Spectral Centroid of each waveform, keeps the
    mean of the result.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates

    Returns:
        [float]: Returns a list of the means of each spectral
                 centroid for each waveform.
    """
    print("Extracting Spectral Centroids...")
    centroids = []
    for i in range(len(waveforms)):
        cent = librosa.feature.spectral_centroid(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size)
        if flatten:
            centroids.append(cent.flatten())
            continue
        centroids.append(np.mean(cent))
    if flatten:
        print(centroids)
        return utils.matrix_to_columns(np.array(centroids), 'cent')
    return centroids
    


def extract_spectral_rolloff(waveforms, sample_rates):
    """
    Extracts the Spectrall Rolloff of each waveform, keeps the
    mean of the result.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates

    Returns:
        [float]: Returns a list of the means of each spectral
                 rolloff for each waveform.
    """
    print("Extracting Spectral Rolloffs")
    rolloffs = []
    for i in range(len(waveforms)):
        rolloff = librosa.feature.spectral_rolloff(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size)
        if flatten:
            rolloffs.append(rolloff.flatten())
            continue
        rolloffs.append(np.mean(rolloff))
    if flatten:
        return utils.matrix_to_columns(np.array(rolloffs), 'rolloff')
    return rolloffs



def extract_pitch_chroma(waveforms, sample_rates, pool_axis):
    """
    Extracts the Pitch Chroma of each waveform, then reduces
    them to a single dimension using mean pooling on the columns.
    The result is stored in a dictionary where each key is a column
    of data.

    Args:
        waveforms (np.ndarray): audio time series
        sample_rates (np.ndarray): audio sample rates
        pool_axis (0 or 1): 0 for column pooling, 1 for row pooling

    Returns:
        (dict): Dictionary of mean pooled pitch chroma data
    """
    print("Extracting pitch chroma...")
    c_matrices = []
    c_vectors = []
    for i in range(len(waveforms)):
        chroma = librosa.feature.chroma_stft(y=waveforms[i], sr=sample_rates[i], hop_length=hop_size)
        if flatten:
            chroma = chroma.flatten('F')
        c_matrices.append(chroma)
    c_trimmed_matrices = utils.trim_matrices(c_matrices)
    if flatten:
        return utils.matrix_to_columns(c_trimmed_matrices, 'chroma_h')
    for m in c_trimmed_matrices:
        c_vectors.append(np.mean(m, axis=pool_axis))
    return utils.matrix_to_columns(c_vectors, 'chroma_h')



def extract_zero_crossing_rate(waveforms):
    """
    Extracts the Zero Crossing Rate (ZCR) for each
    waveform and computes the mean of the vector.
    
    Args:
        waveforms (np.ndarray): audio time series
        
    Returns:
        ([float]): A list of means of the ZCR data of
                   each waveform.
    """
    zcrs = []
    for i in range(len(waveforms)):
        zcr = librosa.feature.zero_crossing_rate(y=waveforms[i], hop_length=hop_size)
        if flatten:
            zcr.append(zcr.flatten())
            continue
        zcrs.append(np.mean(zcr))
    if flatten:
        return utils.matrix_to_columns(np.array(zcrs), 'zcr')
    return zcrs



def extract_spectral_flatness(waveforms):
    """
    Extracts the Spectral Flatness for each waveform
    and computes the mean of the vector.
    
    Args:
        waveforms (np.ndarray): audio time series

    Returns:
        ([float]): A list of means of the spectral
                   flatness of each waveform.
    """
    s_flats = []
    for i in range(len(waveforms)):
        flat = librosa.feature.spectral_flatness(y=waveforms[i], hop_length=hop_size)
        if flatten:
            s_flats.append(flat.flatten())
            continue
        s_flats.append(np.mean(flat))
    if flatten:
        return utils.matrix_to_columns(np.array(s_flats), 's_flat')
    return s_flats



if __name__ == '__main__':
    # Check if testing mode is on
    if test:
        # load the audio file
        y, sr = librosa.load(test_path)
        # get tempo (bpm)
        # tempo, beat_frames = extract_beat([y], [sr])
        # Get aggregate MFCC features
        # mfcc_sums, mfcc_means, mfcc_stds, mfcc_mins, mfcc_maxs = extract_mfcc([y], [sr])
        # extract_mfcc(y, sr)
        # extract_spectral_contrast(y, sr)
        # extract_spectral_centroid(y, sr)
        extract_spectral_rolloff(y, sr)
    else:
        # initialize audio data for dataframe
        audio_data: dict = {}
        # Fetch all the audio files paths to process
        filenames = get_audio_filenames(train_dir)
        # Extract the labeles for each audio file
        # labels = extract_labels(filenames)
        
        # Load the audio files, extracting their waveforms and sample rates
        waveforms, sample_rates = load_audio_files(filenames)

        # Single Valued Features
        # Extract the bpm and beat frames of every audio file
        # tempos, beat_frames = extract_beat(waveforms, sample_rates)
        # Extract the mean spectral centroid of every audio file
        # s_centroids = extract_spectral_centroid(waveforms, sample_rates)
        # Extract the mean spectral rolloff of every audio file
        # s_rolloffs = extract_spectral_rolloff(waveforms, sample_rates)
        # Extract the mean zero corssing rate of every audio file
        # zcrs = extract_zero_crossing_rate(waveforms)
        # Extract the spectral flatness of every audio file
        # s_flats = extract_spectral_flatness(waveforms)

        # Large Dimensionality Features
        # Extract MFCC features
        mfcc_data = extract_mfcc(waveforms, sample_rates, 1)
        # Extract SC features
        # sc_data = extract_spectral_contrast(waveforms, sample_rates, 1)
        # Extract PC features
        # pc_data = extract_pitch_chroma(waveforms, sample_rates, 1)

        # Add data to our audio_data dictionary
        # audio_data['label'] = labels
        # audio_data['bpm'] = tempos
        # if flatten:
        #     audio_data = audio_data | s_centroids
        #     audio_data = audio_data | s_rolloffs
        #     audio_data = audio_data | zcrs
        #     audio_data = audio_data | s_flats
        # else:
        #     audio_data['s_centroid'] = s_centroids
        #     audio_data['s_rolloff'] = s_rolloffs
        #     audio_data['zcr'] = zcrs
        #     audio_data['s_flat'] = s_flats
        audio_data = audio_data | mfcc_data
        # audio_data = audio_data | sc_data
        # audio_data = audio_data | pc_data

        filecount = len(filenames)
        for key in audio_data.keys():
            size = len(audio_data[key])
            if not(size == filecount):
                print(f"key={key}, size={len(audio_data[key])}")
        
        # Load data into dataframe and save to file
        df = pd.DataFrame(audio_data, index=filenames)
        print(df.head())
        df.to_csv('data/train/music_mfcc_flat.csv')
    
