import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.stats as st

def tempo_demo(y, sr):
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
    plt.figure(figsize=(8,4))
    librosa.display.waveshow(y, sr=sr, color='blue', alpha=0.4)
    plt.vlines(beat_times, -1, 1, colors='r')
    plt.xlim(0,25)
    plt.ylim(-1,1)
    plt.title('Beat Times')
    plt.tight_layout()
    plt.show()

def centroid_demo(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_norm = (cent - cent.min()) / (cent.max() - cent.min())
    frames = range(len(cent))
    t = librosa.frames_to_time(frames)
    plt.figure(figsize=(8,4))
    librosa.display.waveshow(y, sr=sr, alpha=0.4, color='b')
    plt.plot(t, cent_norm, color='r')
    plt.xlim(0,30)
    plt.ylim(-1,1)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Spectral Centroid')
    plt.title('Spectral Centroid')
    plt.tight_layout()
    plt.show()

def rolloff_demo(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_norm = (rolloff - rolloff.min()) / (rolloff.max() - rolloff.min())
    frames = range(len(rolloff))
    t = librosa.frames_to_time(frames)
    plt.figure(figsize=(8,4))
    librosa.display.waveshow(y, sr=sr, alpha=0.4, color='b')
    plt.plot(t, rolloff_norm, color='r')
    plt.xlim(0,30)
    plt.ylim(-1,1)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Spectral Rolloff')
    plt.title('Spectral Rolloff')
    plt.tight_layout()
    plt.show()

def flatness_demo(y, sr):
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flatness_norm = (flatness - flatness.min()) / (flatness.max() - flatness.min())
    
    frames = range(len(flatness))
    t = librosa.frames_to_time(frames)

    plt.figure(figsize=(8,4))
    librosa.display.waveshow(y, sr=sr, alpha=0.4, color='b')
    plt.plot(t, flatness_norm, color='r')
    plt.xlim(0, 30)
    plt.ylim(-1,1)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Spectral Flatness')
    plt.title('Spectral Flatness')
    plt.tight_layout()
    plt.show()

def zcr_demo(y, sr):
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min())
    frames = range(len(zcr))
    t = librosa.frames_to_time(frames)
    plt.figure(figsize=(8,4))
    librosa.display.waveshow(y, sr=sr, alpha=0.4, color='b')
    plt.plot(t, zcr_norm, color='r')
    plt.xlim(0, 30)
    plt.ylim(-1,1)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Zero Crossing Rate')
    plt.title("Zero Crossing Rate")
    plt.tight_layout()
    plt.show()

def mfcc_demo(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=8000)
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                y_axis='mel', x_axis='time', fmax=8000, ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.show()

def contrast_demo(y, sr):
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8,8))
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax[0])
    fig.colorbar(img, ax=[ax[0]], format='%+2.0f dB')
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(ylabel='Frequency bands', title='Spectral Contrast')
    plt.show()

def pitch_chroma_demo(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.xlabel('Time (s)')
    plt.title('Chromagram')
    plt.tight_layout()
    plt.show()

def random_search_demo(df: pd.DataFrame):

    # Find paramters that perform best for each metric
    best_balanced = df.iloc[np.argmax(df['bal_acc'], axis=0)]
    best_precision = df.iloc[np.argmax(df['precision'], axis=0)]
    best_recall = df.iloc[np.argmax(df['recall'], axis=0)]
    best_f1 = df.iloc[np.argmax(df['f1'], axis=0)]
    print('Best Balanced Accuracy:\n', best_balanced)
    print('Best Precision:\n', best_precision)
    print('Best Recall:\n', best_recall)
    print('Best F1:\n', best_f1)    

    # 3D graph by balanced accuracy
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['learning_rate'], df['tolerance'], df['penalty'], c=df['bal_acc'])
    ax.set_xlabel('Learning Rate (eta)')
    ax.set_ylabel('Tolerance (epsilon)')
    ax.set_zlabel('Regularization Penalty')
    plt.title('Balanced Accuracy')
    fig.colorbar(scatter, ax=ax)
    plt.show()

    # 3D graph by precision
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['learning_rate'], df['tolerance'], df['penalty'], c=df['precision'])
    ax.set_xlabel('Learning Rate (eta)')
    ax.set_ylabel('Tolerance (epsilon)')
    ax.set_zlabel('Regularization Penalty')
    plt.title('Precision')
    fig.colorbar(scatter, ax=ax)
    plt.show()

    # 3D graph by recall
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['learning_rate'], df['tolerance'], df['penalty'], c=df['recall'])
    ax.set_xlabel('Learning Rate (eta)')
    ax.set_ylabel('Tolerance (epsilon)')
    ax.set_zlabel('Regularization Penalty')
    plt.title('Recall')
    fig.colorbar(scatter, ax=ax)
    plt.show()

    # 3D graph by f1 score
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['learning_rate'], df['tolerance'], df['penalty'], c=df['f1'])
    ax.set_xlabel('Learning Rate (eta)')
    ax.set_ylabel('Tolerance (epsilon)')
    ax.set_zlabel('Regularization Penalty')
    plt.title('F1 Score')
    fig.colorbar(scatter, ax=ax)
    plt.show()

def model_comparison():
    # Load the data
    df = pd.read_csv('data/result/_scores2.csv')
    # Group by model
    gfs = df.groupby('model')
    # Seperate groups
    my_lr = gfs.get_group('My_LR')
    sk_lr = gfs.get_group('SK_LR')
    sk_rf = gfs.get_group('SK_RF')
    sk_gnb = gfs.get_group('SK_GNB')
    sk_gb = gfs.get_group('SK_GB')
    sk_svm = gfs.get_group('SK_SVM')

    names = ['My_LR', 'SK_LR', 'SK_RF', 'SK_GNB', 'SK_GB', 'SK_SVM']
    # model,bal_acc,adj_bal_acc,precision,recall,f1_score
    
    # Balanced accuracy stats
    # intervals
    my_lr_interval = st.t.interval(confidence=0.95, df=len(my_lr)-1, loc=np.mean(my_lr['bal_acc']), scale=st.sem(my_lr['bal_acc']))
    sk_lr_interval = st.t.interval(confidence=0.95, df=len(sk_lr)-1, loc=np.mean(sk_lr['bal_acc']), scale=st.sem(sk_lr['bal_acc']))
    sk_rf_interval = st.t.interval(confidence=0.95, df=len(sk_rf)-1, loc=np.mean(sk_rf['bal_acc']), scale=st.sem(sk_rf['bal_acc']))
    sk_gnb_interval = st.t.interval(confidence=0.95, df=len(sk_gnb)-1, loc=np.mean(sk_gnb['bal_acc']), scale=st.sem(sk_gnb['bal_acc']))
    sk_gb_interval = st.t.interval(confidence=0.95, df=len(sk_gb)-1, loc=np.mean(sk_gb['bal_acc']), scale=st.sem(sk_gb['bal_acc']))
    sk_svm_interval = st.t.interval(confidence=0.95, df=len(sk_svm)-1, loc=np.mean(sk_svm['bal_acc']), scale=st.sem(sk_svm['bal_acc']))
    # interval mins-maxs
    int_mins = [my_lr_interval[0], sk_lr_interval[0], sk_rf_interval[0], sk_gnb_interval[0], sk_gb_interval[0], sk_svm_interval[0]]
    int_maxs = [my_lr_interval[1], sk_lr_interval[1], sk_rf_interval[1], sk_gnb_interval[1], sk_gb_interval[1], sk_svm_interval[1]]
    # value min-max
    mins = [np.min(my_lr['bal_acc']), np.min(sk_lr['bal_acc']), np.min(sk_rf['bal_acc']), np.min(sk_gnb['bal_acc']), np.min(sk_gb['bal_acc']), np.min(sk_svm['bal_acc'])]
    maxs = [np.max(my_lr['bal_acc']), np.max(sk_lr['bal_acc']), np.max(sk_rf['bal_acc']), np.max(sk_gnb['bal_acc']), np.max(sk_gb['bal_acc']), np.max(sk_svm['bal_acc'])]
    print('Statistics for _scores2.csv:')
    print(f'My_LR: interval={my_lr_interval}, min={mins[0]}, max={maxs[0]}')
    print(f'SK_LR: interval={sk_lr_interval}, min={mins[1]}, max={maxs[1]}')
    print(f'SK_RF: interval={sk_rf_interval}, min={mins[2]}, max={maxs[2]}')
    print(f'SK_GNB: interval={sk_gnb_interval}, min={mins[3]}, max={maxs[3]}')
    print(f'SK_GB: interval={sk_gb_interval}, min={mins[4]}, max={maxs[4]}')
    print(f'SK_SVM: interval={sk_svm_interval}, min={mins[5]}, max={maxs[5]}')
    # create data
    ba_df = pd.DataFrame({'open': int_mins,
                          'close': int_maxs,
                          'high': maxs,
                          'low': mins},
                          index=names)
    
    plt.figure(figsize=(6,6))
    plt.bar(ba_df.index, ba_df['close']-ba_df['open'], width=0.3, bottom=ba_df['open'], color='b')
    plt.bar(ba_df.index, ba_df['high']-ba_df['close'], width=0.03, bottom=ba_df['close'], color='b')
    plt.bar(ba_df.index, ba_df['low']-ba_df['open'], width=0.03, bottom=ba_df['open'], color='b')
    plt.xticks(rotation=30, ha='right')
    plt.xlabel('Model')
    plt.ylabel('Balanced Accuracy')
    plt.title('ML Model Comparision for Genre Classification')
    plt.show()


if __name__ == '__main__':
    # Load the demo song
    test_path = 'data/train/blues/blues.00000.au'
    y, sr = librosa.load(test_path)

    # Demo tempo
    tempo_demo(y, sr)

    # Demo Spectral Centroid
    centroid_demo(y, sr)

    # Demo Spectral Rolloff
    rolloff_demo(y, sr)

    # Demo Spectral Flatness
    flatness_demo(y, sr)

    # Demo Zero Crossing Rate
    zcr_demo(y, sr)

    # Demo MFCC
    mfcc_demo(y, sr)

    # Demo Spectral Contrast
    contrast_demo(y, sr)

    # Demo Pitch Chroma
    pitch_chroma_demo(y, sr)

    # model comparison
    model_comparison()