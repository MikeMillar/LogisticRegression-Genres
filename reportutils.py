import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt

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