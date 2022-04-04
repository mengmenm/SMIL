import numpy as np
import librosa
import os
from PIL import Image
# from keras.utils import to_categorical

def wav2mfcc(file_path, max_pad_len=20):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    
    wave = np.asfortranarray(wave[::3])
    
    mfcc = librosa.feature.mfcc(wave, sr=8000, n_mfcc=20)
    
    pad_width = max_pad_len - mfcc.shape[1]
  
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

def get_data(root):

    labels = []
    mfccs = []

    for f in os.listdir(root):
        if f.endswith('.wav'):
            # MFCC
            mfccs.append(wav2mfcc(root + f))

            # List of labels
            label = f.split('_')[0]
            labels.append(label)

    return np.asarray(mfccs), labels

if __name__ == '__main__':
    root = '../data/sound/0/'
    mfccs, labels = get_data(root)
    print(mfccs[0])
    print(len(labels))