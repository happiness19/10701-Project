'''
this is the file for reading in audio inputs and processing them into MFCCs
'''

import fnmatch
import os
import random
import numpy as np
# from python_speech_features import mfcc
import librosa.feature
import scipy.io.wavfile as wav
from scipy.io.wavfile import write as wav_write
import librosa

# make mfcc np array from wav file using speech features package
def make_mfcc(sig, rate = 8000):
    mfcc_feat = mfcc(sig, rate)
    mfcc_feat = mfcc_feat.T
    return mfcc_feat

def make_split_audio_array(folder, num_splits = 5):
    lst = []
    for filename in os.listdir(folder):
        if filename.endswith('wav'):
            normed_sig = make_standard_length(filename)
            chunk = normed_sig.shape[0]/num_splits
            for i in range(num_splits - 1):
                lst.append(normed_sig[i*chunk:(i+2)*chunk])
    lst = np.array(lst)
    lst = lst.reshape(lst.shape[0], -1)
    return lst

def downsample(filename, outrate=8000, write_wav = False):
    y, sr = librosa.load(filename, sr=22050)
    down_sig = librosa.core.resample(y, sr, outrate, scale=True)
    if not write_wav:
        return down_sig, outrate
    if write_wav:
        wav_write('{}_down_{}.wav'.format(filename, outrate), outrate, down_sig)

def make_standard_length(filename, n_samps=240000):
    down_sig, rate = downsample(filename)
    normed_sig = librosa.util.fix_length(down_sig, n_samps, mode= "wrap")
    normed_sig = (normed_sig - np.mean(normed_sig))/np.std(normed_sig)
    outrate = 8000
    return normed_sig

# make mfcc np array from wav file using librosa package
def make_librosa_mfcc(filename):
    y = make_standard_length(filename)
    mfcc_feat = librosa.feature.mfcc(y=y, sr=8000, n_mfcc=13)
    return mfcc_feat

def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def load_single_language_audio(directory):
    root = "../../../../../Downloads/cslu_fae/speech/"
    X = None
    initialized = False
    # files = find_files(root + directory)
    files = find_files(directory)
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        mfcc = make_librosa_mfcc(filename)
        print("mfcc shape")
        print(mfcc.shape)
        print("end")
        if (not initialized):
            X = mfcc
            initialized = True
        else:
            mfcc = np.reshape(1,13, 469)
            X = np.append(X, mfcc, axis=0)
    print(X.shape)
    return X

def load_audio():
    A = load_single_language_audio("AR")
    print("A done")
    J = load_single_language_audio("JA")
    X = np.append(A, J)
    y = np.append(np.ones(len(A)), np.zeros(len(J)))
    np.savetxt("../AR_mfcc",A)
    np.savetxt("../JA_mfcc",J)
    return X, y

# load_audio()
load_single_language_audio("../test")
    
