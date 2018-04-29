'''
this is the file for reading in audio inputs and processing them into MFCCs
'''
import numpy as np
# from python_speech_features import mfcc
import librosa.feature
import scipy.io.wavfile as wav
from scipy.io.wavfile import write as wav_write
import librosa
import os

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
    print(mfcc_feat.shape)
    return mfcc_feat

def load_audio(filename="../../../../../Downloads/cslu_fae/speech/AR/FAR00013.wav"):
    b = make_librosa_mfcc(filename)

load_audio("../test.wav")
    
