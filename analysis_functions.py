import glob
import numpy as np
from scipy.fft import fft
from scipy.signal import firwin, lfilter, iirnotch, filtfilt

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, cross_val_predict

import seaborn as sns
import matplotlib.pyplot as plt

# the following functions are useful for offline analysis of the recorded data 
# see an example of how to use them in the analysis_tutorial jupyter notebook

# the following functions are like the main one in the Processing_nk.py file
# each function will load in the saved data for the specified task period
def load_img_data(file_path, target_subject, target_session, img_channel, img_ref_channel, preprocess=True, fs=500, duration_time = 3.5, add_delay_time = 0.6, q = 2):

    img_band = [1, 125]
    
    first_flag = True

    max_session_num = 9

    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/IMG_S{target_session}B{i_b}.npy'
        tmp_block = np.load(tmp_file)
        if first_flag:
            img_trials_raw = tmp_block
            first_flag = False
        else:
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block), axis = 0)
    
    first_flag = True
    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/IMG_L_S{target_session}B{i_b}.npy'
        tmp_labels = np.load(tmp_file).astype(int)

        if first_flag:
            img_labels = tmp_labels
            first_flag = False
        else:
            img_labels = np.concatenate((img_labels, tmp_labels), axis = 0)

    img_trials = img_trials_raw[:, img_channel, :]

    if preprocess:
        img_ref = img_trials_raw[:, img_ref_channel, :]
        add_delay = int(add_delay_time * fs)

        img_trials = rereference(img_trials, img_ref, channel_axis = 1)
        #img_trials = remove_mean(img_trials, 2)

        img_tmp = img_trials
        
        img_notched = notchFilter(img_trials, 60, 500)
        img_notched = notchFilter(img_notched, 120, 500)
        img_trials, n_discard = bandpass(img_notched, img_band, fs, 2, add_delay)
        img_trials = img_trials[:, :, n_discard:n_discard + int(duration_time * fs)]
        img_trials = standardize(img_trials)
        #img_trials = decimate(img_trials, q, axis = 2)
    
    return img_trials, img_labels

def load_obs_data(file_path, target_subject, target_session, obs_channel, obs_ref_channel, preprocess=True, fs=500, duration_time = 3.5, add_delay_time = 0.6, q = 2):

    obs_band = [1, 40]
    
    first_flag = True

    max_session_num = 9

    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/OBS_S{target_session}B{i_b}.npy'
        tmp_block = np.load(tmp_file)
        if first_flag:
            obs_trials_raw = tmp_block
            first_flag = False
        else:
            obs_trials_raw = np.concatenate((obs_trials_raw, tmp_block), axis = 0)
    
    first_flag = True
    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/OBS_L_S{target_session}B{i_b}.npy'
        tmp_labels = np.load(tmp_file).astype(int)

        if first_flag:
            obs_labels = tmp_labels
            first_flag = False
        else:
            obs_labels = np.concatenate((obs_labels, tmp_labels), axis = 0)

    obs_trials = obs_trials_raw[:, obs_channel, :]

    if preprocess:
        obs_ref = obs_trials_raw[:, obs_ref_channel, :]
        add_delay = int(add_delay_time * fs)

        obs_trials = rereference(obs_trials, obs_ref, channel_axis = 1)
        #obs_trials = remove_mean(obs_trials, 2)

        obs_tmp = obs_trials
        
        obs_notched = notchFilter(obs_trials, 60, 500)
        obs_trials, n_discard = bandpass(obs_notched, obs_band, fs, 2, add_delay)
        obs_trials = obs_trials[:, :, n_discard:n_discard + int(duration_time * fs)]
        obs_trials = standardize(obs_trials)
        #obs_trials = decimate(obs_trials, q, axis = 2)
    
    return obs_trials, obs_labels

def load_rest_data(file_path, target_subject, target_session, rest_channel, rest_ref_channel, preprocess=True, fs=500, duration_time = 3.5, add_delay_time = 0.6, q = 2):

    rest_band = [1, 125]

    first_flag = True

    max_session_num = 9

    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/REST_S{target_session}B{i_b}.npy'
        tmp_block = np.load(tmp_file)
        if first_flag:
            rest_trials_raw = tmp_block
            first_flag = False
        else:
            rest_trials_raw = np.concatenate((rest_trials_raw, tmp_block), axis = 0)
    
    first_flag = True
    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/REST_L_S{target_session}B{i_b}.npy'
        tmp_labels = np.load(tmp_file).astype(int)

        if first_flag:
            rest_labels = tmp_labels
            first_flag = False
        else:
            rest_labels = np.concatenate((rest_labels, tmp_labels), axis = 0)

    rest_trials = rest_trials_raw[:, rest_channel, :]

    if preprocess:
        rest_ref = rest_trials_raw[:, rest_ref_channel, :]
        add_delay = int(add_delay_time * fs)

        rest_trials = rereference(rest_trials, rest_ref, channel_axis = 1)
        #rest_trials = remove_mean(rest_trials, 2)

        rest_tmp = rest_trials

        rest_trials, n_discard = bandpass(rest_trials, rest_band, fs, 2, add_delay)
        rest_trials = rest_trials[:, :, n_discard:n_discard + int(duration_time * fs)]
        rest_trials = standardize(rest_trials)
        #rest_trials = decimate(rest_trials, q, axis = 2)

    return rest_trials, rest_labels

def load_rest_obs_data(file_path, target_subject, target_session, img_channel, img_ref_channel, preprocess=True, fs=500, duration_time = 3.5, add_delay_time = 0.6, q = 2):

    img_band = [1, 40]
    
    first_flag = True

    max_session_num = 9

    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/OBS_S{target_session}B{i_b}.npy'
        tmp_file2 = f'{file_path}{target_subject}/S{target_session}/REST_S{target_session}B{i_b}.npy'
        tmp_block = np.load(tmp_file)
        tmp_block2 = np.load(tmp_file2)
        if first_flag:
            img_trials_raw = tmp_block
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block2), axis = 0)
            first_flag = False
        else:
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block), axis = 0)
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block2), axis = 0)
    
    first_flag = True
    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/OBS_L_S{target_session}B{i_b}.npy'
        tmp_file2 = f'{file_path}{target_subject}/S{target_session}/REST_L_S{target_session}B{i_b}.npy'
        tmp_labels = np.load(tmp_file).astype(int)
        tmp_labels2 = np.load(tmp_file2).astype(int)

        if first_flag:
            img_labels = tmp_labels
            img_labels = np.concatenate((img_labels, tmp_labels2), axis = 0)
            first_flag = False
        else:
            img_labels = np.concatenate((img_labels, tmp_labels), axis = 0)
            img_labels = np.concatenate((img_labels, tmp_labels2), axis = 0)

    img_trials = img_trials_raw[:, img_channel, :]

    if preprocess:
        img_ref = img_trials_raw[:, img_ref_channel, :]
        add_delay = int(add_delay_time * fs)

        img_trials = rereference(img_trials, img_ref, channel_axis = 1)
        #img_trials = remove_mean(img_trials, 2)

        img_tmp = img_trials
        
        img_notched = notchFilter(img_trials, 60, 500)
        img_notched = notchFilter(img_notched, 120, 500)
        img_trials, n_discard = bandpass(img_notched, img_band, fs, 2, add_delay)
        img_trials = img_trials[:, :, n_discard:n_discard + int(duration_time * fs)]
        img_trials = standardize(img_trials)
        #img_trials = decimate(img_trials, q, axis = 2)
    
    return img_trials, img_labels

def load_rest_img_data(file_path, target_subject, target_session, img_channel, img_ref_channel, preprocess=True, fs=500, duration_time = 3.5, add_delay_time = 0.6, q = 2, tune=False):

    img_band = [1, 125]
    
    first_flag = True

    if tune:
        max_session_num = 5
    else:
        max_session_num = 9

    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/IMG_S{target_session}B{i_b}.npy'
        tmp_file2 = f'{file_path}{target_subject}/S{target_session}/REST_S{target_session}B{i_b}.npy'
        tmp_block = np.load(tmp_file)
        tmp_block2 = np.load(tmp_file2)
        if first_flag:
            img_trials_raw = tmp_block
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block2), axis = 0)
            first_flag = False
        else:
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block), axis = 0)
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block2), axis = 0)
    
    first_flag = True
    for i_b in range(1, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/IMG_L_S{target_session}B{i_b}.npy'
        tmp_file2 = f'{file_path}{target_subject}/S{target_session}/REST_L_S{target_session}B{i_b}.npy'
        tmp_labels = np.load(tmp_file).astype(int)
        tmp_labels2 = np.load(tmp_file2).astype(int)

        if first_flag:
            img_labels = tmp_labels
            img_labels = np.concatenate((img_labels, tmp_labels2), axis = 0)
            first_flag = False
        else:
            img_labels = np.concatenate((img_labels, tmp_labels), axis = 0)
            img_labels = np.concatenate((img_labels, tmp_labels2), axis = 0)

    img_trials = img_trials_raw[:, img_channel, :]

    if preprocess:
        img_ref = img_trials_raw[:, img_ref_channel, :]
        add_delay = int(add_delay_time * fs)

        img_trials = rereference(img_trials, img_ref, channel_axis = 1)
        #img_trials = remove_mean(img_trials, 2)

        img_tmp = img_trials
        
        img_notched = notchFilter(img_trials, 60, 500)
        img_notched = notchFilter(img_notched, 120, 500)
        img_trials, n_discard = bandpass(img_notched, img_band, fs, 2, add_delay)
        img_trials = img_trials[:, :, n_discard:n_discard + int(duration_time * fs)]
        img_trials = standardize(img_trials)
        #img_trials = decimate(img_trials, q, axis = 2)
    
    return img_trials, img_labels

def load_real_time_data(file_path, target_subject, target_session, img_channel, img_ref_channel, preprocess=True, fs=500, duration_time = 3.5, add_delay_time = 0.6, q = 2):
    # this function has the additional feature to return the predictions made during the real-time bci runs

    img_band = [1, 125]
    
    first_flag = True

    max_session_num = 9

    for i_b in range(5, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/IMG_S{target_session}B{i_b}.npy'
        tmp_block = np.load(tmp_file)
        if first_flag:
            img_trials_raw = tmp_block
            first_flag = False
        else:
            img_trials_raw = np.concatenate((img_trials_raw, tmp_block), axis = 0)
    
    first_flag = True
    for i_b in range(5, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/IMG_L_S{target_session}B{i_b}.npy'
        tmp_labels = np.load(tmp_file).astype(int)

        if first_flag:
            img_labels = tmp_labels
            first_flag = False
        else:
            img_labels = np.concatenate((img_labels, tmp_labels), axis = 0)
            
    first_flag = True
    for i_b in range(5, max_session_num):
        tmp_file = f'{file_path}{target_subject}/S{target_session}/IMG_P_S{target_session}B{i_b}.npy'
        tmp_preds = np.load(tmp_file).astype(int)

        if first_flag:
            img_preds = tmp_preds
            first_flag = False
        else:
            img_preds = np.concatenate((img_preds, tmp_preds), axis = 0)

    img_trials = img_trials_raw[:, img_channel, :]

    if preprocess:
        img_ref = img_trials_raw[:, img_ref_channel, :]
        add_delay = int(add_delay_time * fs)

        img_trials = rereference(img_trials, img_ref, channel_axis = 1)
        #img_trials = remove_mean(img_trials, 2)

        img_tmp = img_trials
        
        img_notched = notchFilter(img_trials, 60, 500)
        img_notched = notchFilter(img_notched, 120, 500)
        img_trials, n_discard = bandpass(img_notched, img_band, fs, 2, add_delay)
        img_trials = img_trials[:, :, n_discard:n_discard + int(duration_time * fs)]
        img_trials = standardize(img_trials)
        #img_trials = decimate(img_trials, q, axis = 2)
    
    return img_trials, img_labels, img_preds

def rereference(x, ref, channel_axis):
    if len(x.shape) == len(ref.shape):
        tmp_ref = np.mean(ref, axis = channel_axis)
    else:
        tmp_ref = ref
            
    exp_ref = np.expand_dims(tmp_ref, axis = channel_axis)
    return x - exp_ref

def remove_mean(x, axis):
    mean_x = np.expand_dims(np.mean(x, axis = axis), axis = axis)
    return x - mean_x

def bandpass(x, cutoff_hz, sample_rate, axis, add_delay, zi = None):
    # bandpass EEG signal using a hann window

    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    taps = firwin(400, [cutoff_hz[0]/nyq_rate, cutoff_hz[1]/nyq_rate], window=('hann'), pass_zero='bandpass')
    
    dim = list(x.shape)
    dim[axis] = (len(taps) + 1) // 2 + add_delay
    dim = tuple(dim)
    pad_x = np.concatenate((x, np.zeros(dim)), axis = axis)
    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, pad_x, axis = axis, zi = zi)
    
    return filtered_x, (len(taps) + 1) // 2 + add_delay

def notchFilter(sig, freq=60, fs=500):
    Q = 30
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig, axis=2)

def standardize(X):
    
    for i in range(X.shape[0]):
        tmp_X = X[i, :, :]
        X[i, :, :] = ((tmp_X.T - np.mean(tmp_X, 1)) / np.std(tmp_X, 1)).T
    
    return X

def feature_spectrum(sig, low, high):
    # use fft to extract features from desired frequency band

    fft_vals = np.abs(np.fft.rfft(sig))
    fft_freq = np.fft.rfftfreq(sig.size, 1/500)
    freq_ix = np.where((fft_freq >= low) & (fft_freq <= high))[0]
    
    return fft_vals[freq_ix]

def create_features(trials, low, high):
    # loop through each trial to extract features

    features = []
    for trial in trials:
        sample = []
        for channel in range(trials.shape[1]):
            sample.append(feature_spectrum(trial[channel,:], low, high))
        features.append(np.hstack(sample))
    
    return features

def segment_trials(trials, window=int(3.5*500), step=int(3.5*500)):
    # use a sliding window to epoch data

    segs = []
    for i in range(0, trials.shape[2]-window+step-1, step):
        segs.append(trials[:,:,i:i+window])
    segments = np.vstack(segs)
    return(segments)