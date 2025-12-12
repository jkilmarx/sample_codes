import glob
import numpy as np
from scipy.fft import fft
from scipy.signal import firwin, lfilter, iirnotch, filtfilt
from sklearn.svm import SVC
from joblib import load
from statistics import mode
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing


# function to load in the saved data from the rest and imagery periods
# also contains an argument to preprocess the data
# set tune to true if you want to just load in the first 2 runs of the real-time session. otherwise it will load all 4 runs
def load_rest_img_data(file_path, target_subject, target_session, img_channel, img_ref_channel, preprocess=True, fs=500, duration_time = 3.5, add_delay_time = 0.6, q = 2, tune=False):

	img_band = [1, 125]
	
	first_flag = True

	if tune:
		max_session_num = 5
	else:
		max_session_num = 9

	# load in imagery and rest EEG data
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
	
	# load in imagery and rest labels
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

def rereference(x, ref, channel_axis):
	if len(x.shape) == len(ref.shape):
		tmp_ref = np.mean(ref, axis = channel_axis)
	else:
		tmp_ref = ref
			
	exp_ref = np.expand_dims(tmp_ref, axis = channel_axis)
	return x - exp_ref

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

class ImgOnlineBlock():
	def __init__(self, fs, channels, ref_channels, current_session, current_subject, root_file_path, clf_file = './model/img_model.joblib', duration_time = 3.5, add_delay_time = 0.6):
		
		self.img_band = [1, 125]
		self.file_path = root_file_path
		
		self.fs = fs
		self.channels = channels
		self.ref_channels = ref_channels
		self.current_session = current_session
		self.current_subject = current_subject
		self.duration_time = duration_time
		self.add_delay_time = add_delay_time
  
		self.n_chan = len(self.channels)
		self.n_time = int(self.duration_time * self.fs)
		self.add_delay = int(self.add_delay_time * self.fs)

		self.clf = load(clf_file)

	def initialize_classifier(self, target_subject, target_sessions):
		# once the first 2 training runs are complete, train the classifier with the desired number of previous sessions

		print('\nBeginning Classifier Initialization')
		first_flag = True

		for session in target_sessions:
			trials, labels = load_rest_img_data(self.file_path, target_subject, session, self.channels, self.ref_channels, preprocess=True)

			if first_flag:
				img_trials = trials
				img_labels = labels
				first_flag = False
			else:
				img_trials = np.concatenate((img_trials, trials), axis=0)
				img_labels = np.concatenate((img_labels, labels), axis=0)

		trials, labels = load_rest_img_data(self.file_path, self.current_subject, self.current_session, self.channels, self.ref_channels, preprocess=True, tune=True)
		img_trials = np.concatenate((img_trials, trials), axis=0)
		img_labels = np.concatenate((img_labels, labels), axis=0)

		# specifiy desired window size and feature frequency band
		window = int(1.75 * 500)
		step = int(window/2)
		mult = 3
		low = 1
		high = 100

		# extract features and train classifier
		segments = segment_trials(img_trials, window, step)
		features = create_features(segments, low, high)
		labels = np.hstack([img_labels]*mult)
		le = preprocessing.LabelEncoder()
		labels = le.fit_transform(labels)

		clf = SVC(kernel='linear')
		clf.fit(features, labels)

		print('\nClassifier Initialized\n')

		return(clf)


	def preprocess_and_classify(self, buffer, new_clf):
		# real-time preprocessing and classification of imagery eeg data

		img_raw = np.expand_dims(buffer, 0)
		img_trials = img_raw[:, self.channels, :]
		img_ref = img_raw[:, self.ref_channels, :]

		img_trials = rereference(img_trials, img_ref, channel_axis = 1)
		img_notched = notchFilter(img_trials, 60, self.fs)
		img_notched = notchFilter(img_notched, 120, self.fs)
		img_trials, n_discard = bandpass(img_notched, self.img_band, self.fs, 2, self.add_delay)
		img_trials = img_trials[:, :, n_discard:n_discard + self.n_time]
		img_trials = standardize(img_trials)

		window = int(1.75 * 500)
		step = int(window/2)
		low = 1
		high = 100

		segments = segment_trials(img_trials, window, step)
		features = create_features(segments, low, high)

		y_pred = new_clf.predict(features)
		#print(y_pred)

		# returning only the predicion from the middle of the imagery period gives best results
		#pred = mode(y_pred)
		pred = y_pred[1]

		return pred
