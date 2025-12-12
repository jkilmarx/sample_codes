from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet, LostError
import time
import datetime
import numpy as np
import signal
import sys
import glob
import traceback
import os
import argparse
from Processing_nk import *

# add a command line option to specify the name for the current subject. in none is specified, it will default to 'demo'
parser = argparse.ArgumentParser(description='BCI experiment parameters')
parser.add_argument('-s', '--subjectid', help='Subject ID', default='demo')
args = parser.parse_args()

# list the path to where the data should be stored. this code will make a subfolder there with the subject name
root_file_path = '../Data/'
subdir = root_file_path + args.subjectid
if os.path.isdir(subdir + '/') == False:
    os.mkdir(subdir)

# use ctrl+c to exit the program. all the data will be saved in backup numpy arrays
def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Good-bye!')
    np.save(file_path + 'markers.npy', np.hstack(markers))
    np.save(file_path + 'marker_time.npy', np.hstack(marker_time))
    np.save(file_path + 'eeg_data.npy', eeg_data)
    np.save(file_path + 'eeg_time.npy', np.hstack(eeg_time))
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# specify how many trials should be saved in each block, and how many blocks will be used to initialize the classifier
TRIAL_NUM_PER_BLOCK = 9
num_tuning_blocks = 4

# list the subject and sessions you want to include in the classifier training
train_subject = 'justin_nk'
train_session = [4,5,6]

# specify the timings and desired channels for the experiment
fs = 500
feedback_duration = 4
feedback_n_time = np.round(fs * feedback_duration).astype(int)

rest_duration = 4
rest_n_time = np.round(fs * rest_duration).astype(int)

obs_duration = 4
obs_n_time = np.round(fs * obs_duration).astype(int)

img_duration = 4
img_n_time = np.round(fs * img_duration).astype(int)

prefix_duration = 0.5
prefix_n_time = np.round(fs * prefix_duration).astype(int)

img_channel = np.array([1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32]) - 1
img_ref_channel = np.array([10, 21]) - 1
img_delay = 0.6

# look for previous session folders and create a new subfolder for the current session. 
sStart = '1'
start_sess_num = int(sStart)

sess_num = start_sess_num
while True:
    testpath = subdir + '/S' + str(sess_num) + '/'
    foundfiles = glob.glob(testpath + '*.npy')            
    if len(foundfiles) == 0:
        this_sess_num = sess_num
        break
    sess_num = sess_num + 1

newdir = subdir + '/S' + str(this_sess_num)
if os.path.isdir(newdir + '/') == False:
    os.mkdir(newdir)

file_path = newdir + '/'

print("Saving to: " + file_path)

feedback_block_num = 1
rest_block_num = 1
obs_block_num = 1
img_block_num = 1

# init img online block
ImgOnlineBlock = ImgOnlineBlock(fs, img_channel, img_ref_channel, this_sess_num, args.subjectid, root_file_path, duration_time=3.5)

# create a stream to transmit the feedback to psychopy during real-time runs
info = StreamInfo('Result', 'Results', 1, 0, 'string', 'pythonresult')
# next make an outlet
outlet = StreamOutlet(info)

markers = []
marker_time = []
eeg_data = []
eeg_time = []

while True:

    print('stream stopped.')

    try:
        # first resolve an EEG stream on the lab network
        print('looking for an EEG stream...')

        while True:
            eeg_streams = resolve_byprop('name', 'openvibeSignal', 1, 0.5)
            if len(eeg_streams) > 0:
                break
            time.sleep(0.5)
            
        # create a new inlet to read from the stream
        eeg_inlet = StreamInlet(eeg_streams[0], recover = False)

        print('looking for a marker stream...')

        while True:
            marker_streams = resolve_byprop('name', 'event_markers', 1, 0.5)
            if len(marker_streams) > 0:
                break
            time.sleep(0.5)

        # create a new inlet to read from the stream
        marker_inlet = StreamInlet(marker_streams[0], recover = False)


    except Exception as e:
        print(e)

    print('streaming...')

    # create a buffer array to read the incoming EEG data into
    n_chan = eeg_inlet.channel_count
    buffer = np.random.rand(n_chan, obs_n_time + prefix_n_time)

    # initialize some parameters for the experiment
    epoch_sample_count = -1
    last_label = 0
    last_obs_label = 0
    last_type = 0

    feedback_trial_count = 0
    rest_trial_count = 0
    obs_trial_count = 0
    img_trial_count = 0

    feedback_block_trials = np.zeros((TRIAL_NUM_PER_BLOCK, n_chan, feedback_n_time + prefix_n_time))
    feedback_block_labels = np.ones((TRIAL_NUM_PER_BLOCK, )).astype(int) * -1

    rest_block_trials = np.zeros((TRIAL_NUM_PER_BLOCK, n_chan, rest_n_time + prefix_n_time))
    rest_block_labels = np.ones((TRIAL_NUM_PER_BLOCK, )).astype(int) * -1

    obs_block_trials = np.zeros((TRIAL_NUM_PER_BLOCK, n_chan, obs_n_time + prefix_n_time))
    obs_block_labels = np.ones((TRIAL_NUM_PER_BLOCK, )).astype(int) * -1

    img_block_trials = np.zeros((TRIAL_NUM_PER_BLOCK, n_chan, img_n_time + prefix_n_time))
    img_block_labels = np.ones((TRIAL_NUM_PER_BLOCK, )).astype(int) * -1
    img_block_predicts = np.ones((TRIAL_NUM_PER_BLOCK, )).astype(int) * -1

    marker_inlet.pull_chunk()
    eeg_inlet.pull_chunk()
    
    try:
        while True:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)
            try:
                eeg_sample, eeg_t = eeg_inlet.pull_sample(0)
                marker_sample, marker_t = marker_inlet.pull_sample(0)
            except LostError as le:
                print(le)
                break

            # read marker stream and identify correct label
            if marker_sample != None:
                markers.append(marker_sample)
                marker_time.append(marker_t)
                last_label = marker_sample[0]

                if last_label in [210, 220, 230]:
                    epoch_sample_count = obs_n_time
                    print('\nNew obs epoch...')
                elif last_label in [310, 320, 330]:
                    epoch_sample_count = img_n_time
                    print('New img epoch...')
                elif last_label == 100 and rest_block_num < num_tuning_blocks+1:
                    epoch_sample_count = rest_n_time
                    print('New rest epoch...')
                elif last_label == 100 and rest_block_num == num_tuning_blocks+1:
                    epoch_sample_count = feedback_n_time
                    print('New feedback epoch...')

            # read EEG stream and save data 
            if eeg_sample != None:
                eeg_data.append(eeg_sample)
                eeg_time.append(eeg_t)
                buffer[:, :-1] = buffer[:, 1:]
                buffer[:, -1] = np.array(eeg_sample)

                # use a countdown for trial timings
                if epoch_sample_count > -1:
                    epoch_sample_count -= 1
                if epoch_sample_count == 0:

                    # in the first 2 runs, this data should be saved as rest periods
                    if last_label in [100, 101] and rest_block_num < num_tuning_blocks+1:
                        print('End rest epoch...')
                        last_rest_label = last_label
                        rest_block_trials[rest_trial_count, :, :] = buffer
                        rest_block_labels[rest_trial_count] = last_label

                        rest_trial_count += 1

                        if rest_trial_count == TRIAL_NUM_PER_BLOCK:
                            print("\nSaving Rest Block: " + str(rest_block_num))

                            np.save(file_path + f'REST_S{this_sess_num}B{rest_block_num}.npy', rest_block_trials)
                            np.save(file_path + f'REST_L_S{this_sess_num}B{rest_block_num}.npy', rest_block_labels)

                            if rest_block_num == num_tuning_blocks:
                                clf = ImgOnlineBlock.initialize_classifier(train_subject, train_session)

                            rest_trial_count = 0
                            rest_block_num += 1


                    # in the last 2 runs, this data should be saved as feedback periods
                    elif last_label in [100, 101] and rest_block_num == num_tuning_blocks+1:
                        print('End feedback epoch...')
                        last_feedback_label = last_label
                        feedback_block_trials[feedback_trial_count, :, :] = buffer
                        feedback_block_labels[feedback_trial_count] = last_label

                        feedback_trial_count += 1

                        if feedback_trial_count == TRIAL_NUM_PER_BLOCK:
                            print("\nSaving Feedback Block: " + str(feedback_block_num))

                            np.save(file_path + f'FB_S{this_sess_num}B{feedback_block_num}.npy', feedback_block_trials)
                            np.save(file_path + f'FB_L_S{this_sess_num}B{feedback_block_num}.npy', feedback_block_labels)

                            feedback_trial_count = 0
                            feedback_block_num += 1


                    # save data during observation periods
                    elif last_label in [210, 220, 230, 211,221,231]:
                        print('End obs epoch...')
                        last_obs_label = last_label
                        obs_block_trials[obs_trial_count, :, :] = buffer
                        obs_block_labels[obs_trial_count] = last_label

                        obs_trial_count += 1

                        if obs_trial_count == TRIAL_NUM_PER_BLOCK:
                            print("\nSaving obs Block: " + str(obs_block_num))

                            np.save(file_path + f'OBS_S{this_sess_num}B{obs_block_num}.npy', obs_block_trials)
                            np.save(file_path + f'OBS_L_S{this_sess_num}B{obs_block_num}.npy', obs_block_labels)

                            obs_trial_count = 0
                            obs_block_num += 1


                    # save data in imagery periods
                    elif last_label in [310, 320, 330,311,321,331]:
                        print('End img epoch...')
                        last_img_label = last_label
                        img_block_trials[img_trial_count, :, :] = buffer
                        img_block_labels[img_trial_count] = last_label

                        # if we are in the real-time runs, send the data over for classification
                        if img_block_num >= num_tuning_blocks+1:
                            result = ImgOnlineBlock.preprocess_and_classify(buffer, clf)
                            img_block_predicts[img_trial_count] = result

                            if result == 0:
                                pred = 'REST'
                            elif result == 1:
                                pred = 'FACE'
                            elif result == 2:
                                pred = 'SCENE'

                            outlet.push_sample([pred])
                            print(f'send a result: {pred}')

                        img_trial_count += 1

                        if img_trial_count == TRIAL_NUM_PER_BLOCK:
                            print("\nSaving img Block: " + str(img_block_num))

                            np.save(file_path + f'IMG_S{this_sess_num}B{img_block_num}.npy', img_block_trials)
                            np.save(file_path + f'IMG_L_S{this_sess_num}B{img_block_num}.npy', img_block_labels)
                            np.save(file_path + f'IMG_P_S{this_sess_num}B{img_block_num}.npy', img_block_predicts)

                            img_trial_count = 0
                            img_block_num += 1


    except Exception as e:
        traceback.print_exc()
        print(e)