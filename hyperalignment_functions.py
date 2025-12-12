import hypertools as hyp
import numpy as np
import nilearn
from nilearn.input_data import NiftiMasker
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import time
import seaborn as sea
from matplotlib.pyplot import figure
from sklearn.linear_model import LogisticRegression
import itertools

def z_score(baseline_data, run_data):
    baseline_sigmas = np.sqrt(np.var(baseline_data, axis=0))
    out_data = (run_data/baseline_sigmas)
    return out_data

def get_data(sub_names, subdir, zscore='all', alignment=None, timer=False):
    
    """
    load in fMRI and behavioral data into a single pandas dataframe
    alignment can be set to 'anatomical' to load in MNI aligned data
    """
    
    if timer:
        start = time.time()
        
    data = pd.DataFrame()
    for sub in sub_names:

        print('Loading fMRI data for %s' %sub)
        
        if sub[0:2] == 'ff':

            # create a masker and give it the fMRI run data
            if alignment == 'anatomical':
                mask = (subdir + "/mask_m1s1.nii.gz")
            else:
                mask = (subdir + "/%s/ref/mask_lh_m1s1.nii") %sub
            masker = NiftiMasker(mask_img=mask, standardize=False, detrend=True, t_r=2)

            # load in fmri data
            allds = []
            for i in range(1,9):
                if alignment == 'anatomical':
                    fmri = (subdir + "/%s/bold_mni/sess1/rrun-00%s_mni.nii.gz") %(sub,i)
                else:
                    fmri = (subdir + "/%s/bold/sess1/rrun-00%s.nii") %(sub,i)
                ds = masker.fit_transform(fmri)
                allds.append(ds)
            for i in range(1,9):
                if alignment == 'anatomical':
                    fmri = (subdir + "/%s/bold_mni/sess2/rrun-00%s_mni.nii.gz") %(sub,i)
                else:
                    fmri = (subdir + "/%s/bold/sess2/rrun-00%s.nii") %(sub,i)
                ds = masker.fit_transform(fmri)
                allds.append(ds)
            fmri_masked = np.vstack(allds)

            # eliminate TRs 1,2,3,7,8 
            a = (([False]*20 + ([False]*3 + [True]*3 + [False]*2)*20)*8)*2
            three_trs = fmri_masked[a]
            
            # get baseline data
            b = (([True]*20 + ([False]*8)*20)*8)*2
            baseline = fmri_masked[b]

            # average accross TRs 4,5,6
            fmri_data = np.empty((0,len(fmri_masked[1])))
            for i in range(0, len(three_trs), 3):
                temp = np.mean(three_trs[i:i+3],axis=0)
                fmri_data = np.append(fmri_data, [temp], axis=0)

            
            fmri_zscored = []
            for i in range(16):
                if zscore == 'baseline':
                    temp = z_score(baseline[(20*i):(20*i+20)], fmri_data[(20*i):(20*i+20)])
                if zscore == 'all':
                    temp = z_score(fmri_masked[(180*i):(180*i+180)], fmri_data[(20*i):(20*i+20)])
                fmri_zscored.append(temp) 
            
            fmri_zscored = np.vstack(fmri_zscored)
            # set any nan values to 0
            fmri_zscored = np.nan_to_num(fmri_zscored)
            temp = pd.DataFrame(fmri_zscored)

            # load in the behavioral data to a single pandas dataframe
            behavioral1 = pd.read_csv((subdir + "/%s/ref/ft-data-sess1.txt") %sub)
            behavioral1['session'] = '1'
            behavioral1['subject'] = '%s' %sub
            behavioral2 = pd.read_csv((subdir + "/%s/ref/ft-data-sess2.txt") %sub)
            behavioral2['session'] = '2'
            behavioral2['run_num'] = behavioral2['run_num'] + 8
            behavioral2['subject'] = '%s' %sub
            behavioral3 = pd.concat([behavioral1, behavioral2],axis=0,ignore_index=True)
            # save only one label for each trial
            behavioral3 = behavioral3[::10]
            # add in fmri data to dataframe
            behavioral3['fmri'] = temp.values.tolist()
            behavioral3 = behavioral3.drop(['press', 'accuracy', 'rt', 'low_threshold', 'high_threshold', 'trial_num'], axis=1)
            behavioral3 = behavioral3.drop([0, 2600])
            data = data.append(behavioral3, ignore_index=True)
            
        if sub[0:2] == 'ft':
            # create a masker and give it the fMRI run data
            if alignment == 'anatomical':
                mask = (subdir + "/mask_m1s1.nii.gz")
            else:
                mask = (subdir + "/%s/ref/mask_lh_hand_knob.nii") %sub
            masker = NiftiMasker(mask_img=mask, standardize=False, detrend=True, t_r=2)

            # load in fmri data
            allds = []
            for i in range(1,9):
                if alignment == 'anatomical':
                    fmri = (subdir + "/%s/bold_mni/localizer/rrun-00%i_mni.nii.gz") %(sub,i)
                else:
                    fmri = (subdir + "/%s/bold/localizer/rrun-00%i.nii") %(sub,i)
                ds = masker.fit_transform(fmri)
                allds.append(ds)
            fmri_masked = np.vstack(allds)

            # eliminate TRs 1,2,3,7,8 
            a = (([False]*10 + ([False]*3 + [True]*3 + [False]*2)*20)*8)
            three_trs = fmri_masked[a]
            
            # get baseline data
            b = (([True]*10 + ([False]*8)*20)*8)
            baseline = fmri_masked[b]

            # average accross TRs 4,5,6
            fmri_data = np.empty((0,len(fmri_masked[1])))
            for i in range(0, len(three_trs), 3):
                temp = np.mean(three_trs[i:i+3],axis=0)
                fmri_data = np.append(fmri_data, [temp], axis=0)
            
            fmri_zscored = []
            for i in range(8):
                if zscore == 'baseline':
                    temp = z_score(baseline[(10*i):(10*i+10)], fmri_data[(20*i):(20*i+20)])
                if zscore == 'all':
                    temp = z_score(fmri_masked[(170*i):(170*i+170)], fmri_data[(20*i):(20*i+20)])
                fmri_zscored.append(temp)            
            
            fmri_zscored = np.vstack(fmri_zscored)
            # set any nan values to 0
            fmri_zscored = np.nan_to_num(fmri_zscored)
            temp = pd.DataFrame(fmri_zscored)
            
            # load in the behavioral data to a single pandas dataframe
            behavioral1 = pd.read_csv((subdir + "/%s/ref/localizer_press.csv") %sub)
            behavioral1['session'] = '1'
            behavioral1['subject'] = '%s' %sub
            # save only one label for each trial
            behavioral2 = behavioral1.groupby('trial',sort=False).first()
            # add in fmri data to dataframe
            behavioral2['fmri'] = temp.values.tolist()
            behavioral2 = behavioral2.rename(index=str, columns={'target_finger': 'probe', 'run': 'run_num'})
            behavioral3 = behavioral2.drop(['trial_time', 'force_0', 'force_1', 'force_2', 'force_3', 'press_type'], axis=1).reset_index()
            behavioral3.index.name = None
            behavioral3['run_num'] = behavioral3['run_num'].shift(-1)
            behavioral4 = behavioral3[:-1]
            data = data.append(behavioral4, ignore_index=True, sort=True)

        if timer:
            end = time.time()
            elapsed = end - start
            m, s = divmod(elapsed, 60)
            print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
            
    return data

def select_features(data, sub_names, runs1, runs2, features=100, drop_fingers=[5], timer=False):
    
    """
    perform ANOVA feature selection on 100 best voxels and returns the transform
    runs can be set to any value 1-15 for feature selection
    a list of values [0-3] can be used to drop subsets of fingers, a value of [5] does not drop any fingers by default
    """
    
    if timer:
        start = time.time()
    
    # drop selected fingers from dataframe
    for finger in drop_fingers:
        idx = data.index[data['probe'] == finger].tolist()
        data = data.drop(idx, axis=0).reset_index(drop=True)
        
    # select features for each subject from selected runs
    transforms = []
    for sub in sub_names:
        sub_group = data.loc[data['subject'] == sub]
        
        all_fmri = []
        all_conditions = []
        if sub[0:2] == 'ff':
            for run in runs1:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['fmri'].values.tolist()
                all_fmri.append(fmri_data)
                conditions = run_group['probe'].values
                all_conditions.append(conditions)
                
        if sub[0:2] == 'ft':
            for run in runs2:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['fmri'].values.tolist()
                all_fmri.append(fmri_data)
                conditions = run_group['probe'].values
                all_conditions.append(conditions)
            
        # perform feature selection
        fmri_data = np.vstack(all_fmri)
        conditions = np.hstack(all_conditions)
        selection = SelectKBest(f_classif, k=features).fit(fmri_data, conditions)
        transforms.append(selection)
        
    if timer: 
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
        
    return transforms


def apply_features(data, transform, sub_names, runs1, runs2, drop_fingers=[5], leave_out=False, timer=False):
    
    """
    apply selected features transform to selected runs
    adds new features to original dataframe
    if leave_out is true, the data used for feature selection will be removed from dataframe
    """
    
    if timer:
        start=time.time()
        
    # drop selected fingers from dataframe
    for finger in drop_fingers:
        idx = data.index[data['probe'] == finger].tolist()
        data = data.drop(idx, axis=0).reset_index(drop=True)
    
    # apply feature selection to each subject and selected runs
    new_data = pd.DataFrame()
    i=0
    for sub in sub_names:
        sub_group = data.loc[data['subject'] == sub]
        if sub[0:2] == 'ff':
            for run in runs1:
                run_group = sub_group.loc[sub_group['run_num'] == (run)].copy()
                fmri_data = run_group['fmri'].values.tolist()
                features = transform[i].transform(fmri_data)
                temp = pd.DataFrame(features)
                run_group['features'] = temp.values.tolist()
                new_data = new_data.append(run_group, ignore_index=True)
        
        if sub[0:2] == 'ft':
            for run in runs2:
                run_group = sub_group.loc[sub_group['run_num'] == (run)].copy()
                fmri_data = run_group['fmri'].values.tolist()
                features = transform[i].transform(fmri_data)
                temp = pd.DataFrame(features)
                run_group['features'] = temp.values.tolist()
                new_data = new_data.append(run_group, ignore_index=True)
        i = i+1
    
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
    
    return new_data

def create_common_model(data, sub_names, runs1, runs2, get_template=False, get_transforms=True, timer=False):
    
    '''
    Create a common model space using procrustes transformation according to Haxby et al. 2011
    Set get_template = True to return template for common model space.
    Set get_transforms = True to return transforms for each subject into common model space.
    '''
    
    if timer:
        start = time.time()
        
    if get_template:
        get_transforms=False
    
    # compile data to use for hyperalignment
    fmri_all = []
    for sub in sub_names:
        sub_group = data.loc[data['subject'] == sub]

        fmri_run = []
        if sub[0:2] == 'ff':
            for run in runs1:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['features'].values.tolist()
                fmri_run.append(fmri_data)
            fmri_sub = np.vstack(fmri_run)
            fmri_all.append(fmri_sub)

        if sub[0:2] == 'ft':
            for run in runs2:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['features'].values.tolist()
                fmri_run.append(fmri_data)
            fmri_sub = np.vstack(fmri_run)
            fmri_all.append(fmri_sub)          

    # stage 1: iteratively align subjects together to construct common template        
    for x in range(0, len(fmri_all)):
        if x==0:
            template = fmri_all[x].copy()
        else:
            new_align = hyp.tools.procrustes(fmri_all[x], template / (x+1), transform=False)
            template += new_align
    template /= len(fmri_all)

    # stage 2: re-align each subject to common template found in stage 1 to create new common template
    template2 = np.zeros(template.shape)
    for x in range(0, len(fmri_all)):
        new_align = hyp.tools.procrustes(fmri_all[x], template, transform=False)
        template2 += new_align
    template2 /= len(fmri_all)

    # stage 3: obtain transforms for each subject to common template space
    transforms = [np.zeros(template2.shape)] * len(fmri_all)
    for x in range(0, len(fmri_all)):
        transform = hyp.tools.procrustes(fmri_all[x], template2, transform=True)
        transforms[x] = transform
        
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
        
    if get_template:
        return template2
    if get_transforms:
        return transforms
    
def align_with_model(data, template, sub_names, runs1, runs2, timer=False):
    
    # obtain transformation for hyperalignment into a common model space from selected runs of new subject
    # this is used if template is returned from create_common_model
    
    if timer:
        start = time.time()
        
    # compile data to use for hyperalignment
    fmri_all = []
    for sub in sub_names:
        sub_group = data.loc[data['subject'] == sub]

        fmri_run = []
        if sub[0:2] == 'ff':
            for run in runs1:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['features'].values.tolist()
                fmri_run.append(fmri_data)
            fmri_sub = np.vstack(fmri_run)
            fmri_all.append(fmri_sub)

        if sub[0:2] == 'ft':
            for run in runs2:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['features'].values.tolist()
                fmri_run.append(fmri_data)
            fmri_sub = np.vstack(fmri_run)
            fmri_all.append(fmri_sub)          

    template=template[:len(fmri_all[0])]
    # stage 3: obtain transforms for each subject to common template space
    transforms = [np.zeros(template.shape)] * len(fmri_all)
    for x in range(0, len(fmri_all)):
        transform = hyp.tools.procrustes(fmri_all[x], template, transform=True)
        transforms[x] = transform
        
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
        
    return transforms
                    
def hyperalign(data, transform, sub_names, runs1, runs2, timer=False):
    
    """
    apply hyperalignment transforms onto selected runs
    function will align each subject into first subject's space
    adds new hyperaligned data to given dataframe
    """
    
    if timer:
        start = time.time()
    
    new_data = pd.DataFrame()
        
    # compile all subjects' data for hyperalignment
    i=0
    for sub in sub_names:
        sub_group = data.loc[data['subject'] == sub]
        # apply transform to selected runs and save to the new dataframe 
        if sub[0:2] == 'ff':
            for run in runs1:
                run_group = sub_group.loc[sub_group['run_num'] == (run)].copy()
                fmri_data = run_group['features'].values.tolist()
                d = np.asmatrix(fmri_data)
                res = (d*transform[i]).A
                temp = pd.DataFrame(res)
                run_group['fmri_hyper']=temp.values.tolist()
                new_data = new_data.append(run_group, ignore_index=True)
    
        if sub[0:2] == 'ft':
            for run in runs2:
                run_group = sub_group.loc[sub_group['run_num'] == (run)].copy()
                fmri_data = run_group['features'].values.tolist()
                d = np.asmatrix(fmri_data)
                res = (d*transform[i]).A
                temp = pd.DataFrame(res)
                run_group['fmri_hyper']=temp.values.tolist()
                new_data = new_data.append(run_group, ignore_index=True)
        i=i+1
        
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
        
    return new_data

def hyper_score(data, sub_names, train_runs1, test_runs1, train_runs2, test_runs2, drop_fingers=[5], get_predictions=False, get_score=True, plot=False, timer=False):
    
    """
    obtain score from hyperaligned data
    function will train and test a decoder on selected runs
    decoder can be trained with or without the data used for for feature selection and alignment
    testing should only be done on the data not used for feature selection and alignment
    also included is an optional plot argument to plot confusion matrices for each subject
    """
    
    if timer:
        start = time.time()
        
    if get_predictions:
        get_score=False
    
    scores = []
    all_preds = []
    all_conds = []
    for sub in sub_names:
        sub_group = data.loc[data['subject'] == sub]

        # group data for testing
        fmri_test = []
        conditions_test = []
        if sub[0:2] == 'ff':
            for run in test_runs1:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['fmri_hyper'].values.tolist()
                fmri_test.append(fmri_data)
                conditions = run_group['probe'].values
                conditions_test.append(conditions)
        
        if sub[0:2] == 'ft':
            for run in test_runs2:
                run_group = sub_group.loc[sub_group['run_num'] == (run)]
                fmri_data = run_group['fmri_hyper'].values.tolist()
                fmri_test.append(fmri_data)
                conditions = run_group['probe'].values
                conditions_test.append(conditions)
        
        # leave-one-subject-out for training
        temp = []
        for i in sub_names:
            if (sub != i):
                temp.append(i)

        # group data for training
        fmri_train = []
        conditions_train = []
        for subj in temp:
            sub_group = data.loc[data['subject'] == subj]

            if subj[0:2] == 'ff':
                for run in train_runs1:
                    run_group = sub_group.loc[sub_group['run_num'] == (run)]
                    fmri_data = run_group['fmri_hyper'].values.tolist()
                    fmri_train.append(fmri_data)
                    conditions = run_group['probe'].values
                    conditions_train.append(conditions)
                    
            if subj[0:2] == 'ft':
                for run in train_runs2:
                    run_group = sub_group.loc[sub_group['run_num'] == (run)]
                    fmri_data = run_group['fmri_hyper'].values.tolist()
                    fmri_train.append(fmri_data)
                    conditions = run_group['probe'].values
                    conditions_train.append(conditions)

        fmri_train = np.vstack(fmri_train)
        fmri_test = np.vstack(fmri_test)
        conditions_train = np.hstack(conditions_train)
        conditions_test = np.hstack(conditions_test)
        all_conds.append(conditions_test)
        
        # use a linear support vector machine classifier for creating predictions
        svc = SVC(kernel='linear')
        pred = svc.fit(fmri_train, conditions_train).predict(fmri_test)
        all_preds.append(pred)
        score = accuracy_score(conditions_test, pred)
        scores.append(score)
        
        # plot confusion matrices for each subject
        if plot:
            if drop_fingers == [5]:
                print('%s score: %s' %(sub, scores))
                plot_confusion_matrix(conditions_test, pred, normalize=True, title = '%s hyperalignment: %0.6s' %(sub, score)) 
            else:
                print('%s score: %s' %(sub, scores))
                plot_confusion_matrix(conditions_test, pred, normalize=True, title = '%s drop finger %s: %0.6s' %(sub, drop_fingers, score))
            plt.show()
    
    # plot confusion matrix for all subjects
    preds = np.hstack(all_preds)
    conds = np.hstack(all_conds)
    predictions = []
    predictions.append(all_preds)
    predictions.append(all_conds)
    mean = np.mean(scores)
    if plot:
        if drop_fingers == [5]:
            plot_confusion_matrix(conds, preds, normalize=True, title = 'All subjects hyperalignment: %0.6s' %(mean))
        else:
            plot_confusion_matrix(conds, preds, normalize=True, title = 'All subjects drop finger %s: %0.6s' %(drop_fingers, mean))
        plt.show()  
    
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
        
    if get_predictions:
        return predictions
    if get_score:
        return scores
    
def cross_validate(data, sub_names, drop_fingers=[5], get_scores=True, get_matrix=False, plot=False, timer=False):
    
    '''
    perform a leave-one-run out cross-validation for feature selection, hyperalignment, and classification
    if get_matrix is set to true, this function will return the predictions for each trial rather than the accuracy scores
    '''
    
    if timer:
        start = time.time()
    
    if get_matrix:
        get_scores = False
        
    total_runs = np.arange(8)
    
    train_runs = []
    test_runs = []
    for test_run in total_runs:
        test_runs.append([test_run])
        temp = []
        for train_run in total_runs:
            if train_run != test_run:
                temp.append(train_run)
        train_runs.append(temp)
        
    # get predictions for a leave-one-run-out cross-validation
    predictions=[]
    for i in range(len(train_runs)):
        feat_trans = select_features(data=data, sub_names=sub_names, runs1=train_runs[i], runs2=train_runs[i], drop_fingers=drop_fingers)
        features_drop = apply_features(data=data, transform=feat_trans, sub_names=sub_names, runs1=total_runs, runs2=total_runs, drop_fingers=drop_fingers)
        features_all = apply_features(data=data, transform=feat_trans, sub_names=sub_names, runs1=total_runs, runs2=total_runs)
        hyper_trans = create_common_model(data=features_drop, sub_names=sub_names, runs1=train_runs[i], runs2=train_runs[i])
        hyper = hyperalign(data=features_all, transform=hyper_trans, sub_names=sub_names, runs1=total_runs, runs2=total_runs)
        prediction = hyper_score(data=hyper, sub_names=sub_names, train_runs1=train_runs[i], test_runs1=test_runs[i], train_runs2=train_runs[i], test_runs2=test_runs[i], drop_fingers=drop_fingers, get_predictions=True, plot=False)
        predictions.append(prediction)
        
    # get classifier scores for each sub
    all_preds = []
    all_conds = []
    array=[]
    sub_scores = []
    for j in range(len(sub_names)):
        sub = sub_names[j]
        sub_preds = []
        sub_conds = []
        for i in range(8):
            sub_preds.append(predictions[i][0][j])
            sub_conds.append(predictions[i][1][j])    

        sub_pred = np.hstack(sub_preds)
        sub_cond = np.hstack(sub_conds)
        score = accuracy_score(sub_pred, sub_cond)
        sub_scores.append(score)
        
        if plot:
            if drop_fingers == [5]:
                plot_confusion_matrix(sub_cond, sub_pred, normalize=True, title = '%s score: %0.6s' %(sub, score))
                plt.show()
            else:
                plot_confusion_matrix(sub_cond, sub_pred, normalize=True, title = '%s drop finger %s: %0.6s' %(sub, drop_fingers, score))
                plt.show()         

        cm = confusion_matrix(sub_cond, sub_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
        for i in range(4):
                array.append(cm[i][i])

        all_preds.append(sub_pred)
        all_conds.append(sub_cond)

    all_pred = np.hstack(all_preds)
    all_cond = np.hstack(all_conds)
    score = accuracy_score(all_pred, all_cond)
    
    if plot:
        if drop_fingers == [5]:
            plot_confusion_matrix(all_cond, all_pred, normalize=True, title = 'All subjects hyperalignment: %0.6s' %(score))
        else:
            plot_confusion_matrix(all_cond, all_pred, normalize=True, title = 'All subjects drop finger %s: %0.6s' %(drop_fingers, score))
        plt.show()
      
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
    
    if get_scores:
        return sub_scores
    if get_matrix:
        return array
    
def within_score(data, sub_names, session=1, plot=False, timer=False):
    
    """
    scoring function for within-subject classification using leave-one-run-out cross-validation
    session can be changed depending on which subjects you are testing on
    """    
    
    if timer:
        start = time.time()
        
    if session == 'both':
        total_runs = np.arange(16)
    if session == 1:
        total_runs = np.arange(8)
    if session == 2:
        total_runs = np.arange(8)
        total_runs = total_runs + 8
    
    # loop through each subject individually
    means = []
    all_preds = []
    all_conds = []
    for sub in sub_names:
        sub_group = data.loc[data['subject'] == sub]
        
        # loop though each run individually
        scores = []
        sub_conds = []
        sub_preds = []
        for test_run in total_runs:

            # loop though each run again to separate test and train data
            fmri_train= []
            conditions_train = []
            for train_run in total_runs:
                # separate test and train fMRI data
                if train_run != test_run:
                    run_group = sub_group.loc[sub_group['run_num'] == (train_run)]
                    fmri_data = run_group['fmri'].values.tolist()
                    fmri_train.append(fmri_data)
                    conditions = run_group['probe'].values
                    conditions_train.append(conditions)
                    
            # select features from the train data and apply to train data
            fmri_train = np.vstack(fmri_train)
            conditions_train = np.hstack(conditions_train)
            selection = SelectKBest(f_classif, k=100).fit(fmri_train, conditions_train)
            fmri_train = selection.transform(fmri_train)

            # apply features to test data
            run_group = sub_group.loc[sub_group['run_num'] == (test_run)]
            fmri_data = run_group['fmri'].values.tolist()
            fmri_test = selection.transform(fmri_data)
            conditions_test = run_group['probe'].values
            sub_conds.append(conditions_test)

            # use a linear support vector machine classifier for creating predictions
            svc = SVC(kernel='linear')
            pred = svc.fit(fmri_train, conditions_train).predict(fmri_test)
            sub_preds.append(pred)
            score = accuracy_score(conditions_test, pred)
            scores.append(score)
            
        sub_preds = np.hstack(sub_preds)
        sub_conds = np.hstack(sub_conds)
        
        all_preds.append(sub_preds)
        all_conds.append(sub_conds)
        
        # calculate the average cross-validated score for each subject
        mean = np.mean(scores)
        means.append(mean)
        
        # plot confusion matrices for each subject
        if plot:
            plot_confusion_matrix(sub_preds, sub_conds, normalize=True, title = '%s within-subject: %0.6s' %(sub, mean))
            plt.show()    
    
    all_preds = np.hstack(all_preds)
    all_conds = np.hstack(all_conds)
    # plot confusion matrix for all subjects
    if plot:
        grand_mean = np.mean(means)
        plot_confusion_matrix(all_preds, all_conds, normalize=True, title = 'All subjects within-subject: %0.6s' %(grand_mean))
        plt.show() 
    
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s)))  
        
    return means

def anatomical_score(data, sub_names, plot=False, timer=False):
    
    '''
    perform a leave-one-subject-out cross-validation on anatomically aligned data
    '''
    
    if timer:
        start = time.time()
    
    # make sure we are only using one session so the indexing doesn't mess up
    data = data.loc[data['session'] == '1']
    
    fmri_data = data['fmri']
    fmri_data = np.vstack(fmri_data)
    conditions = data['probe'].values
    conditions = np.hstack(conditions)
    
    # create a support vector classifier with linear kernel
    svc = SVC(kernel='linear')
    
    # do a leave one subject out cross validation 
    session_label = data['subject']
    cv = LeaveOneGroupOut()
    cv_score = cross_val_score(svc, fmri_data, conditions, cv = cv, groups = session_label)
    
    pred = cross_val_predict(svc, fmri_data, conditions, cv = cv, groups = session_label)
    if plot:
        for i in range(len(sub_names)):
            sub = sub_names[i]
            index = len(data.loc[data['subject'] == sub])
            upper = (i+1) * index
            lower = i * index
            plot_confusion_matrix(conditions[lower:upper], pred[lower:upper], normalize=True, title='%s anatomical: %0.6s' %(sub, cv_score[i]))
            plt.show()
        
        plot_confusion_matrix(conditions, pred, normalize=True, title='All subjects anatomical: %0.6s' %np.mean(cv_score))
    
    if timer:
        end = time.time()
        elapsed = end - start
        m, s = divmod(elapsed, 60)
        print('Time elapsed: %s minutes %s seconds' %(round(m), round(s))) 
    
    return cv_score

def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax