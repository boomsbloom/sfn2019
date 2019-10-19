import glob, os, ast, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bambi import Model
from scipy.stats import zscore
import pymc3 as pm
import edfreader as edfr
import pystan
import scipy.stats as st

palette = ["#66c2a5","#8da0cb"]

def load_eyetracking_data(data_dir,pids):
    msgs = {'iti':['state iti','state choice'],
        'choice':['state choice','Response at'],
        'confirm':['Response at','Confirmation end'],
        'isi':['state isi','state feedback'],
        'feedback':['state feedback','state iti']}

    eye_data = {}
    for pid in pids:
        trial_data = {}
        ascName = os.path.join(data_dir,pid,"INC%s.asc"%(pid))

        for msg in msgs.keys():
            f = edfr.read_edf(filename=ascName,
                              start=msgs[msg][0],
                              stop=msgs[msg][1],
                              missing=0.0,
                              debug=False)
            trial_data[msg] = f
        eye_data[pid] = trial_data
    
    return eye_data

def get_pupil_data(eye_data, pids):
    pupil_data = {'trial_number':[],'size':[],'is_blink':[],'time':[],'subject_id':[],'phase':[]}
    for pid in pids:
        trial_data = eye_data[pid]

        trial_phases = list(trial_data.keys())

        for phase in trial_phases:

            n_trials = np.shape(trial_data[phase])[0]
            for trial in range(n_trials):
                blks = np.array(trial_data[phase][trial]['events']["Eblk"])
                if len(blks) != 0:
                    starts = blks[:,0]
                    stops = blks[:,1]
                else:
                    starts = []
                    stops = []
                is_blk = False
                for i, t in enumerate(trial_data[phase][trial]['trackertime']):
                    size = trial_data[phase][trial]['size'][i]
                    if t in starts: is_blk = True
                    elif t in stops: is_blk = False

                    pupil_data['is_blink'].append(is_blk)
                    pupil_data['time'].append(t)
                    pupil_data['trial_number'].append(trial+1)
                    pupil_data['size'].append(size)
                    pupil_data['subject_id'].append(pid)
                    pupil_data['phase'].append(phase)

    pupil_data = pd.DataFrame.from_dict(pupil_data)
    
    return pupil_data

def preprocess_pupil_data(pupil_data):
    # preprocess data

    # exclude data within a blink
    pupil_data = pupil_data[pupil_data["is_blink"] == False]

    # remove all remaining zeros
    pupil_data = pupil_data[pupil_data["size"] != 0.0]

    # exclude outliers and zscore for each participant
    pupil_data_clean = []
    for pid in pupil_data['subject_id'].unique():

        pid_data = pupil_data[pupil_data.subject_id == pid]

        # exclude outliers
        upper = np.mean(pid_data["size"]) + 3*np.std(pid_data["size"])
        lower = np.mean(pid_data["size"]) - 3*np.std(pid_data["size"])

        pid_data = pid_data[(pid_data["size"] > lower) & (pid_data["size"] < upper)]
        
        # perform subtractive baseline correction
        base_df = {'trial_number':[],'baseline_mean':[],'baseline_median':[]}
        for trial in pid_data['trial_number'].unique():

            trial_data = pid_data[pid_data.trial_number == trial]

            iti_data = trial_data[trial_data.phase == "iti"]
            base_df['baseline_mean'].append(np.mean(iti_data[-10:]["size"]))
            base_df['baseline_median'].append(np.median(iti_data[-10:]["size"]))
            base_df['trial_number'].append(trial)
            
        base_df = pd.DataFrame.from_dict(base_df)

        pid_data = pid_data.merge(base_df,on="trial_number")
        pid_data["size_mean_diff"] = pid_data["size"] - pid_data["baseline_mean"]
        pid_data["size_median_diff"] = pid_data["size"] - pid_data["baseline_median"]
        
        # z score within each trial
        t_data = []
        for trial in pid_data['trial_number'].unique():
            trial_data = pid_data[pid_data.trial_number == trial]
            trial_data["sizeZbyTrial"] = zscore(trial_data['size'])
            t_data.append(trial_data)
        pid_data = pd.concat(t_data)
        
        # z score pupil size within participant (whole exp)
        pid_data['sizeZ'] = zscore(pid_data['size'])

        pupil_data_clean.append(pid_data)

    pupil_data = pd.concat(pupil_data_clean)
    
    return pupil_data

def plot_3way_pupils(fit_model=None, mean_pupil_data=None, old=0):

    n_bins = 6
    y_var = "meanPupilZ"
    col_var = "environment"
    col_factors = [1,0]
    plot_order = [0,1]
    x_var = "t_since_reversal"
    x_factors = [np.arange(0,30,5),np.arange(0,55,5)]
    x_scaler = [0.2010,0.10]
    main_factors = [165,160]

    col_levels = list(mean_pupil_data[col_var].unique())
    titles = [l.capitalize() for l in col_levels]

    _, axes = plt.subplots(1,2,figsize=(12, 6))

    trace_df = fit_model.to_df(ranefs=True)

    for pid in mean_pupil_data.subject_id.unique():
        for col_i, col in enumerate(col_factors):
            mean_ps = [] 
            for i, x in enumerate(x_factors[col_i]):
                y = trace_df["Intercept"] + trace_df["1|subject_id[%s]"%pid] + \
                     x*trace_df["%s"%x_var] + x*trace_df["%s|subject_id[%s]"%(x_var,pid)] + \
                     main_factors[col_i]*trace_df["trial_number"] + main_factors[col_i]*trace_df["trial_number|subject_id[%s]"%(pid)] + \
                     col*trace_df["%s[T.%s]"%(col_var,col_levels[1])] + col*trace_df["%s[T.%s]|subject_id[%s]"%(col_var,col_levels[1],pid)] + \
                     x*col*trace_df["%s:%s[T.%s]"%(x_var,col_var,col_levels[1])] + \
                     x*col*trace_df["%s:%s[T.%s]|subject_id[%s]"%(x_var,col_var,col_levels[1],pid)] + \
                     old*trace_df["old_trial[T.True]"] + old*trace_df["old_trial[T.True]|subject_id[%s]"%pid] + \
                     old*col*trace_df["%s[T.%s]:old_trial[T.True]"%(col_var,col_levels[1])] + old*col*trace_df["%s[T.%s]:old_trial[T.True]|subject_id[%s]"%(col_var,col_levels[1],pid)] + \
                     old*x*trace_df["%s:old_trial[T.True]"%(x_var)] + old*x*trace_df["%s:old_trial[T.True]|subject_id[%s]"%(x_var,pid)] + \
                     old*x*col*trace_df["%s:%s[T.%s]:old_trial[T.True]"%(x_var,col_var,col_levels[1])] + \
                     old*x*col*trace_df["%s:%s[T.%s]:old_trial[T.True]|subject_id[%s]"%(x_var,col_var,col_levels[1],pid)]

                mean_ps.append(np.mean(y))

            axes[plot_order[col_i]].plot(x_factors[col_i]*x_scaler[col_i],mean_ps,color=palette[col_i],linewidth=2,alpha=0.4)


    for col_i, col in enumerate(col_factors):
        mean_ps = []
        for i, x in enumerate(x_factors[col_i]):
            y = trace_df["Intercept"]  + \
                 main_factors[col_i]*trace_df["trial_number"] + \
                 x*trace_df["%s"%x_var] + \
                 col*trace_df["%s[T.%s]"%(col_var,col_levels[1])] + \
                 x*col*trace_df["%s:%s[T.%s]"%(x_var,col_var,col_levels[1])] + \
                 old*trace_df["old_trial[T.True]"] + \
                 old*col*trace_df["%s[T.%s]:old_trial[T.True]"%(col_var,col_levels[1])] + \
                 old*x*trace_df["%s:old_trial[T.True]"%(x_var)] + \
                 old*x*col*trace_df["%s:%s[T.%s]:old_trial[T.True]"%(x_var,col_var,col_levels[1])]
            
            mean_ps.append(np.mean(y))

        with plt.rc_context({'lines.solid_capstyle': 'butt'}):
            axes[plot_order[col_i]].plot(x_factors[col_i]*x_scaler[col_i],mean_ps,color=palette[col_i],linewidth=8)

    point_kwargs = {'join':False,"hue_order":col_levels,"errwidth":5,"scale":2,"palette":[".75"]}
    tick_labels = [np.linspace(0,25,6),np.linspace(0,50,6)]
    titles = ["Low","High"]
    for i_ax, ax in enumerate(axes):
        col_data = mean_pupil_data[mean_pupil_data[col_var] == col_levels[col_factors[plot_order[i_ax]]]]
        col_data[x_var] = list(pd.cut(col_data[x_var],bins=n_bins,labels=range(n_bins)))

        if old == 0: plt_old = False
        elif old == 1: plt_old = True
        sns.pointplot(x=x_var,y=y_var,data=col_data[col_data.old_trial == old],ax=ax,**point_kwargs)

        plt.setp(ax.collections, facecolor="w",zorder=100)

        ax.set_ylabel("Pupil Size (Z-scored)")
        ax.set_xlabel("Trials Since Reversal")
        ax.set_title(titles[col_factors[plot_order[i_ax]]])
        ax.set_xticklabels([int(x) for x in tick_labels[i_ax]])
        ax.set_ylim((-1,0.75))

    for col_i, col in enumerate(col_factors):
        lows, highs = [], []
        for i, x in enumerate(x_factors[col_i]):
            y = trace_df["Intercept"]  + \
                 main_factors[col_i]*trace_df["trial_number"] + \
                 x*trace_df["%s"%x_var] + \
                 col*trace_df["%s[T.%s]"%(col_var,col_levels[1])] + \
                 x*col*trace_df["%s:%s[T.%s]"%(x_var,col_var,col_levels[1])] + \
                 old*trace_df["old_trial[T.True]"] + \
                 old*col*trace_df["%s[T.%s]:old_trial[T.True]"%(col_var,col_levels[1])] + \
                 old*x*trace_df["%s:old_trial[T.True]"%(x_var)] + \
                 old*x*col*trace_df["%s:%s[T.%s]:old_trial[T.True]"%(x_var,col_var,col_levels[1])]

            hpd = pm.stats.hpd(y)
            lows.append(hpd[0])
            highs.append(hpd[1])
        axes[plot_order[col_i]].fill_between(x_factors[col_i]*x_scaler[col_i], lows, highs, color=palette[col_i], alpha=0.15, linewidth=0)

    plt.tight_layout()
    sns.despine();
    
    return _