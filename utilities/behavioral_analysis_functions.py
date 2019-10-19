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


def loadLabData(pids=None,data_dir=None):

    dsets = []
    for pid in pids:
        pid_files = glob.glob(os.path.join(data_dir,pid,"*_stateData.csv"))
        pid_data = pd.concat([pd.read_csv(file) for file in pid_files])
        pid_data['subject_id'] = [pid] * len(pid_data)
        dsets.append(pid_data)
    full_data = pd.concat(dsets)
    n_subjects = len(full_data.subject_id.unique())
    return n_subjects, full_data

def loadMturkData(environments=None,data_dir=None):
    if len(environments) > 0:
        d_sets = []
        for env in environments:
            data_dir_env = os.path.join(data_dir,env)
            pid_files = glob.glob(os.path.join(data_dir_env,"*_experiment_data.csv"))
            env_data = pd.concat([pd.read_csv(file) for file in pid_files])
            if 'stabvol' in environments:
                env_data['condition'] = [env] * len(env_data)
            else:
                env_data['environment'] = [env] * len(env_data)
            d_sets.append(env_data)
        full_data = pd.concat(d_sets)
        n_subjects = len(full_data.subject_id.unique())
    
    return n_subjects, full_data

def cleanData(data=None):

    global t_counter
    global switch_counter
    switch_counter = 0
    def get_t_since_reversal(row):
        global t_counter
        if row.switch_trial or row.trial_number == 1: t_counter = 0
        else: t_counter+=1
        return t_counter

    def get_switch_number(row):
        global switch_counter
        if row.trial_number == 1: switch_counter = 0
        if row.switch_trial: switch_counter+=1
        return switch_counter

    data = data[data.phase == "choice"]
    data['t_since_reversal'] = data.apply(lambda row: get_t_since_reversal(row),axis=1)
    data['switch_number'] = data.apply(lambda row: get_switch_number(row),axis=1)
    data["old_value"] = data["old_value"].astype("float")
    data = data[data.choice != "no_response"]
    data["value"] = data["value"].astype("float")
    data["lucky_chosen"] = data["lucky_chosen"].astype("float")
    
    return data


def getOldData(data=None):

    old_data = data[data.old_trial == True]
    old_data["old_chosen"] = old_data["old_chosen"].astype("float")
    old_data["trial_number"] = old_data["trial_number"].astype("float")

    encoded_t_since_reversals, encoded_switch_numbers = [], []

    for pid in old_data.subject_id.unique():
        pid_data = data[data.subject_id == pid]
        pid_old_data = old_data[old_data.subject_id == pid]
        for i_row, row in pid_old_data.iterrows():
                old_trial_n = row.old_trial_number
                encoded_t_since_reversal = pid_data[pid_data.trial_number == old_trial_n].t_since_reversal.iloc[0]
                encoded_t_since_reversals.append(encoded_t_since_reversal)
                encoded_switch_number = pid_data[pid_data.trial_number == old_trial_n].switch_number.iloc[0]
                encoded_switch_numbers.append(encoded_switch_number)

    old_data['encoded_t_since_reversal'] = encoded_t_since_reversals
    old_data['encoded_switch_number'] = encoded_switch_numbers
    old_data['within_switch'] = np.where(old_data['encoded_switch_number'] == old_data['switch_number'], 1, 0)
    
    return old_data


def get_inc_choices(data=None):

    inc_choices = pd.DataFrame()
    inc_choices['orange_chosen'] = pd.factorize(data.choice)[0] #0 is blue, 1 is orange
    inc_choices['outcome_t'] = data.value.reset_index(drop=True)
    inc_choices['environment'] = data.environment.reset_index(drop=True)
    inc_choices['subject_id'] = data.subject_id.reset_index(drop=True)

    shifted_data = []
    for  pid in data.subject_id.unique():
        shifted_df = pd.DataFrame()
        pid_data = data[data.subject_id == pid]
        pid_inc_data = inc_choices[inc_choices.subject_id == pid]
        l, o = pd.factorize(pid_data.choice)
        if o[0] == "orange": l = np.abs(l - 1)
        pid_data['orange_chosen'] = l
        for i in range(1,5):
            shifted_df['outcome_t%s'%i] = pid_data.value.shift(i).reset_index(drop=True)
            shifted_df['orange_chosen_t%s'%i] = pid_data.orange_chosen.shift(i).reset_index(drop=True)
            shifted_df['environment_t%s'%i] = pid_data.environment.shift(i).reset_index(drop=True)
        shifted_data.append(shifted_df)
    shifted_data = pd.concat(shifted_data)
    inc_choices = pd.concat([inc_choices.reset_index(drop=True), shifted_data.reset_index(drop=True)], axis=1)

    # This participant is missing a lot of data due to failing to respond during the second half
    # Excluding does not change the effect.
    inc_choices = inc_choices[inc_choices.subject_id != "120"]
    
    return inc_choices


def signal_detect(row):
    if 'old' in row.response:
        if row.object_type == "old":
            return "hit"
        elif row.object_type == "new":
            return "fa"
    elif 'new' in row.response:
        if row.object_type == "old":
            return "miss"
        elif row.object_type == "new":
            return "cr"
    elif row.response == "dont_know":
        return "idk"
    
    
def compute_dprime(n_Hit=None,n_Miss=None,n_FA=None,n_CR=None):
    import scipy
    
    # Ratios
    hit_rate = n_Hit/(n_Hit + n_Miss)
    fa_rate = n_FA/(n_FA + n_CR)
    
    # Adjusted ratios
    hit_rate_adjusted = (n_Hit+ 0.5)/((n_Hit+ 0.5) + n_Miss + 1)
    fa_rate_adjusted = (n_FA+ 0.5)/((n_FA+ 0.5) + n_CR + 1)

    # dprime
    dprime = scipy.stats.norm.ppf(hit_rate_adjusted) - scipy.stats.norm.ppf(fa_rate_adjusted)
    
    return dprime

def plot_2way_logistic(data,ppc,model_fit,y_var,x_var,col_var,col_factors,x_factors,x_scaler,plot_order,x_label,y_label,n_bins=None,tick_labels=None,col_levels=None):
    '''
        Plots:
            - posterior predictive fit from model as thick line (not on poster)
            - Individual subject model estimates from the fit model
            - Mean data and 95% confidence intervals
    '''
    
    if col_levels == None: col_levels = list(data[col_var].unique())
    titles = [l.capitalize() for l in col_levels]
    titles = ["Low","High"]
    
    _, axes = plt.subplots(1,2,figsize=(12, 6))

    trace_df = model_fit.to_df(ranefs=True)
    
    for pid in data.subject_id.unique():
        for col_i, col in enumerate(col_factors):
            mean_ps = [] 
            for i, x in enumerate(x_factors[col_i]):
                xb = trace_df["Intercept"] + trace_df["1|subject_id[%s]"%pid] + \
                     x*trace_df[x_var] + x*trace_df["%s|subject_id[%s]"%(x_var,pid)] + \
                     col*trace_df["%s[T.%s]"%(col_var,col_levels[1])] + col*trace_df["%s[T.%s]|subject_id[%s]"%(col_var,col_levels[1],pid)] + \
                     x*col*trace_df["%s:%s[T.%s]"%(x_var,col_var,col_levels[1])] + \
                     x*col*trace_df["%s:%s[T.%s]|subject_id[%s]"%(x_var,col_var,col_levels[1],pid)]
                p = np.exp(xb) / (1 + np.exp(xb))
                mean_ps.append(np.mean(p))
                
            axes[plot_order[col_i]].plot(x_factors[col_i]*x_scaler[col_i],mean_ps,color=palette[col_i],linewidth=2,alpha=0.3)

    β = st.beta((ppc[y_var] == 1).sum(axis=0),
                (ppc[y_var] == 0).sum(axis=0))

    predictive_df = {x_var:list(data[x_var]),
                     'pred_out':list(list(β.mean()[:,0])),
                     col_var:list(data[col_var])}
    predictive_df = pd.DataFrame.from_dict(predictive_df)
    if n_bins == None: predictive_df[x_var] = predictive_df[x_var] * x_scaler[col_i]

    reg_kwargs = {'logistic':True,"truncate":True,"ci":None,"marker":"","line_kws":{"linewidth":8}}
    point_kwargs = {'join':False,"hue_order":col_levels,"errwidth":5,"scale":2,"palette":[".75"]}
    
    for i_ax, ax in enumerate(axes):
        
        col_pred_data = predictive_df[predictive_df[col_var] == col_levels[col_factors[plot_order[i_ax]]]]
        col_data = data[data[col_var] == col_levels[col_factors[plot_order[i_ax]]]]
        
        if n_bins != None:
            
            col_pred_data[x_var] = list(pd.cut(col_pred_data[x_var],bins=n_bins,labels=range(n_bins)))
            col_data[x_var] = list(pd.cut(col_data[x_var],bins=n_bins,labels=range(n_bins)))

#         with plt.rc_context({'lines.solid_capstyle': 'butt'}):
#             sns.regplot(x=x_var,y="pred_out",data=col_pred_data,color=palette[i_ax],ax=ax,**reg_kwargs);
        sns.pointplot(x=x_var,y=y_var,data=col_data,ax=ax,**point_kwargs)

        plt.setp(ax.collections, facecolor="w",zorder=50)
        ax.set_ylim((0.25,1.0))
        ax.set_xlim((-0.25,5.25))
        if n_bins != None: ax.set_xticklabels([int(x) for x in tick_labels[i_ax]])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(titles[col_factors[plot_order[i_ax]]])
        ax.legend_ = None

    plt.tight_layout()
    sns.despine();
    
    return _

def plot_2way_logistic_single(data,ppc,model_fit,y_var,x_var,col_var,col_factors,x_factors,x_scaler,plot_order,x_label,y_label,n_bins=None,tick_labels=None):
    '''
        Plots:
            - Group-level model-fit for each condition
    '''
    
    col_levels = ["stable","volatile"]
    titles = ["Low", "High"]
    
    _, ax = plt.subplots(1,1,figsize=(6, 6))

    trace_df = model_fit.to_df(ranefs=True)
    tick_labels = [0.0,0.2,0.4,0.6,0.8,1.0]
    for col_i, col in enumerate(col_factors):
        mean_ps, lows, highs = [], [], []
        for i, x in enumerate(x_factors[col_i]):
            xb = trace_df["Intercept"] + \
                 x*trace_df[x_var] + \
                 col*trace_df["%s[T.%s]"%(col_var,col_levels[1])] + \
                 x*col*trace_df["%s:%s[T.%s]"%(x_var,col_var,col_levels[1])]
            p = np.exp(xb) / (1 + np.exp(xb))
            hpd = pm.stats.hpd(p)
            mean_ps.append(np.mean(p))
            lows.append(hpd[0])
            highs.append(hpd[1])

        with plt.rc_context({'lines.solid_capstyle': 'butt'}):
            ax.plot(x_factors[col_i]*x_scaler[col_i], mean_ps, color=palette[col_i], linewidth=8)
            ax.fill_between(x_factors[col_i]*x_scaler[col_i], lows, highs, color=palette[col_i],alpha=0.2,linewidth=0)
            
#     point_kwargs = {'join':False,"hue_order":["volatile","stable"],"errwidth":5,"scale":2,"palette":palette}#,"palette":[".75"]}
#     sns.pointplot(x="old_value",y="old_chosen",data=data,hue="environment",dodge=0.1,**point_kwargs)

    ax.set_ylim((0.25,1.0))
    ax.set_xlim((-0.25,5.25))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xticks(range(6))
    ax.set_xticklabels(tick_labels)

    plt.tight_layout()
    sns.despine();
    
    return _