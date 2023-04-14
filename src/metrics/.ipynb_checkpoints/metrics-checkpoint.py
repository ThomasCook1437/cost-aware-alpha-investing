import numpy as np
import pandas as pd
import scipy.stats as ss

def compute_metrics_real(R, n):
    ntest = np.sum(R != None)
    nreject = np.sum(R[R != None])
    num_skipped = np.sum(R == None)
    if len(n[n!= None]) < 1:
        nsample = 0
        nsample_tot = 0
        nsample_rej = 0
    else:
        nsample = np.nanmean(n[n != None])
        nsample_tot = np.sum(n[n != None])
        if nreject > 0:
            nsample_rej = np.nanmean(n[R == 1])
        else:
            nsample_rej = None
    return {'ntest': ntest,
            'nskip' : num_skipped,
            'nreject': nreject,
            'nsample': nsample,
            'nsample_tot': nsample_tot,
            'nsample_rej': nsample_rej
           } 
    
def choose_n(n):
    (low, high) = (int(n), int(n)+1)
    mantissa = n - low
    return np.random.choice([low, high], p = [1-mantissa, mantissa])

def compute_metrics(R, alt_true, eta): 
    ntest = len(R)
    V = R * (1-alt_true)
    true_reject = sum(R * (alt_true))
    false_reject = sum(R * (1-alt_true))
    return {'ntest': ntest, 'tr': true_reject, 'fr': false_reject,'nskip' : 0, 
            'nsample' : 1,
            'nsample_tot' : ntest,
            'true_alt_tested': np.sum(alt_true), 
            'true_alt_total': np.sum(alt_true),
            'nsample_rej' : 1}

def compute_metrics_ca(R, alt_true, eta, n, median = False):
    ntest = np.sum(R != None)
    num_skipped = np.sum(R == None)
    if num_skipped > 0:
        # print("WE SKIPPED")
        R[R == None] = 0
    V = R * (1-alt_true)
    true_reject = sum(R * (alt_true))
    false_reject = sum(R * (1-alt_true))
    
    nsample_rej = None
    if true_reject + false_reject >0:
        nsample_rej = np.nanmean(n[R == 1])
    
    if median:
        return {'ntest': ntest, 'tr': true_reject, 'fr': false_reject, 
            'nskip' : num_skipped, 'nsample' : np.mean(n[n != None]), 'nsample_med' : np.median(n[n != None]), 
            'nsample_tot' : np.sum(n[n != None]),
            'true_alt_tested': np.sum(alt_true[n != None]),
            'true_alt_total': np.sum(alt_true),
            'nsample_rej' : nsample_rej
           }
    else:
        return {'ntest': ntest, 'tr': true_reject, 'fr': false_reject, 
            'nskip' : num_skipped, 'nsample' : np.mean(n[n != None]), 
            'nsample_tot' : np.sum(n[n != None]),
            'true_alt_tested': np.sum(alt_true[n != None]),
            'true_alt_total': np.sum(alt_true),
            'nsample_rej' : nsample_rej
           }
    
    
def compute_metrics_cab(R, alt_true, eta, n):
    batch_skipped, true_reject, false_reject = 0, 0, 0
    num_skipped = np.zeros(len(R))
    num_samples = np.zeros(len(R))
    power_denom_count_alt = 0 # not counting when n=0 as the test is not run
    for i in range(len(R)):
        if np.array((R[i])).all() == None:
            batch_skipped += 1
        else:
            num_skipped[i] = np.bincount(n[i].astype(int))[0]
            num_samples[i] = np.mean(n[i])
            true_reject += sum(R[i] * alt_true[i])
            false_reject += sum(R[i] * (1-alt_true[i]))
            power_denom_count_alt += sum(np.bincount(alt_true[i].astype(int) * n[i].astype(int))[1:])
    #power = true_reject / (power_denom_count_alt + 0.95)
    return {'btest': len(R)-batch_skipped, 'tr': true_reject, 'fr': false_reject,
            'nskip' : np.mean(num_skipped), 'nsample' : np.mean(num_samples),
            'nsample_tot' : np.sum(n[n != None]),
            'bskip' : batch_skipped, 'power_denom' : power_denom_count_alt}
    

def choose_n_array(n):
    for i in range(len(n)):
        n[i] = choose_n(n[i])
    return n 

    
def filter_twostep(result_dict):
    n_seq = result_dict['n']
    
    if n_seq[-1] == None:
        for i in np.arange(len(n_seq)-1, 0, -1):
            if n_seq[i] != None:
                end_ind = i+1
                break
        return (result_dict['R'][:end_ind], result_dict['n'][:end_ind]) 
    return (result_dict['R'], result_dict['n'])
    
    
    