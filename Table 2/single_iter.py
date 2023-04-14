#Standard imports
import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.stats.weightstats import ztest as ztest
import pandas as pd
import math
import uuid
import tqdm
import shutil
import time

#Import src package
import os
import sys
sys.path.insert(0, '/home/tjcook/alpha-investing-r1')
import src

sys.path.append(r'/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx/apifiles/Python/api_39')
sys.path.append(r'/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx/apifiles/Python/gams')
from gams import *

import sqlite3

import stat

def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='One iteration of caero n = 1 for table 1.')
    parser.add_argument('--dataset_num', metavar='iter_', type=int, action='store', default=1,
                    help='Data set number (1-10000).', required=True)
    
    args = parser.parse_args()
    k = args.dataset_num
    
    conn = sqlite3.connect("results.db")
    
    max_size = 1000 #Number of sequential tests per iter
    mu = 2 #alternative mean

    #Initialize mFDR alpha0, eta
    alpha0 = 0.05
    eta = 1 - alpha0

    results = pd.DataFrame(columns = ['method','ntest', 'tr', 'fr', 'nskip', 'nsample', 'nsample_tot',
    'true_alt_tested', 'true_alt_total', 'nsample_rej'])

    #Set what methods will be used along with corresponding kwargs
    method_dict = {
                'constant__spending' : {'spending_func' : 'constant',
                                             'investing_func': 'alpha spending',
                                             'alpha' : alpha0, 
                                             'eta' : eta,
                                             'n_pure' : 1,
                                        'Wdollar': 1000,
                                        },
    
              'relative__spending' : {'spending_func' : 'relative',
                                             'investing_func': 'alpha spending',
                                             'alpha' : alpha0, 
                                             'eta' : eta, 
                                             'n_pure' : 1,
                                        'Wdollar': 1000,
                                        }, 
                'constant__investing' : {'spending_func' : 'constant',
                                             'investing_func': 'alpha investing',
                                             'alpha' : alpha0, 
                                             'eta' : eta,
                                             'n_pure' : 1,
                                        'Wdollar': 1000,
                                        },
    
              'relative__investing' : {'spending_func' : 'relative',
                                             'investing_func': 'alpha investing',
                                             'alpha' : alpha0, 
                                             'eta' : eta, 
                                             'n_pure' : 1,
                                        'Wdollar': 1000,
                                        },      
            'constant__rewardsk1' : {'spending_func' : 'constant',
                                             'investing_func': 'alpha rewards',
                                             'alpha' : alpha0, 
                                             'eta' : eta,
                                             'n_pure' : 1,
                                             'k' : 1,
                                        'Wdollar': 1000,
                                        },
    
              'relative__rewardsk1' : {'spending_func' : 'relative',
                                             'investing_func': 'alpha rewards',
                                             'alpha' : alpha0, 
                                             'eta' : eta, 
                                             'n_pure' : 1,
                                            'k' : 1,
                                        'Wdollar': 1000,
                                        },      
            'constant__rewardsk1.1' : {'spending_func' : 'constant',
                                             'investing_func': 'alpha rewards',
                                             'alpha' : alpha0, 
                                             'eta' : eta,
                                             'n_pure' : 1,
                                             'k' : 1.1,
                                        'Wdollar': 1000,
                                        },
    
              'relative__rewardsk1.1' : {'spending_func' : 'relative',
                                             'investing_func': 'alpha rewards',
                                             'alpha' : alpha0, 
                                             'eta' : eta, 
                                             'n_pure' : 1,
                                            'k' : 1.1,
                                        'Wdollar': 1000,
                                        },      
            'constant__ero_investing' : {'spending_func' : 'constant',
                                             'investing_func': 'ero investing',
                                             'alpha' : alpha0, 
                                             'eta' : eta,
                                             'model_name' : 'ero', 
                                             'n_pure' : 1,
                                             'working_dir' : './gams'+str(k)+'/',
                                        'Wdollar': 1000,
                                        },

              'relative__ero_investing' : {'spending_func' : 'relative',
                                             'investing_func': 'ero investing',
                                             'alpha' : alpha0, 
                                             'eta' : eta,
                                             'model_name' : 'ero', 
                                             'n_pure' : 1,
                                             'working_dir' : './gams'+str(k)+'/',
                                        'Wdollar': 1000,
                                        },
        
        'costaware_relative1': {'spending_func' : 'costaware',
                                           'investing_func': 'costaware',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'model_name' : 'caero_relative', 
                                           'working_dir' : './caero_relative_'+str(k)+'/',
                                           'n_pure' :1, 
                                           'n_upper' : 1,
                                            'Wdollar': 1000,},
        'costaware_relative10': {'spending_func' : 'costaware',
                                           'investing_func': 'costaware',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'model_name' : 'caero_relative', 
                                           'working_dir' : './caero_relative_'+str(k)+'/',
                                           'n_pure' :None, 
                                           'n_upper' : 10,
                                            'Wdollar': 1000,},
        'costaware_relative100': {'spending_func' : 'costaware',
                                           'investing_func': 'costaware',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'model_name' : 'caero_relative', 
                                           'working_dir' : './caero_relative_'+str(k)+'/',
                                           'n_pure' :None, 
                                           'n_upper' : 100,
                                            'Wdollar': 1000,},
        'costaware_relativestar': {'spending_func' : 'costaware',
                                           'investing_func': 'costaware',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'model_name' : 'caero_relative', 
                                           'working_dir' : './caero_relative_'+str(k)+'/',
                                           'n_pure' :None, 
                                           'n_upper' : 1000,
                                            'Wdollar': 1000,},
        
        'lord1': {'spending_func' : 'lord1',
                                           'investing_func': 'lord1',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'n_pure' :1, 
                                            'Wdollar': 1000,},
        'lord2': {'spending_func' : 'lord2',
                                           'investing_func': 'lord2',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'n_pure' :1, 
                                            'Wdollar': 1000,},
        'lord3': {'spending_func' : 'lord3',
                                           'investing_func': 'lord3',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'n_pure' :1, 
                                            'Wdollar': 1000,},
        'lord++': {'spending_func' : 'lord++',
                                           'investing_func': 'lord++',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'n_pure' :1, 
                                            'Wdollar': 1000,},
        'saffron': {'spending_func' : 'saffron',
                                           'investing_func': 'saffron',
                                           'alpha' : alpha0, 
                                           'eta' : eta,
                                           'n_pure' :1, 
                                            'Wdollar': 1000,},
    }

    #wealth_df keeps track of alpha-wealth vs test for each method. Although not
    #used in the paper it's something we still record.
    wealth_df = pd.DataFrame(columns = ['method', 'iter', 'wealth_list'])
    
    #Set seed for reproducibility
    np.random.seed(int(k))
    
    #Initialize the data vector
    y = np.zeros((max_size, 1000))
    
    #Sample the data 
    alt_true = [] #Indicator for true alternative 
    
    for j in range(max_size):
        y_alt_true = int(np.random.uniform() >= 0.9)
        alt_true.append(y_alt_true)
        if y_alt_true == 1:
            y[j] = np.random.normal(loc=mu, size = 1000)
        else:
            y[j] = np.random.normal(size = 1000)

    #Use each multiple testing method on this data
    for method in method_dict:
        # Initialize wealth, rejections, and parameters
        Walpha = [alpha0*eta, ]
        Wdollar = [1000,] 
        R = []
        varphi = []
        alpha = []
        psi = []
        n_list = []

        # Initialize spending and investings schemes
        rule = src.procedures.investing_rule(**method_dict[method])
        
        # Due to the way our gams code works, we make sure each iteration has it's own
        # working directory to prevent conflicts when running iterations in parallel.
        if 'working_dir' in method_dict[method]:
            rule.temp_gms = method + "_temp_" + str(k)
        
        # In case a relative constraint is added to a method that typically doesn't have one.
        print("beginning method: ", method)
        if Walpha[0] != rule.Walpha0:
            Walpha[0] = rule.Walpha0
        
        # Run the decision rule. An error will be thrown if not possible.
        for i in range(max_size):
            try:
                rule.step(q = 0.9)
            except:
                print("method: ", method, " couldn't find a solution after ", i, " tests.")
                if 'working_dir' in method_dict[method]:
                    #Delete gams directory if done. Adding a brief sleep resulted in 
                    # more frequent successful removal when working in parallel.
                    time.sleep(0.1)
                    shutil.rmtree(method_dict[method]['working_dir'], ignore_errors=True)
                break
            
            # Keep a list of values here
            varphi.append(rule.varphi)
            alpha.append(rule.alpha)
            psi.append(rule.psi)
            n = int(rule.n)
            
            # Perform the test
            if n == 1:
                res = src.procedures.norm_test(y[i][0], rule.alpha, n)
            else:
                res = src.procedures.norm_test(y[i][0:n], rule.alpha, n)
            
            
            #When using SAFFRON this is needed
            if method == 'saffron':
                if res['p_val'] < rule.lbd:
                    rule.candidates.append(1)
                else:
                    rule.candidates.append(0)
        
                if res['p_val'] < rule.alpha:
                    res['R'] = 1
                else:
                    res['R'] = 0
            
                if res['R'] == 1:
                    if rule.first == 0:
                        rule.first = 1
                        rule.flag = 1
                    rule.last_rej = np.append(rule.last_rej, i+1).astype(int)
            
                varphi[-1] = (1 - rule.candidates[-1])*rule.alpha
                psi[-1] = (1 - rule.lbd)*rule.alpha0 - rule.flag*rule.w0
            
            # Update wealth, rejections
            R.append(res['R'])
            Walpha.append(Walpha[-1] - varphi[-1] + R[-1]*psi[-1])
            Wdollar.append(Wdollar[-1] - n) 

            # Update params method. This updates the values necessary to correctly use
            # the decision rule's step method for the next test.
            rule.update_params(Walpha[-1], Wdollar[-1],
                                       R, i) 
            
            #Delete the gams directory, same as above.
            if 'working_dir' in method_dict[method]:
                time.sleep(0.01)
                shutil.rmtree(method_dict[method]['working_dir'],ignore_errors=True)
            
            #Check to see if we have exceeded wealths or num iters.
            # print(Wdollar[-1])
            if Walpha[-1] <= 0 or Wdollar[-1] <= 0:
                print("Wealth depleted after ", i+1, " tests. W alpha :", Walpha[-1], ", Wdollar: ", Wdollar[-1])
                break
        
        # Add results
        results = pd.concat([results,
                             pd.DataFrame({'method' : method,
                                 **src.metrics.compute_metrics_ca(np.array(R),np.array(alt_true)[:len(R)],
                                                                eta, n=np.ones_like(R),median = False)},
                                                                  index=[k])])
    
        wealth_str = ",".join([str(i) for i in Walpha])
        dollar_str = ",".join([str(i) for i in Wdollar])
        
        #Keep track of dollar and wealth 
        wealth_df = pd.concat([wealth_df, pd.DataFrame({'method': method,
                                                        'iter' : k,
                                                        'wealth_list' : wealth_str,
                                                        'dollar_list' : dollar_str}, index = [k])])
                                                        
    # When complete, add this iteration to the db
    results.to_sql('tbl1', con = conn, if_exists = 'append')
    wealth_df.to_sql("wealth", con= conn, if_exists = 'append')