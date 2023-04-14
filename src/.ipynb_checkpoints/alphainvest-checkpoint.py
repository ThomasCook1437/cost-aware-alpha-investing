import os
import numpy as np
import scipy.stats as ss

# from gams import *

import pandas as pd
import math
import uuid
import sys
sys.path.append(r'/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx/apifiles/Python/api_39')
sys.path.append(r'/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx/apifiles/Python/gams')
from gams import *

######### Varphi Investment Schemes #########

#Constant
def constant_spending_scheme(param_dict):

    param_dict['varphi'] = np.min([param_dict['rate']*param_dict['Walpha0'], param_dict['Walpha']])
    
    return param_dict

#Relative
def relative_spending_scheme(Walpha, rate=0.1):
    
    varphi = .1 * Walpha
    
    return varphi


#Relative200
# Note relative 200 is redundant, a maximum number of tests
# parameter will be used in simulations instead of implementing
# a separate scheme.

#############################################


######### Alpha, Psi Investing Methods #########

#Alpha spending
def alpha_spending(varphi):
    return {'alpha' : varphi, 'psi' : 0}

#Foster and Stine alpha investing
def alpha_investing(param_dict):
    alpha = param_dict['varphi'] / (1 + param_dict['varphi'])
    param_dict['psi'] = param_dict['varphi'] + param_dict['omega']
    return param_dict

#alpha spending with rewards
def alpha_spending_with_rewards(varphi, Walpha0, k, rho):
    alpha = varphi / k
    psi = np.min([alpha / rho + Walpha0/k , 1 - ((1 - Walpha0)/k)])
    
    return {'alpha': alpha, 'psi': psi}

#ERO
def ero_investing(varphi, model_name, muA, sigma, n_pure,
                  working_dir = "./gams/"):
        
    ws = GamsWorkspace(system_directory = "/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx",
                   debug=DebugLevel.Off,
                   working_directory = working_dir)

    cp = ws.add_checkpoint()
    job = ws.add_job_from_file(os.path.join("..", model_name + '.gms'))
    job.run(checkpoint=cp)

    mi = cp.add_modelinstance()
    phi_db = mi.sync_db.add_parameter("phi", varphi, "cost of the test")
    mua_db = mi.sync_db.add_parameter("muA", muA, "alternative mean")
    sigma_db = mi.sync_db.add_parameter("sigma", sigma, "std dev of data")
    n_db = mi.sync_db.add_parameter("n", n_pure, "predetermind num. samples")
    opt = ws.add_options()
    opt.all_model_types = 'conopt'
                                 
    mi.instantiate("ero using nlp maximizing er1", 
                   [GamsModifier(phi_db),GamsModifier(mua_db),
                    GamsModifier(sigma_db),GamsModifier(n_db)], opt)
    phi_db.add_record().value = varphi
    mua_db.add_record().value = muA
    sigma_db.add_record().value = sigma
    n_db.add_record().value = n_pure
    
    mi.solve()
    assert mi.model_status < 3, "Did not solve"
    
    return {'alpha' : mi.sync_db.get_variable("alpha").find_record().level,
            'psi' : mi.sync_db.get_variable("psi").find_record().level}

#############################################


######### Cost-Aware  ######### 

#cost-aware

######################################################

######### LORD and SAFFRON ######### 
#easier to not split these rules!

#LORD1

#LORD2

#LORD3

#LORD++

#SAFFRON

######################################################


# def constant_spending_rule(Walpha, Walpha0, rate=0.1, omega=0.05):

#     varphi = np.min([rate*Walpha0, Walpha])
#     alpha = varphi/(1+varphi)
#     psi = varphi + omega

#     return {'alpha':alpha, 'varphi':varphi, 'psi':psi}

def norm_test(y, alpha, varphi, n = 1):
    
    (y_avg, y_sd, y_n) = (np.mean(y), np.std(y), len(y))

    R = 0
    p_val = 1-ss.norm.cdf( y_avg*np.sqrt(y_n)/y_sd, loc=0, scale=1) 
    if p_val < alpha:
        R = 1

    return {'R': R, 'p_val': p_val}



# def main():
    
#     alpha0 = 0.05
#     eta = 1 - alpha0

#     rate = 0.1

#     method_dict = {'constant investing' : {'spending scheme' : constant_spending_scheme,
#                                         'investment method' : alpha_investing,
#                                         'max_tests' : 1000},
#                 'relative investing' : {'spending scheme' : relative_spending_scheme,
#                                         'investment method' : alpha_investing,
#                                         'max_tests' : 1000},
#                 'relative200 investing' : {'spending scheme' : relative_spending_scheme,
#                                         'investment method' : alpha_investing,
#                                         'max_tests' : 200},
                
#                 'constant ero' : {'spending scheme' : constant_spending_scheme,
#                                         'investment method' : alpha_investing,
#                                         'max_tests' : 1000},
#                 'relative ero' : {'spending scheme' : relative_spending_scheme,
#                                         'investment method' : alpha_investing,
#                                         'max_tests' : 1000},
#                 'relative200 ero' : {'spending scheme' : relative_spending_scheme,
#                                         'investment method' : alpha_investing,
#                                         'max_tests' : 1000}}
        
#     for key in method_dict:
#         Walpha = [alpha0*eta,]
#         R = []
#         for i in range(10):
#             np.random.seed(i)
#             y = np.random.normal(size=2)
#             bet = method_dict[key]['spending scheme'](Walpha[-1], Walpha[0], rate)
#             res = norm_test(y, **bet)
            
#             R.append(res['R'])
#             Walpha.append( Walpha[-1] -bet['varphi'] + res['R']*bet['psi'])
    
#             print(f"{bet['varphi']:.4f} {bet['psi']:.4f} {Walpha[-1]:.4f}")
    
#             if Walpha[-1] <= 0:
#                 break
#         print(R)
        

class investing_rule:
    def __init__(self, spending_func, investing_func, 
                Walpha, Walpha0, rate = 0.1):
        self.spending_func = spending_func
        self.investing_func = investing_func
        self.initialize_params(Walpha, Walpha0, rate)
        
    def initialize_params(self):
        if self.spending_func == constant_spending_scheme:
            self.spending_params = {'Walpha' : alpha * eta, 
                                    'Walpha0' : alpha * eta, 
                                    'rate' : 0.1,
                                    'varphi' : 0}
        elif method_func == relative_spending_scheme:
            raise NotImplementedError
    
        else:
            raise NotImplementedError    
            
            
        if self.investing_func == alpha_investing:
            self.investing_params =  {'omega' : 0.05, 
                                      'alpha' : 0, 
                                      'psi' : 0}
            
        self.params = {self.spending_params, self.investing_params}
    def update_params(self, Walpha, Wdollar, R, i):
        self.params['Walpha'] = Walpha
        self.params['Wdollar'] = Wdollar
    
    def __next__(self):
        self.spending_func(self)
        
    def
    
    
def init_spending_dict(method_func, alpha = 0.05, rate = 0.1, gamma = None):
    
    eta = 1 - alpha
    
    if method_func == constant_spending_scheme:
        return {'Walpha' : alpha * eta, 
                'Walpha0' : alpha * eta, 
                'rate' : 0.1,
                'varphi' : 0,
                }
    
    elif method_func == relative_spending_scheme:
        return {}
    
    else:
        raise NotImplementedError

def init_investing_dict(method_func):
    
    if method_func == alpha_investing:
        return {'omega' : 0.05,
                'alpha' : 0,
                'psi' : 0, 
                    }
    



def temp_main():
    #np.random.seed(i)
    method_dict = {'constant investing' : {'spending scheme' : constant_spending_scheme,
                                        'investing method' : alpha_investing,
                                        'max_tests' : 1000},
                # 'relative investing' : {'spending scheme' : relative_spending_scheme,
                #                         'investment method' : alpha_investing,
                #                         'max_tests' : 1000},
                # 'relative200 investing' : {'spending scheme' : relative_spending_scheme,
                #                         'investment method' : alpha_investing,
                #                         'max_tests' : 200},
                
                # 'constant ero' : {'spending scheme' : constant_spending_scheme,
                #                         'investment method' : alpha_investing,
                #                         'max_tests' : 1000},
                # 'relative ero' : {'spending scheme' : relative_spending_scheme,
                #                         'investment method' : alpha_investing,
                #                         'max_tests' : 1000},
                # 'relative200 ero' : {'spending scheme' : relative_spending_scheme,
                #                         'investment method' : alpha_investing,
                #                         'max_tests' : 1000}
                }
    
    
    alpha0 = 0.05
    eta = 1 - alpha0
    n_pure = 1

    result = {}
    
    for method in method_dict:
        # Initialize wealths and rejections
        Walpha = [alpha0*eta, ]
        Wdollar = [1000,]
        R = []
        varphi = []
        alpha = []
        psi = []
        
        # Initialize spending and investings schemes
        spending_dict = init_spending_dict(method_dict[method]['spending scheme'])
        investing_dict = init_investing_dict(method_dict[method]['investing method'])
        param_dict = {**spending_dict, **investing_dict}
        
        for i in range(method_dict[method]['max_tests']):
            #Sample the data
            y = np.random.normal(size=2)
            
            #Spend and invest wealth
            param_dict = method_dict[method]['spending scheme'](param_dict)
            param_dict = method_dict[method]['investing method'](param_dict)
            
            varphi.append(param_dict['varphi'])
            alpha.append(param_dict['alpha'])
            psi.append(param_dict['psi'])
            
            if method == 'cost_aware_ero':
                n = param_dict['n']
            else:
                n = n_pure
            
            #Perform test
            res = norm_test(y, alpha[-1], n)
            
            #Update wealth, rejections
            R.append(res['R'])
            Walpha.append(Walpha[-1] - varphi[-1] + R[-1]*psi[-1])
            Wdollar.append(Wdollar[-1] - n)
            
            #Update params
            param_dict = update_param_dict(Walpha[-1], Wdollar[-1],
                                       R[-1], i, 
                                       param_dict) 
            
            #Check to see if we have exceeded wealths or num iters.
            if Walpha[-1] <= 0 or Wdollar[-1] <= 0:
                break
        
        result[method] = {'Walpha' : Walpha,
                          'Wdollar': Wdollar,
                          'R' : R,
                          'varphi' : varphi,
                          'alpha' : alpha,
                          'psi' : psi}
                          
    print(result)
    return result
            
            
            
if __name__== '__main__':
    temp_main()




