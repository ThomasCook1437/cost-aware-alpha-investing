import os
import numpy as np
import scipy.stats as ss

import pandas as pd
import math
import uuid
import sys
sys.path.append(r'/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx/apifiles/Python/api_39')
sys.path.append(r'/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx/apifiles/Python/gams')
from gams import *

def norm_test(y, alpha, n = 1, sigma = 1):
    
    (y_avg, y_sd, y_n) = (np.mean(y), sigma, n) #replaced np.std(y) with 1 (known) sigma

    R = 0
    p_val = 1-ss.norm.cdf( y_avg*np.sqrt(y_n)/y_sd, loc=0, scale=1) 
    if p_val < alpha:
        R = 1

    return {'R': R, 'p_val': p_val}       

class investing_rule:
    def __init__(self, spending_func, investing_func, alpha, eta,
                rate = 0.1, Wdollar = 1000, k = 1.0, omega = 0.05,
                gamma = np.ones(1000), muA = 2, sigma = 1, n_pure = 1,
               model_name = 'gams', working_dir =  "./gams/" , n_upper = None, temp_gms = '123',horizon = 1):
        self.initialize_funcs(spending_func, investing_func)
        self.initialize_params(alpha = alpha, eta = eta,
                rate = rate, Wdollar = Wdollar , k = k, omega = omega,
                gamma = gamma, muA = muA, sigma = sigma, n_pure = n_pure,
               model_name = model_name, working_dir =  working_dir, n_upper = n_upper, horizon = horizon)
        self.temp_gms = temp_gms
    
    def initialize_funcs(self, spending_func, investing_func):
        if spending_func == 'constant':
            self.spending_func = self.constant_spending_scheme
        
        elif spending_func == 'relative':
            self.spending_func = self.relative_spending_scheme
        
        elif spending_func == 'relative200':
             self.spending_func = self.relative200_spending_scheme
                
        elif spending_func in ['costaware','lord1', 'lord2', 'lord3', 'lord++', 'saffron']:
            self.spending_func = self.no_spending_scheme
        
        else:
            #'lord1', 'lord2', 'lord3', 'lord++', 'saffron'
            raise NotImplementedError
        
        if investing_func == 'alpha investing':
            self.investing_func = self.alpha_investing
        elif investing_func == 'alpha rewards':
            self.investing_func = self.alpha_spending_with_rewards
        elif investing_func == 'alpha spending':
            self.investing_func = self.alpha_spending
        elif investing_func== 'ero investing':
            self.investing_func = self.ero_investing
        elif investing_func == 'costaware':
            self.investing_func = self.caero_investing
        elif investing_func == 'fhcostaware':
            self.investing_func = self.finite_horizon
        elif investing_func == 'tscostaware':
            self.investing_func = self.tscaero_investing
        elif investing_func== 'lord1':
            self.investing_func = self.lord1_investing
            self.starting_frac = 0.1
        elif investing_func == 'lord2':
            self.investing_func = self.lord2_investing
            self.starting_frac = 0.1
        elif investing_func == 'lord3':
            self.investing_func = self.lord3_investing
            self.starting_frac = 0.5
        elif investing_func == 'lord++':
            self.investing_func = self.lordww_investing
            self.starting_frac = 0.5
        elif investing_func == 'saffron':
            self.investing_func = self.saffron_investing
            self.starting_frac = 0.5
        else:
            raise NotImplementedError
    
    def initialize_params(self, alpha, eta,
                rate = 0.1, Wdollar = 1000, k = 1.0, omega = 0.05,
                gamma = np.ones(1000), muA = 2, sigma = 1, n_pure = 1,
               model_name = 'ero', working_dir =  "./gams/", n_upper = None, horizon = 1):
        self.n_pure = n_pure
        self.j = 0
        if self.n_pure is not None:
            self.n = self.n_pure
        if self.spending_func == self.constant_spending_scheme or self.spending_func == self.relative_spending_scheme:
            self.Walpha = alpha * eta
            self.Walpha0 = alpha * eta
            self.rate = rate
            self.varphi = 0
            self.Wdollar = Wdollar
            
        elif self.spending_func == self.relative200_spending_scheme:
            self.Walpha = alpha * eta
            self.Walpha0 = alpha * eta
            self.rate = rate
            self.varphi = 0
            self.Wdollar = Wdollar
            self.test_num = 1
        
        elif self.spending_func == self.no_spending_scheme:
            self.Walpha = alpha * eta
            self.Walpha0 = alpha * eta
            self.varphi = 0
            self.Wdollar = Wdollar
        else:
            raise NotImplementedError    
            
            
        if self.investing_func == self.alpha_investing:
            self.omega = omega
            self.alpha = 0
            self.psi = 0
            
        elif self.investing_func == self.alpha_spending:
            self.alpha = 0
            self.psi = 0
            
        elif self.investing_func == self.alpha_spending_with_rewards:
            self.alpha = 0
            self.psi = 0
            self.k = k
            self.muA = muA
            self.sigma = sigma
            self.n = n_pure
            
        elif self.investing_func == self.ero_investing:
            self.alpha = 0
            self.psi = 0
            self.muA = muA
            self.sigma = sigma
            self.n_pure = n_pure
            self.model_name = model_name
            self.working_dir = working_dir
            self.n_pure = n_pure
        
        elif self.investing_func == self.caero_investing:
            self.alpha = 0
            self.psi = 0
            self.muA = muA
            self.sigma = sigma
            self.q = 0.9
            self.model_name = model_name
            self.working_dir = working_dir
            self.n_pure = n_pure
            self.n_upper = n_upper
            self.n = 1
            
        elif self.investing_func == self.tscaero_investing:
            self.alpha = 0
            self.psi = 0
            self.muA1 = muA
            self.sigma1 = sigma
            self.muA2 = muA
            self.sigma2 = sigma
            self.q1 = 0.9
            self.q2 = 0.9
            self.model_name = model_name
            self.working_dir = working_dir
            self.n_pure = n_pure
            self.n_upper = n_upper
            self.n = 1
            
        elif self.investing_func == self.finite_horizon:
            self.alpha = 0
            self.psi = 0
            self.muA = muA
            self.sigma = sigma
            self.q = 0.9
            # self.q2 = 0.9
            self.model_name = model_name
            self.working_dir = working_dir
            self.n_pure = n_pure
            self.n_upper = n_upper
            self.n = 1
            self.horizon = horizon
            
        elif (self.investing_func == self.lord1_investing or 
              self.investing_func == self.lord2_investing or
              self.investing_func == self.lord3_investing or
              self.investing_func == self.lordww_investing):
            m = np.arange(1,1001)
            self.gamma = 0.07720838 * np.log(np.asarray([np.max([2,i]) for i in m]))
            self.gamma /= ( m * np.exp(np.sqrt(np.log(m))))
            self.gamma /= np.sum(self.gamma)
            self.alpha0 = alpha
            self.Walpha0 = self.alpha0 * self.starting_frac
            self.b0 = self.alpha0 - self.Walpha0
            self.rej_list = []
            self.j = 0
            self.rejection = False
            self.first_rej = None
            self.last_rej = None
            self.alpha = 0
            self.psi = 0
            self.n_pure = n_pure
            self.n = 1
        
        elif self.investing_func == self.saffron_investing:
            gamma_vec_exponent = 1.6
            self.gamma = np.true_divide(np.ones(1000), np.power(np.arange(1000)+1, gamma_vec_exponent))
            self.gamma /= np.sum(self.gamma)
            self.Walpha0 = alpha * self.starting_frac
            self.candidates = []
            self.last_rej_ = 0
            self.first = 0
            self.flag = 0
            self.last_rej = []
            self.w0 = self.Walpha0
            self.lbd = 0.5
            self.alpha = 0
            self.psi = 0
            self.n_pure = n_pure
            self.n = 1
            self.j = 0
            self.alpha0 = alpha
            self.lbd_policy = 'fixed'
        
    def update_params(self, Walpha, Wdollar, R, i):
        self.Walpha = Walpha
        self.Wdollar = Wdollar
        
        if self.n_pure is not None:
            if self.Wdollar < self.n_pure:
                self.n_pure = self.Wdollar
        if self.spending_func == self.relative200_spending_scheme:
            self.test_num = i + 1
        if (self.investing_func == self.lord1_investing or 
              self.investing_func == self.lord2_investing or
              self.investing_func == self.lord3_investing or
              self.investing_func == self.lordww_investing):
            if R[-1] == 1:
                self.rejection = True
                self.Walpha_lr = Walpha
                if self.first_rej is None:
                    self.first_rej = i
                    self.last_rej = i
                else:
                    self.last_rej = i
                self.rej_list.append(i)
        self.j += 1
    
    def step(self, q = 0.9, q_list = None):
        self.spending_func()
        self.q = q
        if q_list is not None:
            self.q_list = q_list
            self.q1 = q_list[0]
            self.q2 = q_list[1]
        self.investing_func()
        
        return {'varphi' : self.varphi,
                'alpha' : self.alpha,
                'psi' : self.psi}
    
    #Constant SPENDING Scheme
    def constant_spending_scheme(self):

        self.varphi = np.min([self.rate * self.Walpha0, self.Walpha])

    #Relative
    def relative_spending_scheme(self):
        self.varphi = self.rate * self.Walpha
        if self.Walpha < 1/1000 * self.Walpha0:
            raise Exception("Relative stops when Walpha less than 1/1000 * initial wealth")
            
    #Relative200
    def relative200_spending_scheme(self):
        self.varphi = self.rate * self.Walpha
        if self.test_num >= 200:
            raise Exception("Relative 200 stops after 200 tests")
            
    #Used when spending and investing done simultaneously
    def no_spending_scheme(self):
        self.varphi = self.varphi
    
    #Alpha spending (investing rule)
    def alpha_spending(self):
        self.alpha = self.varphi, 
        self.psi = 0
    
    #Foster and Stine alpha investing
    def alpha_investing(self):
        self.alpha = self.varphi / (1 + self.varphi)
        self.psi = self.varphi + self.omega
    
    #alpha spending with rewards
    def alpha_spending_with_rewards(self):
        self.alpha = self.varphi / self.k
        self.rho = 1 - ss.norm.cdf( ((0 - self.muA)*np.sqrt(self.n)/self.sigma) + ss.norm.ppf(1-self.alpha, 0,1), 0, 1)
        self.psi = np.min([self.alpha / self.rho + self.Walpha0/self.k , 1 - ((1 - self.Walpha0)/self.k)])
    
    #ERO
    def ero_investing(self, absolute_to_gms = "/home/tjcook/alpha-investing-r1/src/procedures"):
        # original args: varphi, model_name, muA, sigma, n_pure, working_dir = "./gams/"  
        ws = GamsWorkspace(system_directory = "/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx",
                       debug=DebugLevel.Off,
                       working_directory = self.working_dir)
    
        cp = ws.add_checkpoint()
        job = ws.add_job_from_file(os.path.join(absolute_to_gms, self.model_name + '.gms'))
        job.run(checkpoint=cp)
    
        mi = cp.add_modelinstance()
        #print(self.varphi, type(self.varphi))
        phi_db = mi.sync_db.add_parameter("phi", 0, "cost of the test")
        mua_db = mi.sync_db.add_parameter("muA", 0, "alternative mean")
        sigma_db = mi.sync_db.add_parameter("sigma", 0, "std dev of data")
        if self.model_name != 'ero-n1':
            n_db = mi.sync_db.add_parameter("n", 0, "sample size")
        # nup_db = mi.sync_db.add_parameter("nup", 0, "upper bound on n")
        opt = ws.add_options()
        opt.all_model_types = 'conopt'
                                     
        if self.model_name != 'ero-n1':
            mi.instantiate("ero using nlp maximizing er1", 
                       [GamsModifier(phi_db),GamsModifier(mua_db),
                        GamsModifier(sigma_db),
                        GamsModifier(n_db),], opt)
        else: 
            mi.instantiate("ero using nlp maximizing er1", 
                       [GamsModifier(phi_db),GamsModifier(mua_db),
                        GamsModifier(sigma_db),], opt)
        phi_db.add_record().value = self.varphi
        mua_db.add_record().value = self.muA
        sigma_db.add_record().value = self.sigma
        # nup_db.add_record().value = self.n_pure
        if self.model_name != 'ero-n1':
            n_db.add_record().value = self.n_pure
        
        mi.solve()
        assert mi.model_status < 3, "Did not solve"
        
        self.alpha = mi.sync_db.get_variable("alpha").find_record().level
        self.psi = mi.sync_db.get_variable("psi").find_record().level
        self.rho = mi.sync_db.get_variable("rho").find_record().level
    
    #CAERO
    def caero_investing(self, absolute_to_gms = "/home/tjcook/alpha-investing-r1/src/procedures"):
        # original args: varphi, model_name, muA, sigma, n_pure, working_dir = "./gams/"  
        model_status = 4
        reset_counter = 0
        while model_status >= 3 and reset_counter < 3:
            ws = GamsWorkspace(system_directory = "/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx",
                       debug=DebugLevel.Off,
                       working_directory = self.working_dir)
            cp = ws.add_checkpoint()
            self.update_caero_text()
            job = ws.add_job_from_file(os.path.join(absolute_to_gms, 'caero' + self.temp_gms + '.gms'))
            job.run(checkpoint=cp)
            mi = cp.add_modelinstance()
            q0_db = mi.sync_db.add_parameter("q0", 0, "prior of the null")
            muA_db = mi.sync_db.add_parameter("muA", 0, "alternative mean")
            sigma_db = mi.sync_db.add_parameter("sigma", 0, "std dev of data")
            Wa_db = mi.sync_db.add_parameter("Wa", 0, "alpha wealth")
            Wd_db = mi.sync_db.add_parameter("Wd", 0, "dollar wealth")
            
            if self.n_pure is not None and self.n_upper is None:
                n_db = mi.sync_db.add_variable("n", 0, VarType.Positive)
                nup_db = mi.sync_db.add_parameter("nup", 0, "upper bound on n")
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                                  [GamsModifier(q0_db), GamsModifier(Wa_db), 
                                   GamsModifier(Wd_db), GamsModifier(muA_db), 
                                   GamsModifier(sigma_db), GamsModifier(n_db, UpdateAction.Fixed, nup_db),], opt)
                nup_db.add_record().value = self.n_pure
                n_db.add_record().value = self.n_pure
            elif self.n_upper is not None:
                n_db = mi.sync_db.add_variable("n", 0, VarType.Positive)
                nup_db = mi.sync_db.add_parameter("nup", 0, "upper bound on n")
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                              [GamsModifier(q0_db), GamsModifier(Wa_db), 
                               GamsModifier(Wd_db), GamsModifier(muA_db), 
                               GamsModifier(sigma_db),
                               GamsModifier(n_db, UpdateAction.Upper, nup_db), ], opt)
                nup_db.add_record().value = self.n_upper
                # n_db.add_record().value = 1
            else:
                n_db = mi.sync_db.add_variable("n", 0, VarType.Positive)
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                              [GamsModifier(q0_db), GamsModifier(Wa_db), 
                               GamsModifier(Wd_db), GamsModifier(muA_db), 
                               GamsModifier(sigma_db), GamsModifier(n_db)], opt)
                # n_db.add_record().value = 1

            
            q0_db.add_record().value = self.q
            Wa_db.add_record().value = self.Walpha
            Wd_db.add_record().value = self.Wdollar
            muA_db.add_record().value = self.muA
            sigma_db.add_record().value = self.sigma

            mi.solve()
            os.remove(os.path.join(absolute_to_gms, 'caero' + self.temp_gms + '.gms'))
            reset_counter += 1
            model_status = mi.model_status
        if model_status >= 3 and reset_counter == 3:
            raise RuntimeError('Failed to find locally feasible solution after 100 reinitializations')
        
        self.varphi = mi.sync_db.get_variable("phi").find_record().level
        self.alpha = mi.sync_db.get_variable("alpha").find_record().level
        self.psi = mi.sync_db.get_variable("psi").find_record().level
        self.rho = mi.sync_db.get_variable("rho").find_record().level
        self.n = mi.sync_db.get_variable("n").find_record().level
    
    def update_caero_text(self, absolute_to_gms = "/home/tjcook/alpha-investing-r1/src/procedures"):

        with open(os.path.join(absolute_to_gms, self.model_name + '.gms')) as f:
            lines = f.readlines()
        
        temp_list = lines[7].split(' /')
        new_mu = " " + str(self.muA)
        temp_list[1] = new_mu

        lines[7] = ' /'.join(temp_list)

        temp_list = lines[9].split(' /')
        new_sigma = " " + str(self.sigma)
        temp_list[1] = new_sigma

        lines[9] = ' /'.join(temp_list)
        
        temp_list = lines[10].split(' /')
        new_prior = " " + str(self.q)
        temp_list[1] = new_prior

        lines[10] = ' /'.join(temp_list)

        temp_list = lines[12].split(' /')
        new_dollar = " " + str(self.Wdollar)
        temp_list[1] = new_dollar

        lines[12] = ' /'.join(temp_list)

        temp_list = lines[13].split(' /')
        new_alpha = " " + str(self.Walpha)
        temp_list[1] = new_alpha

        lines[13] = ' /'.join(temp_list)
        with open(os.path.join(absolute_to_gms, 'caero' + self.temp_gms + '.gms'), 'w') as f:
            for line in lines:
                f.write(line)
    
    
    def update_tscaero_text(self, absolute_to_gms = "/home/tjcook/alpha-investing-r1/src/procedures"):
        with open(os.path.join(absolute_to_gms, self.model_name + '.gms')) as f:
                lines = f.readlines()

        temp_list = lines[7].split(' /')
        new_mu1 = " " + str(self.muA1)
        temp_list[1] = new_mu1
        lines[7] = ' /'.join(temp_list)

        temp_list = lines[8].split(' /')
        new_mu2 = " " + str(self.muA2)
        temp_list[1] = new_mu2
        lines[8] = ' /'.join(temp_list)

        temp_list = lines[10].split(' /')
        new_sigma1 = " " + str(self.sigma1)
        temp_list[1] = new_sigma1
        lines[10] = ' /'.join(temp_list)

        temp_list = lines[11].split(' /')
        new_sigma2 = " " + str(self.sigma2)
        temp_list[1] = new_sigma2
        lines[11] = ' /'.join(temp_list)

        temp_list = lines[12].split(' /')
        new_prior1 = " " + str(self.q1)
        temp_list[1] = new_prior1
        lines[12] = ' /'.join(temp_list)

        temp_list = lines[13].split(' /')
        new_prior2 = " " + str(self.q2)
        temp_list[1] = new_prior2
        lines[13] = ' /'.join(temp_list)

        temp_list = lines[15].split(' /')
        new_dollar = " " + str(self.Wdollar)
        temp_list[1] = new_dollar
        lines[15] = ' /'.join(temp_list)

        temp_list = lines[16].split(' /')
        new_alpha = " " + str(self.Walpha)
        temp_list[1] = new_alpha
        lines[16] = ' /'.join(temp_list)

        with open(os.path.join(absolute_to_gms, 'tscaero' + self.temp_gms + '.gms'), 'w') as f:
            for line in lines:
                f.write(line)
    
    #TSCAERO
    def tscaero_investing(self, absolute_to_gms = "/home/tjcook/alpha-investing-r1/src/procedures"):
        # original args: varphi, model_name, muA, sigma, n_pure, working_dir = "./gams/"  
        model_status = 4
        reset_counter = 0
        while model_status >= 3 and reset_counter < 100:
            ws = GamsWorkspace(system_directory = "/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx",
                       debug=DebugLevel.Off,
                       working_directory = self.working_dir)
            
            cp = ws.add_checkpoint()
            self.update_tscaero_text()
            job = ws.add_job_from_file(os.path.join(absolute_to_gms, 'tscaero' + self.temp_gms + '.gms'))
            job.run(checkpoint=cp)
            mi = cp.add_modelinstance()
            q1_db = mi.sync_db.add_parameter("q1", 0, "prior of the first null")
            q2_db = mi.sync_db.add_parameter("q2", 0, "prior of the second null")
            muA1_db = mi.sync_db.add_parameter("muA1", 0, "alternative mean1")
            muA2_db = mi.sync_db.add_parameter("muA2", 0, "alternative mean2")
            sigma1_db = mi.sync_db.add_parameter("sigma1", 0, "sigma1") 
            sigma2_db = mi.sync_db.add_parameter("sigma2", 0, "sigma2") 
            Wa_db = mi.sync_db.add_parameter("Wa", 0, "alpha wealth")
            Wd_db = mi.sync_db.add_parameter("Wd", 0, "dollar wealth")
            n1_db = mi.sync_db.add_variable("n1", 0, VarType.Positive)
            n2_db = mi.sync_db.add_variable("n2", 0, VarType.Positive)
            n1up_db = mi.sync_db.add_parameter("n1up", 0, "upper bound on n1")
            n2up_db = mi.sync_db.add_parameter("n2up", 0, "upper bound on n2")
            
            if self.n_pure is not None and self.n_upper is None:
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                                  [GamsModifier(q1_db),GamsModifier(q2_db), GamsModifier(Wa_db), 
                                   GamsModifier(Wd_db), GamsModifier(muA1_db),GamsModifier(muA2_db), 
                                   GamsModifier(sigma1_db),GamsModifier(sigma2_db), 
                                   GamsModifier(n1_db, UpdateAction.Fixed, n1up_db),
                                   GamsModifier(n2_db, UpdateAction.Fixed, n2up_db),], opt)
                
                n1up_db.add_record().value = self.n_pure
                n1_db.add_record().value = self.n_pure
                n2up_db.add_record().value = self.n_pure
                n2_db.add_record().value = self.n_pure
            elif self.n_upper is not None:
                # nup_db = mi.sync_db.add_parameter("nup", 0, "upper bound on n")
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                                  [GamsModifier(q1_db),GamsModifier(q2_db), GamsModifier(Wa_db), 
                                   GamsModifier(Wd_db), GamsModifier(muA1_db),GamsModifier(muA2_db), 
                                   GamsModifier(sigma1_db),GamsModifier(sigma2_db), 
                                   GamsModifier(n1_db, UpdateAction.Upper, n1up_db),
                                   GamsModifier(n2_db, UpdateAction.Upper, n2up_db),], opt)
               
                n1up_db.add_record().value = self.n_upper
                n1_db.add_record().value = 1
                n2up_db.add_record().value = self.n_upper
                n2_db.add_record().value = 1
            else:
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                                  [GamsModifier(q1_db),GamsModifier(q2_db), GamsModifier(Wa_db), 
                                   GamsModifier(Wd_db), GamsModifier(muA1_db),GamsModifier(muA2_db), 
                                   GamsModifier(sigma1_db),GamsModifier(sigma2_db), 
                                   GamsModifier(n1_db),
                                   GamsModifier(n2_db),], opt)
               
                n1_db.add_record().value = 1
                n2_db.add_record().value = 1

            
            q1_db.add_record().value = self.q1
            q2_db.add_record().value = self.q2
            Wa_db.add_record().value = self.Walpha
            Wd_db.add_record().value = self.Wdollar
            muA1_db.add_record().value = self.muA1
            muA2_db.add_record().value = self.muA2
            sigma1_db.add_record().value = self.sigma1
            sigma2_db.add_record().value = self.sigma2

            mi.solve()
            os.remove(os.path.join(absolute_to_gms, 'tscaero' + self.temp_gms + '.gms'))
        
            reset_counter += 1
            model_status = mi.model_status
        if model_status >= 3 and reset_counter == 100:
            raise RuntimeError('Failed to find locally feasible solution after 100 reinitializations')
        
        self.varphi = mi.sync_db.get_variable("phi1").find_record().level
        self.alpha = mi.sync_db.get_variable("alpha1").find_record().level
        self.psi = mi.sync_db.get_variable("psi1").find_record().level
        self.rho = mi.sync_db.get_variable("rho1").find_record().level
        self.n = mi.sync_db.get_variable("n1").find_record().level
        
    
    def generate_finitehorizon_text(self, absolute_to_gms = "/home/tjcook/alpha-investing-r1/src/procedures"):
        with open(os.path.join(absolute_to_gms, 'fh.gms')) as f:
            lines = f.readlines()
        #update size of j
        temp_list = lines[7].split(' /')
        new_set = 'j1*j'+str(self.horizon)+'/\n'
        temp_list[-1] = new_set
        lines[7] = ' /'.join(temp_list)

        temp_list = lines[11].split(' /')
        new_dollar = " " + str(self.Wdollar)
        temp_list[1] = new_dollar
        lines[11] = ' /'.join(temp_list)

        temp_list = lines[12].split(' /')
        new_alpha = " " + str(self.Walpha)
        temp_list[1] = new_alpha
        lines[12] = ' /'.join(temp_list)

        temp_list = lines[18].split(' /')
        update_qlist = ' '
        for i in range(self.horizon):
            if i == self.horizon - 1:
                update_qlist += 'j' + str(i+1) + '=' + str(self.q_list[i])
            else:
                update_qlist += 'j' + str(i+1) + '=' + str(self.q_list[i]) + ", "
        temp_list[-2] = update_qlist
        lines[18] = ' /'.join(temp_list)

        temp_list = lines[79].split("'")
        new_last = "j" + str(self.horizon)
        temp_list[1] = new_last
        lines[79] = "'".join(temp_list)

        with open(os.path.join(absolute_to_gms, 'fh' + self.temp_gms + '.gms'), 'w') as f:
                for line in lines:
                    f.write(line)

    
    def finite_horizon(self,absolute_to_gms = "/home/tjcook/alpha-investing-r1/src/procedures"):
        
        model_status = 4
        reset_counter = 0
        
        self.generate_finitehorizon_text()
        
        while model_status >= 3 and reset_counter < 100:
            ws = GamsWorkspace(system_directory = "/opt/ohpc/pub/compiler/gams/gams38.2_linux_x64_64_sfx",
                       debug=DebugLevel.Off,
                       working_directory = self.working_dir)
            
            cp = ws.add_checkpoint()
            # self.update_finitehorizon_text()
            job = ws.add_job_from_file(os.path.join(absolute_to_gms, 'fh' + self.temp_gms + '.gms'))
            job.run(checkpoint=cp)
            mi = cp.add_modelinstance()
            j_db = mi.sync_db.add_set("j", 1)
            muA_db = mi.sync_db.add_parameter("muA", 1, "alternative mean")
            sigma_db = mi.sync_db.add_parameter("sigma", 1, "std dev of data") 
            Wa_db = mi.sync_db.add_parameter("Wa", 0, "alpha wealth")
            Wd_db = mi.sync_db.add_parameter("Wd", 0, "dollar wealth")
            n_db = mi.sync_db.add_variable("n", 1, VarType.Positive)
            nup_db = mi.sync_db.add_parameter("nup", 1, "upper bound on n")
            q0_db = mi.sync_db.add_parameter("q0", 1, "prior of the null")
            
            # for i in range(self.horizon): 
            #     j_db.sync_db.add_record("j{}".format(i+1))
            #     key = ("j{}".format(i+1))
            #     muA_db.sync_db.add_record(key).value = 2.0
            #     sigma_db.sync_db.add_record(key).value = 1.0
            #     q0_db.sync_db.add_record(key).value = q[i]
            
            if self.n_pure is not None and self.n_upper is None:
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                                  [GamsModifier(q0_db), GamsModifier(Wa_db), 
                                   GamsModifier(Wd_db), GamsModifier(muA_db), 
                                   GamsModifier(sigma_db),
                                   GamsModifier(n_db, UpdateAction.Fixed, nup_db),], opt)
                
                
                
                for i in range(self.horizon): 
                    j_db.add_record("j{}".format(i+1))
                    key = ("j{}".format(i+1))
                    muA_db.add_record(key).value = float(self.muA)
                    sigma_db.add_record(key).value = self.sigma
                    q0_db.add_record(key).value = self.q_list[i]
                    n_db.add_record(key).value = self.n_pure
                    nup_db.add_record(key).value = float(self.n_pure)
                
            elif self.n_upper is not None:                
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                                  [GamsModifier(q0_db), GamsModifier(Wa_db), 
                                   GamsModifier(Wd_db), GamsModifier(muA_db), 
                                   GamsModifier(sigma_db),
                                   GamsModifier(n_db, UpdateAction.Upper, nup_db),], opt)
                
                
                
                for i in range(self.horizon): 
                    j_db.add_record("j{}".format(i+1))
                    key = ("j{}".format(i+1))
                    muA_db.add_record(key).value = float(self.muA)
                    sigma_db.add_record(key).value = self.sigma
                    q0_db.add_record(key).value = self.q_list[i]
                    n_db.add_record(key).value = float(self.n_upper)
                    nup_db.add_record(key).value = float(self.n_upper)
                
                
            else:
                opt = ws.add_options()
                opt.all_model_types = 'conopt'
                mi.instantiate("ero using nlp maximizing er1", 
                                  [GamsModifier(q0_db), GamsModifier(Wa_db), 
                                   GamsModifier(Wd_db), GamsModifier(muA_db), 
                                   GamsModifier(sigma_db),
                                   GamsModifier(n_db),], opt)
                
                
                
                for i in range(self.horizon): 
                    j_db.add_record("j{}".format(i+1))
                    key = ("j{}".format(i+1))
                    muA_db.add_record(key).value = float(self.muA)
                    sigma_db.add_record(key).value = self.sigma
                    q0_db.add_record(key).value = self.q_list[i]
                    #n_db.add_record(key).value = 1
                    

            
            
            Wa_db.add_record().value = self.Walpha
            Wd_db.add_record().value = self.Wdollar
            
            mi.solve()
            os.remove(os.path.join(absolute_to_gms, 'fh' + self.temp_gms + '.gms'))
        
            reset_counter += 1
            model_status = mi.model_status
        if model_status >= 3 and reset_counter == 100:
            raise RuntimeError('Failed to find locally feasible solution after 100 reinitializations')
        self.varphi = mi.sync_db.get_variable("phi").find_record('j1').level
        self.alpha = mi.sync_db.get_variable("alpha").find_record('j1').level
        self.psi = mi.sync_db.get_variable("psi").find_record('j1').level
        self.rho = mi.sync_db.get_variable("rho").find_record('j1').level
        self.n = mi.sync_db.get_variable("n").find_record('j1').level
        
    
    
    def lord1_investing(self):
        if self.rejection:
            self.varphi = self.gamma[self.j - self.last_rej] * self.b0
            self.alpha = self.varphi
            self.psi = self.b0
        else:
            self.varphi = self.gamma[self.j] * self.Walpha0
            self.alpha = self.varphi
            self.psi = self.b0
    
    def lord2_investing(self):
        ## to init:
        #gamma, b0, rejection index list, current index, wealth0, rejection indicator
        if self.rejection:
            self.varphi = self.gamma[self.j]*self.Walpha0 + np.sum(self.gamma[[self.j - idx for idx in self.rej_list]])*self.b0
            self.alpha = self.varphi
            self.psi = self.b0
        else:
            self.varphi = self.gamma[self.j] * self.Walpha0
            self.alpha = self.varphi
            self.psi = self.b0
    
    def lord3_investing(self):
        ## to init:
        #gamma, b0, rejection index list, current index, wealth0, rejection indicator
        if self.rejection:
            self.varphi = self.gamma[self.j - self.last_rej] * self.Walpha_lr
            self.alpha = self.varphi
            self.psi = self.b0
        else:
            self.varphi = self.gamma[self.j] * self.Walpha0
            self.alpha = self.varphi
            self.psi = self.b0
    
    def lordww_investing(self):
        ## to init:
        #gamma, b0, rejection index list, current index, wealth0, rejection indicator, first rejection
        if self.rejection:
            if self.first_rej == self.last_rej:
                self.varphi = self.gamma[self.j] * self.Walpha0 + (self.alpha0 - self.Walpha0)*self.gamma[self.j - self.first_rej]
                self.alpha = self.varphi
                self.psi = self.b0
            else:
                self.varphi = self.gamma[self.j]*self.Walpha0 + (self.alpha0 - self.Walpha0)*self.gamma[self.j - self.first_rej] + self.alpha0*np.sum(self.gamma[self.rej_list[1:]])
                self.alpha = self.varphi
                self.psi = self.b0
        else:
            self.varphi = self.gamma[self.j] * self.Walpha0
            self.alpha = self.varphi
            self.psi = self.b0     
            
    def saffron_investing(self):
        #Init gamma, last reject idx, first rejection idx, flag, last rejection list, w0, lbd
        if self.j == 0:
            self.alpha = np.min([self.lbd, (1-self.lbd)*self.gamma[0] * self.w0])
        else:    
            candidates_total = int(np.sum(self.candidates))
            zero_gam = self.gamma[self.j - candidates_total]
                
            if len(self.last_rej) >= 0:
                if len(self.last_rej) >= 1:
                    candidates_after_first = int(np.sum(self.candidates[self.last_rej[0] + 1:]))
                    first_gam = self.gamma[self.j - self.last_rej[0] - candidates_after_first]
                else:
                    first_gam = 0
                if len(self.last_rej) >= 2:
                    sum_gam = self.gamma[self.j * np.ones(len(self.last_rej) - 1, dtype=int) - self.last_rej[1:] - self.count_candidates(self.last_rej, self.candidates)]
                    sum_gam = sum(sum_gam)
                else:
                    sum_gam = 0
                next_alpha = min(self.lbd, (1 - self.lbd) * zero_gam * self.w0 + (1 - self.lbd) * (self.alpha0 - self.w0) * first_gam + (1 - self.lbd) * self.alpha0 * sum_gam)
                self.alpha = next_alpha
                if self.lbd_policy == 'update':
                    self.lbd = next_alpha
            else:
                next_alpha = min(self.lbd, (1 - self.lbd) * zero_gam * self.w0)
                self.alpha = next_alpha
                if self.lbd_policy == 'update':
                    self.lbd = next_alpha
            self.flag = 0
            
    def count_candidates(self, last_rej, candidates):
        ret_val = []
        for j in range(1,len(last_rej)):
            ret_val = np.append(ret_val, sum(candidates[last_rej[j]:]))
        return ret_val.astype(int)
        
        
        
def compute_metrics(R, alt_true, eta): ### THIS ASSUMES A SINGLE SAMPLE - NEEDS TO BE EDITED
    ntest = len(R)
    V = R * (1-alt_true)
    true_reject = sum(R * (alt_true))
    false_reject = sum(R * (1-alt_true))
    return {'ntest': ntest, 'tr': true_reject, 'fr': false_reject,'nskip' : 0, 
            'nsample' : 1,
            'nsample_tot' : ntest,
            'true_alt_tested': np.sum(alt_true), 
            'true_alt_total': np.sum(alt_true)}

def temp_main():
    np.random.seed(i)
    
    results = pd.DataFrame(columns = ['dataset', 'scheme', 'method', 'ntest', 'tr', 'fr', 'mfdr', 'nskip', 'nsample', 'nsample_tot','true_alt_tested','true_alt_total', 'nsample_rej'])
    
    method_dict = {'constant__alpha_investing' : {'spending' : 'constant',
                                                  'investing': 'alpha investing',
                                                  'max_tests': 1000}}
    
    alpha0 = 0.05
    eta = 1 - alpha0
    n_pure = 1

    #Sample the data
    max_size = method_dict[method]['max_tests']        
    
    y_alt_true = np.random.choice([0,1] , p = [0.9,0.1])
    if y_alt_true ==1:
        y = np.random.normal(loc=2)
    else:
        y = np.random.normal()
        
    if type(y) == float:
        y = np.array(y).reshape(1,)
    alt_true.append(y_alt_true)
    
    for method in method_dict:
        # Initialize wealths and rejections
        Walpha = [alpha0*eta, ]
        Wdollar = [1000,]
        R = []
        varphi = []
        alpha = []
        psi = []
        alt_true = []
        # Initialize spending and investings schemes
        rule = investing_rule('constant', 'alpha investing', alpha0, eta)
        
        for i in range(method_dict[method]['max_tests']):
            # #Sample the data
            
            # y_alt_true = np.random.choice([0,1], p = [0.9,0.1])
            # if y_alt_true ==1:
            #     y = np.random.normal(loc=2)
            # else:
            #     y = np.random.normal()
                
            # if type(y) == float:
            #     y = np.array(y).reshape(1,)
            # alt_true.append(y_alt_true)
            
            #Spend and invest wealth
            rule.step()
            
            varphi.append(rule.varphi)
            alpha.append(rule.alpha)
            psi.append(rule.psi)
            
            if method == 'cost_aware_ero':
                n = rule.n
            else:
                n = n_pure
            
            #Perform test
            res = norm_test(y, rule.alpha, n)
            
            #Update wealth, rejections
            R.append(res['R'])
            Walpha.append(Walpha[-1] - varphi[-1] + R[-1]*psi[-1])
            Wdollar.append(Wdollar[-1] - n)
            
            #Update params
            rule.update_params(Walpha[-1], Wdollar[-1],
                                       R[-1], i) 
            
            #Check to see if we have exceeded wealths or num iters.
            if Walpha[-1] <= 0 or Wdollar[-1] <= 0:
                break
        
        perf_metrics = compute_metrics(np.array(R), np.array(alt_true), eta)
        
        results = pd.concat([results, 
                                     pd.DataFrame.from_dict({'dataset': [1], 
                                                             'scheme': [method_dict[method]['spending']], 
                                                             'method': [method_dict[method]['investing']], 
                                                             **perf_metrics})], 
                                    ignore_index=True)
                          
    print(result)
    return results
            
            
            
if __name__== '__main__':
    result =temp_main()




