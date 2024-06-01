import numpy as np
import scipy.optimize as optimize

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_2d, interp_3d, interp_2d_vec, binary_search, interp_3d_vec
from consav.linear_interp_1d import _interp_1d
from consav import quadrature
from scipy.optimize import minimize,  NonlinearConstraint

# user-specified functions
import UserFunctions as usr

# set gender indication as globals
woman = 1
man = 2

class HouseholdModelClass(EconModelClass):
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. cpp
        self.cpp_filename = 'cppfuncs/solve.cpp'
        self.cpp_options = {'compiler':'vs'}
        
    def setup(self):
        par = self.par
        
        par.R = 1.03
        par.beta = 1.0/par.R # Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife
        
       

        # Utility: gender-specific parameters
        par.rho_w = 2.0        # CRRA
        par.rho_m = 2.0        # CRRA
        
        par.alpha1_w = 1.0
        par.alpha1_m = 1.0
        
        par.alpha2_w = 1.0
        par.alpha2_m = 1.0
        
        par.phi_w = 0.2
        par.phi_m = 0.2
        
        par.theta0_w = 0.1 # constant disutility of work for women
        par.theta0_m = 0.1 # constant disutility of work for men
        par.theta1_w = 0.05 # additional disutility of work from children, women 
        par.theta1_m = 0.05 # additional disutility of work from children, men
        par.gamma = 0.5 # disutility of work elasticity

        # state variables
        par.T = 10
        
        # wealth
        par.num_A = 20
        par.max_A = 10.0
        
        # human capital 
        par.num_H = 20
        par.max_H = 5.0

        # income
        par.wage_const_1 = np.log(10.0) # constant, men
        par.wage_const_2 = np.log(10.0) # constant, women
        par.wage_K_1 = 0.1 # return on human capital, men
        par.wage_K_2 = 0.1 # return on human capital, women
        par.delta = 0.1 # depreciation in human capital

        # child-related transfers
        par.uncon_uni = 0.1
        par.means_level = 0.1
        par.means_slope = 0.001
        par.cond = -0.1
        par.cond_high = -0.1

        # bargaining power
        par.num_power = 20

        # love/match quality
        par.num_love = 20
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5

        # pre-computation
        par.num_Ctot = 100
        par.max_Ctot = par.max_A*2
        par.num_Htot = 100
        par.max_Htot = par.max_A*2

        par.num_A_pd = par.num_A * 2

        # simulation
        par.seed = 9210
        par.simT = par.T
        par.simN = 50_000

        # grids        
        par.k_max = 20.0 # maximum point in HC grid
        par.num_k = 15 #30 # number of grid points in HC grid
        par.num_n = 2 # maximum number in my grid over children

        #interest rate
        par.r = 0.03

        # birth probability
        par.p_birth = 0.05
        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        par.simT = par.T
        self.setup_grids()
        
        # singles
        shape_single = (par.T, par.num_n, par.num_A, par.num_k)
        sol.Vw_single = np.nan + np.ones(shape_single)
        sol.Vm_single = np.nan + np.ones(shape_single)
        sol.Cw_priv_single = np.nan + np.ones(shape_single)
        sol.Cm_priv_single = np.nan + np.ones(shape_single)
        sol.Cw_pub_single = np.nan + np.ones(shape_single)
        sol.Cm_pub_single = np.nan + np.ones(shape_single)
        sol.Cw_tot_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_single = np.nan + np.ones(shape_single)
        sol.Hw_single = np.nan + np.ones(shape_single)
        sol.Hm_single = np.nan + np.ones(shape_single)

        sol.Vw_trans_single = np.nan + np.ones(shape_single)
        sol.Vm_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_priv_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_priv_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_pub_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_pub_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_tot_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_trans_single = np.nan + np.ones(shape_single)
        sol.Hw_trans_single = np.nan + np.ones(shape_single)
        sol.Hm_trans_single = np.nan + np.ones(shape_single)

        # couples
        shape_couple = (par.T,par.num_power,par.num_love,par.num_A, par.num_k, par.num_k, par.num_n)
        sol.Vw_couple = np.nan + np.ones(shape_couple)
        sol.Vm_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_couple = np.nan + np.ones(shape_couple)
        sol.Hw_couple = np.nan + np.ones(shape_couple)
        sol.Hm_couple = np.nan + np.ones(shape_couple)

        sol.Vw_remain_couple = np.nan + np.ones(shape_couple)
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_remain_couple = np.nan + np.ones(shape_couple)
        sol.Hw_remain_couple = np.nan + np.ones(shape_couple)
        sol.Hm_remain_couple = np.nan + np.ones(shape_couple)

        sol.power_idx = np.zeros(shape_couple,dtype=np.int_)
        sol.power = np.zeros(shape_couple)

        # temporary containers
        sol.savings_vec = np.zeros(par.num_shock_love)
        sol.Vw_plus_vec = np.zeros(par.num_shock_love) 
        sol.Vm_plus_vec = np.zeros(par.num_shock_love) 


        # pre-compute optimal consumption allocation 
        shape_pre = (par.num_power,par.num_Ctot)
        sol.pre_Ctot_Cw_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_Cm_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_C_pub = np.nan + np.ones(shape_pre)

        #adding pre-computation for hours
        shape_pre = (par.num_power,par.num_Htot)
        sol.pre_Htot_Hw_kid = np.nan + np.ones(shape_pre)
        sol.pre_Htot_Hm_kid = np.nan + np.ones(shape_pre)
        sol.pre_Htot_Hw_nokid = np.nan + np.ones(shape_pre)
        sol.pre_Htot_Hm_nokid = np.nan + np.ones(shape_pre)
        
        # simulation
        shape_sim = (par.simN,par.simT)
        sim.Cw_priv = np.nan + np.ones(shape_sim)
        sim.Cm_priv = np.nan + np.ones(shape_sim)
        sim.Cw_pub = np.nan + np.ones(shape_sim)
        sim.Cm_pub = np.nan + np.ones(shape_sim)
        sim.Cw_tot = np.nan + np.ones(shape_sim)
        sim.Cm_tot = np.nan + np.ones(shape_sim)
        sim.C_tot = np.nan + np.ones(shape_sim)
        
        sim.A = np.nan + np.ones(shape_sim)
        sim.Aw = np.nan + np.ones(shape_sim)
        sim.Am = np.nan + np.ones(shape_sim)
        sim.couple = np.nan + np.ones(shape_sim)
        sim.power_idx = np.ones(shape_sim,dtype=np.int_)
        sim.power = np.nan + np.ones(shape_sim)
        sim.love = np.nan + np.ones(shape_sim)

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = np.random.normal(size=shape_sim)

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)
        sim.init_Aw = par.div_A_share * sim.init_A #np.zeros(par.simN)
        sim.init_Am = (1.0 - par.div_A_share) * sim.init_A #np.zeros(par.simN)
        sim.init_couple = np.ones(par.simN,dtype=np.bool_)
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        
    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(0.0,par.max_A,par.num_A,1.1)

        par.grid_Aw = par.div_A_share * par.grid_A
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # human capital grid
        par.kw_grid = nonlinspace(0.0,par.k_max,par.num_k,1.1)
        par.km_grid = nonlinspace(0.0,par.k_max,par.num_k,1.1)

        # number of children grid
        par.nw_grid = np.arange(0,par.num_n)
        par.nm_grid = np.arange(0,par.num_n)


        # power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(0.0,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(1.0 - nonlinspace(0.0,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
        par.grid_power = np.append(first_part,last_part)

        # love grid and shock
        if par.num_love>1:
            par.grid_love = np.linspace(-par.max_love,par.max_love,par.num_love)
        else:
            par.grid_love = np.array([0.0])

        if par.sigma_love<=1.0e-6:
            par.num_shock_love = 1
            par.grid_shock_love,par.grid_weight_love = np.array([0.0]),np.array([1.0])

        else:
            par.grid_shock_love,par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love,par.num_shock_love)

        # pre-computation
        #par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)
        #par.grid_Htot = nonlinspace(1.0e-6,par.max_Htot,par.num_Htot,1.1)
        # Adjust non-linear spacing to avoid very small values
        par.epsilon = 1e-5
        par.grid_Ctot = nonlinspace(par.epsilon, par.max_Ctot, par.num_Ctot, 1.1)
        par.grid_Htot = nonlinspace(par.epsilon, par.max_Htot, par.num_Htot, 1.1)

       

    def solve(self):
        sol = self.sol
        par = self.par 

        # setup grids
        self.setup_grids()

        # precompute the optimal intra-temporal consumption allocation for couples given total consumption
        for iP,power in enumerate(par.grid_power):
            for i,C_tot in enumerate(par.grid_Ctot):
                kids = 0 # kids don't matter for consumption allocation
                sol.pre_Ctot_Cw_priv[iP,i], sol.pre_Ctot_Cm_priv[iP,i], sol.pre_Ctot_C_pub[iP,i] = solve_intraperiod_couple(C_tot,power,kids,par) # kids don't matter for consumption allocation
                #print(f"sol.pre_Ctot_Cw_priv: {sol.pre_Ctot_Cw_priv[iP,i]}, sol.pre_Ctot_Cm_priv: {sol.pre_Ctot_Cm_priv[iP,i]}, sol.pre_Ctot_C_pub: {sol.pre_Ctot_C_pub[iP,i]}")
        #precompute the optimal intra-temporal hours allocation for singles given total hours
        for iP,power in enumerate(par.grid_power):
            for i,H_tot in enumerate(par.grid_Htot):
                kids = 0
                sol.pre_Htot_Hw_kid[iP,i], sol.pre_Htot_Hm_kid[iP,i] = solve_intraperiod_couple_hours(H_tot,power,kids,par)
                kids = 1
                sol.pre_Htot_Hw_nokid[iP,i], sol.pre_Htot_Hm_nokid[iP,i] = solve_intraperiod_couple_hours(H_tot,power,kids,par)
                #print(f"sol.pre_Htot_Hw_kid: {sol.pre_Htot_Hw_kid[iP,i]}")
                #print(f"sol.pre_Htot_Hm_kid: {sol.pre_Htot_Hm_kid[iP,i]}")
                #print(f"sol.pre_Htot_Hw_nokid: {sol.pre_Htot_Hw_nokid[iP,i]}")
                #print(f"sol.pre_Htot_Hm_nokid: {sol.pre_Htot_Hm_nokid[iP,i]}")
        # loop backwards
        for t in reversed(range(par.T)):
            print(f"t: {t}")
            self.solve_single(t)
            self.solve_couple(t)

        # total consumption
        sol.C_tot_couple = sol.Cw_priv_couple + sol.Cm_priv_couple + sol.C_pub_couple
        sol.C_tot_remain_couple = sol.Cw_priv_remain_couple + sol.Cm_priv_remain_couple + sol.C_pub_remain_couple
        sol.Cw_tot_single = sol.Cw_priv_single + sol.Cw_pub_single
        sol.Cm_tot_single = sol.Cm_priv_single + sol.Cm_pub_single

        # value of transitioning to singlehood. Done here because absorbing . it is the same as entering period as single.
        sol.Vw_trans_single = sol.Vw_single.copy()
        sol.Vm_trans_single = sol.Vm_single.copy()
        sol.Cw_priv_trans_single = sol.Cw_priv_single.copy()
        sol.Cm_priv_trans_single = sol.Cm_priv_single.copy()
        sol.Cw_pub_trans_single = sol.Cw_pub_single.copy()
        sol.Cm_pub_trans_single = sol.Cm_pub_single.copy()
        sol.Cw_tot_trans_single = sol.Cw_tot_single.copy()
        sol.Cm_tot_trans_single = sol.Cm_tot_single.copy()


    def solve_single(self, t):
        par = self.par
        sol = self.sol
        
        # loop through state variable: wealth
        for iN in range(par.num_n):  # addition of children
            for iA in range(par.num_A):  
                for iK in range(par.num_k):  # addition of human capital
                    
                    # index
                    idx = (t, iN, iA, iK)

                    for gender in ['woman', 'man']:
                        # Resources
                        A = par.grid_Aw[iA] if gender == 'woman' else par.grid_Am[iA]
                        kids = par.nw_grid[iN] if gender == 'woman' else par.nm_grid[iN]
                        K = par.kw_grid[iK] if gender == 'woman' else par.km_grid[iK]
                        

                        if t == (par.T - 1):  # terminal period
                            obj = lambda x: obj_last_single(self, x[0], A, K, gender, kids)
                            
                            # call optimizer
                            hours_min = np.fmax(-A / wage_func(self,K,gender) + 1.0e-5, 0.0)  # minimum amount of hours that ensures positive consumption
                            if iA == 0:
                                init_h = np.maximum(hours_min, 2.0)
                            else:
                                if gender == 'woman':
                                    init_h = np.array([sol.Hw_single[t, iN, iA - 1, iK]])
                                else:
                                    init_h = np.array([sol.Hm_single[t, iN, iA - 1, iK]])
                            
                            res = minimize(obj, init_h, bounds=((hours_min, np.inf),), method='L-BFGS-B')
                            
                            # Store results separately for each gender
                            if gender == 'woman':
                                sol.Cw_priv_single[idx], sol.Cw_pub_single[idx] = cons_last_single(self,res.x[0], A, K, gender)
                                sol.Hw_single[idx] = res.x[0]
                                sol.Vw_single[idx] = -res.fun
                            else:
                                sol.Cm_priv_single[idx], sol.Cm_pub_single[idx] = cons_last_single(self,res.x[0], A, K, gender)
                                sol.Hm_single[idx] = res.x[0]
                                sol.Vm_single[idx] = -res.fun

                            #print(f"C_w: {sol.Cw_priv_single[idx]}")
                            #print(f"Hw: {sol.Hw_single[idx]}")
                            #print(f"Vw: {sol.Vw_single[idx]}")
                            
                        else:  # earlier periods
                            # search over optimal total consumption, C
                            obj = lambda x: -self.value_of_choice_single(x[0], x[1], A, K, kids, gender, t)
                            # bounds on consumption 
                            lb_c = 0.000001  # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.000001  # avoid dividing with zero
                            ub_h = np.inf 

                            bounds = ((lb_c, ub_c), (lb_h, ub_h))
                            #print(f"bounds: {bounds}")
                            # call optimizer
                            idx_last = (t + 1, iN, iA, iK)
                            #print(f"idx_last: {idx_last}")
                            #print(f"sol.Cw_priv_single[idx_last]: {sol.Cw_priv_single[idx_last]}")
                            #print(f"sol.Hw_single[idx_last]: {sol.Hw_single[idx_last]}")
                            if gender == 'woman':
                                init = np.array([sol.Cw_priv_single[idx_last], sol.Hw_single[idx_last]])
                            else:
                                init = np.array([sol.Cm_priv_single[idx_last], sol.Hm_single[idx_last]])
                            
                            res = minimize(obj, init, bounds=bounds, method='L-BFGS-B', tol=1.0e-8) 

                            # Store results separately for each gender
                            if gender == 'woman':
                                sol.Cw_priv_single[idx], sol.Cw_pub_single[idx] = intraperiod_allocation_single(res.x[0], gender, par)
                                sol.Hw_single[idx] = res.x[1]
                                sol.Vw_single[idx] = -res.fun
                                #print(f"sol.Vw_single: {sol.Vw_single[idx]}")
                                #print(f"sol.Cw_priv_single: {sol.Cw_priv_single[idx]}")
                                #print(f"sol.Cw_pub_single: {sol.Cw_pub_single[idx]}")
                                #print(f"sol.Hw_single: {sol.Hw_single[idx]}")
                            else:
                                sol.Cm_priv_single[idx], sol.Cm_pub_single[idx] = intraperiod_allocation_single(res.x[0], gender, par)
                                sol.Hm_single[idx] = res.x[1]
                                sol.Vm_single[idx] = -res.fun
                                #print(f"sol.Vm_single: {sol.Vm_single[idx]}")
                                #print(f"sol.Cm_priv_single: {sol.Cm_priv_single[idx]}")
                                #print(f"sol.Cm_pub_single: {sol.Cm_pub_single[idx]}")
                                #print(f"sol.Hm_single: {sol.Hm_single[idx]}")


    def solve_couple(self, t):
        par = self.par
        sol = self.sol

        remain_Vw, remain_Vm, remain_Cw_priv, remain_Cm_priv, remain_C_pub = (
            np.ones(par.num_power), np.ones(par.num_power), np.ones(par.num_power), np.ones(par.num_power), np.ones(par.num_power))
        remain_Hw, remain_Hm = np.ones(par.num_power), np.ones(par.num_power)  # For storing hours worked

        Vw_next = None
        Vm_next = None
        for iN, kids in enumerate(par.nw_grid):
            for iL, love in enumerate(par.grid_love):
                for iA, A in enumerate(par.grid_A):  # Add human capital state variable!
                    for iKw, Kw in enumerate(par.kw_grid):
                        for iKm, Km in enumerate(par.km_grid):

                            starting_val = None
                            for iP, power in enumerate(par.grid_power):  # Loop over different power levels!
                                # Continuation values
                                if t < (par.T - 1):
                                    Vw_next = self.sol.Vw_couple[t + 1, iP]
                                    Vm_next = self.sol.Vm_couple[t + 1, iP]

                                # Starting values
                                if iP > 0:
                                    C_tot_last = remain_Cw_priv[iP - 1] + remain_Cm_priv[iP - 1] + remain_C_pub[iP - 1]
                                    #print(f"C_tot_last: {C_tot_last}")
                                    starting_val = np.array([C_tot_last])

                                # Solve problem if remaining married
                                remain_Cw_priv[iP], remain_Cm_priv[iP], remain_C_pub[iP], remain_Hw[iP], remain_Hm[iP], remain_Vw[iP], remain_Vm[iP] = self.solve_remain_couple(
                                    t, A, Kw, Km, iL, iP, power, Vw_next, Vm_next, kids, starting_val=starting_val)
                                #print(f"remain_Cw = {remain_Cw_priv[iP]}, remain_Cm = {remain_Cm_priv[iP]}, remain_Cpub = {remain_C_pub[iP]}, remain_Hw = {remain_Hw[iP]}, remain_Hm = {remain_Hm[iP]}, remain_Vw = {remain_Vw[iP]}, remain_Vm = {remain_Vm[iP]}")
                                #print(np.shape(remain_Vw[iP]))

                                # Check the participation constraints - this applies the limited commitment bargaining scheme
                                idx_single_woman = (t, iN, iA, iKw)  # Index with children and HC - how to handle men and women?
                                idx_single_man = (t, iN, iA, iKm)
                                idx_couple = lambda iP: (t, iN, iP, iL, iA, iKw, iKm)

                                list_start_as_couple = (sol.Vw_couple, sol.Vm_couple, sol.Cw_priv_couple, sol.Cm_priv_couple, sol.C_pub_couple, sol.Hw_couple, sol.Hm_couple)
                                list_remain_couple = (remain_Vw, remain_Vm, remain_Cw_priv, remain_Cm_priv, remain_C_pub, remain_Hw, remain_Hm)
                                list_trans_to_single = (sol.Vw_single, sol.Vm_single, sol.Cw_priv_single, sol.Cm_priv_single, sol.Cw_pub_single, sol.Hw_single, sol.Hm_single)  # Last input here not important in case of divorce

                                Sw = remain_Vw - sol.Vw_single[idx_single_woman]
                                Sm = remain_Vm - sol.Vm_single[idx_single_man]
                                #print(f"Sw: {Sw}, Sm: {Sm}")
                                #print(f"shape of Sw: {np.shape(Sw)}")
                                #print(f"shape of Sm: {np.shape(Sm)}")
                                check_participation_constraints(sol.power_idx, sol.power, Sw, Sm, idx_single_woman, idx_single_man, idx_couple, list_start_as_couple, list_remain_couple, list_trans_to_single, par)

                                # Save remain values
                                for iP, power in enumerate(par.grid_power):  # Loop over different power levels!
                                    idx = (t, iP, iL, iA)
                                    sol.Cw_priv_remain_couple[idx] = remain_Cw_priv[iP]
                                    sol.Cm_priv_remain_couple[idx] = remain_Cm_priv[iP]
                                    sol.C_pub_remain_couple[idx] = remain_C_pub[iP]
                                    sol.Hw_remain_couple[idx] = remain_Hw[iP]
                                    sol.Hm_remain_couple[idx] = remain_Hm[iP]
                                    sol.Vw_remain_couple[idx] = remain_Vw[iP]
                                    sol.Vm_remain_couple[idx] = remain_Vm[iP]


    


    def value_of_choice_couple(self, C_tot, H_tot, t, assets, Kw, Km, iL, iP, power, Vw_next, Vm_next, kids):
        sol = self.sol
        par = self.par

        love = par.grid_love[iL]
        #print(f"Inputs - C_tot: {C_tot}, H_tot: {H_tot}, t: {t}, assets: {assets}, Kw: {Kw}, Km: {Km}, iL: {iL}, iP: {iP}, power: {power}, kids: {kids}")
        # Current utility from consumption allocation
        Cw_priv, Cm_priv, C_pub = intraperiod_allocation(C_tot, iP, sol, par)
        Hw, Hm = intraperiod_allocation_hours(H_tot, iP, sol, kids, par)
        #print(f"Cw_priv: {Cw_priv}, Cm_priv: {Cm_priv}, C_pub: {C_pub}, Hw: {Hw}, Hm: {Hm}")
        Vw = usr.util(Cw_priv, C_pub, Hw, woman, kids, par, love)
        Vm = usr.util(Cm_priv, C_pub, Hm, man, kids, par, love)

        # Add continuation value
        if t < (par.T - 1):
            # Calculate income based on work hours and human capital
            income_woman = wage_func(self, Kw, woman) * Hw
            income_man = wage_func(self, Km, man) * Hm
            total_income = income_woman + income_man

            a_next = assets + total_income - C_tot  # Next period's assets
            k_next_woman = Kw + Hw  # Next period's human capital for woman
            k_next_man = Km + Hm  # Next period's human capital for man

            # Ensure variables are numpy arrays
            a_next_arr = np.array([a_next]) if np.isscalar(a_next) else a_next
            k_next_woman_arr = np.array([k_next_woman]) if np.isscalar(k_next_woman) else k_next_woman
            k_next_man_arr = np.array([k_next_man]) if np.isscalar(k_next_man) else k_next_man

            # Ensure Vw_next and Vm_next are 3D numpy arrays
            Vw_next_arr = np.array(Vw_next)
            Vm_next_arr = np.array(Vm_next)

            love_next_vec = love + par.grid_shock_love

            # Perform interpolation
            interp_3d_vec(par.grid_love, par.grid_A, par.kw_grid, Vw_next_arr, love_next_vec, a_next_arr, k_next_woman_arr, sol.Vw_plus_vec)
            interp_3d_vec(par.grid_love, par.grid_A, par.km_grid, Vm_next_arr, love_next_vec, a_next_arr, k_next_man_arr, sol.Vm_plus_vec)

            EVw_plus_no_birth = sol.Vw_plus_vec @ par.grid_weight_love
            EVm_plus_no_birth = sol.Vm_plus_vec @ par.grid_weight_love

            # Child-birth considerations
            if kids >= (par.num_n - 1):
                # Cannot have more children
                EVw_plus_birth = EVw_plus_no_birth
                EVm_plus_birth = EVm_plus_no_birth
            else:
                kids_next = kids + 1

                # Interpolate future values for the next period with a new child
                Vw_next_with_birth = sol.Vw_couple[t + 1, kids_next]
                Vm_next_with_birth = sol.Vm_couple[t + 1, kids_next]

                interp_3d_vec(par.grid_love, par.grid_A, par.kw_grid, Vw_next_with_birth, love_next_vec, a_next_arr, k_next_woman_arr, sol.Vw_plus_vec)
                interp_3d_vec(par.grid_love, par.grid_A, par.km_grid, Vm_next_with_birth, love_next_vec, a_next_arr, k_next_man_arr, sol.Vm_plus_vec)

                EVw_plus_birth = sol.Vw_plus_vec @ par.grid_weight_love
                EVm_plus_birth = sol.Vm_plus_vec @ par.grid_weight_love

            EVw_plus = par.p_birth * EVw_plus_birth + (1 - par.p_birth) * EVw_plus_no_birth
            EVm_plus = par.p_birth * EVm_plus_birth + (1 - par.p_birth) * EVm_plus_no_birth

            Vw += par.beta * EVw_plus
            Vm += par.beta * EVm_plus

        # Return
        Val = power * Vw + (1.0 - power) * Vm
        return Val, Cw_priv, Cm_priv, C_pub, Hw, Hm, Vw, Vm


    def solve_remain_couple(self,t,assets,Kw,Km,iL,iP,power,Vw_next,Vm_next,kids,starting_val = None):
        par = self.par

        if t==(par.T-1): # Terminal period
            # In the last period, all assets are consumed
            C_tot = assets
            # Objective function only for working hours optimization
            obj = lambda x: - self.value_of_choice_couple(C_tot, x[0], t, assets, Kw, Km, iL, iP, power, Vw_next, Vm_next, kids)[0]
            x0 = np.array([0.4]) if starting_val is None else starting_val  # initial guess for H_tot

            # Optimize for working hours
            res = optimize.minimize(obj, x0, bounds=((1.0e-6, np.inf),), method='SLSQP')
            H_tot = res.x[0]
        else:
            print(f"C_tot: {type(C_tot)}, {C_tot.shape}")
            print(f"H_tot: {type(H_tot)}, {H_tot.shape}")
            print(f"assets: {type(assets)}, {assets.shape}")
            print(f"Kw: {type(Kw)}, {Kw.shape}")
            print(f"Km: {type(Km)}, {Km.shape}")
            print(f"iL: {type(iL)}, {iL.shape}")
            print(f"iP: {type(iP)}, {iP.shape}")
            print(f"power: {type(power)}, {power.shape}")
            print(f"Vw_next: {type(Vw_next)}, {Vw_next.shape}")
            print(f"Vm_next: {type(Vm_next)}, {Vm_next.shape}")
            print(f"kids: {type(kids)}, {kids.shape}")
            # objective function
            obj = lambda x: - self.value_of_choice_couple(x[0],x[1],t,assets,Kw, Km,iL,iP,power,Vw_next,Vm_next,kids)[0]
            x0 = np.array([0.4, 0.4]) if starting_val is None else starting_val #initial guess [C_tot, H_tot]

            # optimize
            res = optimize.minimize(obj,x0,bounds=((1.0e-6, np.inf),(1.0e-6, np.inf)) ,method='SLSQP') 
            C_tot = res.x[0]
            H_tot = res.x[1]

        # implied consumption allocation (re-calculation)
        _, Cw_priv, Cm_priv, C_pub, Hw, Hm, Vw, Vm = self.value_of_choice_couple(C_tot,H_tot,t,assets,Kw,Km,iL,iP,power,Vw_next,Vm_next, kids) # throw out value?

        # return objects
        return Cw_priv, Cm_priv, C_pub, Hw, Hm, Vw, Vm
    

    def value_of_choice_single(self,C_tot,hours,assets,capital,kids,gender,t):

        # a. unpack
        par = self.par
        sol = self.sol
        #print(f"C_tot: {C_tot}")
        #b. Specify consumption levels. 
        # flow-utility
        C_priv = usr.cons_priv_single(C_tot,gender,par)
        #print(f"C_priv: {C_priv}")
        C_pub = C_tot - C_priv
        
        # b. penalty for violating bounds. 
        penalty = 0.0
        if C_tot < 0.0:
            penalty += C_tot*1_000.0
            C_tot = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = usr.util(C_priv, C_pub,hours,gender,kids, par)
        #print(f"util: {util}")
        # d. *expected* continuation value from savings
        income = wage_func(self, capital, gender) * hours
        a_next = (1.0+par.r)*(assets + income - C_tot)
        k_next = capital + hours
        
        # Look over V_next for both genders:
        if gender == 'women':
            kids_next = kids
            V_next = sol.Vw_single[t + 1, kids_next]
            
            V_next_no_birth = interp_2d(par.grid_Aw,par.Kw_grid,V_next,a_next,k_next)
            
            # birth
            if (kids>=(par.num_n-1)):
                # cannot have more children
                V_next_birth = V_next_no_birth
            else:
                kids_next = kids + 1
                V_next = sol.Vw_single[t + 1, kids_next]
                V_next_birth = interp_2d(par.grid_Aw,par.Kw_grid,V_next,a_next,k_next)
            
        else:
            kids_next = kids
            V_next = sol.Vm_single[t + 1, kids_next]
            V_next_no_birth = interp_2d(par.grid_Am,par.km_grid,V_next,a_next,k_next)
            # birth
            if (kids>=(par.num_n-1)):
                # cannot have more children
                V_next_birth = V_next_no_birth
            else:
                kids_next = kids + 1
                V_next = sol.Vm_single[t + 1, kids_next]
                V_next_birth = interp_2d(par.grid_Am,par.km_grid,V_next,a_next,k_next)
                
        EV_next = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth
        
        # e. return value of choice (including penalty)
        return util + par.beta*EV_next + penalty
        
   
    def simulate(self):
        sol = self.sol
        sim = self.sim
        par = self.par

        for i in range(par.simN):
            for t in range(par.simT):

                # state variables
                if t==0:
                    A_lag = sim.init_A[i]
                    Aw_lag = sim.init_Aw[i]
                    Am_lag = sim.init_Am[i]
                    couple_lag = sim.init_couple[i]
                    power_idx_lag = sim.init_power_idx[i]
                    love = sim.love[i,t] = sim.init_love[i]

                else:
                    A_lag = sim.A[i,t-1]
                    Aw_lag = sim.Aw[i,t-1]
                    Am_lag = sim.Am[i,t-1]
                    couple_lag = sim.couple[i,t-1]
                    power_idx_lag = sim.power_idx[i,t-1]
                    love = sim.love[i,t]
                
                power_lag = par.grid_power[power_idx_lag]

                # first check if they want to remain together and what the bargaining power will be if they do.
                if couple_lag:                   

                    # value of transitioning into singlehood
                    Vw_single = linear_interp.interp_1d(par.grid_Aw,sol.Vw_trans_single[t],Aw_lag)
                    Vm_single = linear_interp.interp_1d(par.grid_Am,sol.Vm_trans_single[t],Am_lag)

                    idx = (t,power_idx_lag)
                    Vw_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                    Vm_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                    if ((Vw_couple_i>=Vw_single) & (Vm_couple_i>=Vm_single)):
                        power_idx = power_idx_lag

                    else:
                        # value of partnerhip for all levels of power
                        Vw_couple = np.zeros(par.num_power)
                        Vm_couple = np.zeros(par.num_power)
                        for iP in range(par.num_power):
                            idx = (t,iP)
                            Vw_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                            Vm_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                        # check participation constraint
                        Sw = Vw_couple - Vw_single
                        Sm = Vm_couple - Vm_single
                        power_idx = update_bargaining_index(Sw,Sm,power_idx_lag, par)

                    # infer partnership status
                    if power_idx < 0.0: # divorce is coded as -1
                        sim.couple[i,t] = False

                    else:
                        sim.couple[i,t] = True

                else: # remain single

                    sim.couple[i,t] = False

                # update behavior
                if sim.couple[i,t]:
                    
                    # optimal consumption allocation if couple
                    sol_C_tot = sol.C_tot_couple[t,power_idx] 
                    C_tot = linear_interp.interp_2d(par.grid_love,par.grid_A,sol_C_tot,love,A_lag)

                    sim.Cw_priv[i,t], sim.Cm_priv[i,t], C_pub = intraperiod_allocation(C_tot,power_idx,sol,par) #check this!
                    sim.Cw_pub[i,t] = C_pub
                    sim.Cm_pub[i,t] = C_pub

                    # update end-of-period states
                    M_resources = usr.resources_couple(A_lag,par) 
                    sim.A[i,t] = M_resources - sim.Cw_priv[i,t] - sim.Cm_priv[i,t] - sim.Cw_pub[i,t]
                    if t<(par.simT-1):
                        sim.love[i,t+1] = love + par.sigma_love*sim.draw_love[i,t+1]

                    # in case of divorce
                    sim.Aw[i,t] = par.div_A_share * sim.A[i,t]
                    sim.Am[i,t] = (1.0-par.div_A_share) * sim.A[i,t]

                    sim.power_idx[i,t] = power_idx
                    sim.power[i,t] = par.grid_power[sim.power_idx[i,t]]

                else: # single

                    # pick relevant solution for single, depending on whether just became single
                    idx_sol_single = t
                    sol_single_w = sol.Cw_tot_trans_single[idx_sol_single]
                    sol_single_m = sol.Cm_tot_trans_single[idx_sol_single]
                    if (power_idx_lag<0):
                        sol_single_w = sol.Cw_tot_single[idx_sol_single]
                        sol_single_m = sol.Cm_tot_single[idx_sol_single]

                    # optimal consumption allocations
                    Cw_tot = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw_lag)
                    Cm_tot = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am_lag)
                    
                    sim.Cw_priv[i,t],sim.Cw_pub[i,t] = intraperiod_allocation_single(Cw_tot,woman,par)
                    sim.Cm_priv[i,t],sim.Cm_pub[i,t] = intraperiod_allocation_single(Cm_tot,man,par)

                    # update end-of-period states
                    Mw = usr.resources_single(Aw_lag,woman,par)
                    Mm = usr.resources_single(Am_lag,man,par) 
                    sim.Aw[i,t] = Mw - sim.Cw_priv[i,t] - sim.Cw_pub[i,t]
                    sim.Am[i,t] = Mm - sim.Cm_priv[i,t] - sim.Cm_pub[i,t]

                    # not updated: nans
                    # sim.power[i,t] = np.nan
                    # sim.love[i,t+1] = np.nan 
                    # sim.A[i,t] = np.nan

                    sim.power_idx[i,t] = -1

        # total consumption
        sim.Cw_tot = sim.Cw_priv + sim.Cw_pub
        sim.Cm_tot = sim.Cm_priv + sim.Cm_pub
        sim.C_tot = sim.Cw_priv + sim.Cm_priv + sim.Cw_pub
    

############
# routines #
############
def intraperiod_allocation_single(C_tot,gender,par):
    C_priv = usr.cons_priv_single(C_tot,gender,par)
    C_pub = C_tot - C_priv
    return C_priv,C_pub

def intraperiod_allocation(C_tot,iP,sol,par):
    #print(f"Entering intraperiod_allocation with C_tot: {C_tot}, iP: {iP}")
    #print(f"Entering intraperiod_allocation with C_tot: {C_tot}, iP: {iP}")
    # interpolate pre-computed solution
    j1 = binary_search(0,par.num_Ctot,par.grid_Ctot,C_tot)
    Cw_priv = _interp_1d(par.grid_Ctot,sol.pre_Ctot_Cw_priv[iP],C_tot,j1)
    Cm_priv = _interp_1d(par.grid_Ctot,sol.pre_Ctot_Cm_priv[iP],C_tot,j1)
    
    C_pub = C_tot - Cw_priv - Cm_priv 
    #print(f"Cw_priv: {Cw_priv}, Cm_priv: {Cm_priv}, C_pub: {C_pub}")
    #print(f"Cw_priv: {Cw_priv}, Cm_priv: {Cm_priv}, C_pub: {C_pub}")
    return Cw_priv, Cm_priv, C_pub


def intraperiod_allocation_hours(H_tot,iP,sol,kids,par):
    if kids == 0:
        # interpolate pre-computed solution
        j1 = binary_search(0,par.num_Htot,par.grid_Htot,H_tot)
        Hw = _interp_1d(par.grid_Htot,sol.pre_Htot_Hw_nokid[iP],H_tot,j1)
        Hm = _interp_1d(par.grid_Htot,sol.pre_Htot_Hm_nokid[iP],H_tot,j1)
    else:
        # interpolate pre-computed solution
        j1 = binary_search(0,par.num_Htot,par.grid_Htot,H_tot)
        Hw = _interp_1d(par.grid_Htot,sol.pre_Htot_Hw_kid[iP],H_tot,j1)
        Hm = _interp_1d(par.grid_Htot,sol.pre_Htot_Hm_kid[iP],H_tot,j1)
    return Hw, Hm

def solve_intraperiod_couple(C_tot, power, kids, par, hours_woman=0.0, hours_man=0.0, starting_val=None):
    
    # setup estimation. Impose constraint that C_tot = Cw+Cm+C
    bounds = optimize.Bounds([0, 0], [C_tot, C_tot], keep_feasible=True)
    
    obj = lambda x: - (power * usr.util(x[0], C_tot - np.sum(x), hours_woman, woman, kids, par) + (1.0 - power) * usr.util(x[1], C_tot - np.sum(x), hours_man, man, kids, par))

    # estimate
    x0 = np.array([C_tot / 3, C_tot / 3]) if starting_val is None else starting_val
    
    try:
        res = optimize.minimize(obj, x0, bounds=bounds)
    except ValueError as e:
        print(f"Optimization failed: {e}")
        return np.nan, np.nan, np.nan

    if not np.isfinite(res.x).all() or np.iscomplex(res.x).any():
        return np.nan, np.nan, np.nan

    # unpack
    Cw_priv = np.real(res.x[0])
    Cm_priv = np.real(res.x[1])
    C_pub = np.real(C_tot - np.sum(res.x))
    
    return Cw_priv, Cm_priv, C_pub


def solve_intraperiod_couple_hours(H_tot, power, kids, par):
    # Define a small positive value to avoid zero
    epsilon = 1e-5

    # Ensure valid bounds
    if H_tot <= 2 * epsilon:
        epsilon = H_tot / 3
    # setup estimation. Impose constraint that H_tot = Hw + Hm
    bounds = optimize.Bounds([epsilon, epsilon], [H_tot, H_tot], keep_feasible=True)
    
    obj = lambda x: - (power * usr.util(0.5, 0.5, x[0], woman, kids, par) + (1.0 - power) * usr.util(0.5, 0.5, H_tot - x[0], man, kids,par))

    # estimate
    x0 = np.array([H_tot / 2, H_tot / 2]) # intial guess, how about a starting value?
    res = optimize.minimize(obj, x0, bounds=bounds)
    #unpack
    Hw = res.x[0]
    Hm = H_tot - Hw
    return Hw, Hm


def check_participation_constraints(power_idx, power, Sw, Sm, idx_single_woman, idx_single_man, idx_couple, list_couple, list_raw, list_single, par):
    
    # check the participation constraints. Array
    min_Sw = np.min(Sw)
    min_Sm = np.min(Sm)
    max_Sw = np.max(Sw)
    max_Sm = np.max(Sm)

    if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
        for iP in range(par.num_power):
            idx = idx_couple(iP)
            for i, key in enumerate(list_couple):
                list_couple[i][idx] = list_raw[i][iP]

            if all(x < s for x, s in zip(idx, power_idx.shape)):
                power_idx[idx] = iP
                power[idx] = par.grid_power[iP]
            if all(x < s for x, s in zip(idx, power_idx.shape)):
                power_idx[idx] = iP
                power[idx] = par.grid_power[iP]

    elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
        for iP in range(par.num_power):
            idx = idx_couple(iP)
            for i, key in enumerate(list_couple):
                # Debugging prints
                #print(f"Checking list_couple[{i}][{idx}] and list_single[{i}][{idx_single_woman}]")
                #print(f"Shapes: list_couple[{i}].shape = {list_couple[i].shape}, list_single[{i}].shape = {list_single[i].shape}")
                #print(f"indices: idx_single_woman = {idx_single_woman}, idx_single_man = {idx_single_man}, idx = {idx}")
                
                # Ensure indices are within valid range
                if isinstance(idx_single_woman, tuple) and isinstance(idx, tuple):
                    if all(x < s for x, s in zip(idx_single_woman, list_single[i].shape)) and all(x < s for x, s in zip(idx, list_couple[i].shape)):
                        list_couple[i][idx] = list_single[i][idx_single_woman]
                    if all(x < s for x, s in zip(idx_single_man, list_single[i].shape)) and all(x < s for x, s in zip(idx, list_couple[i].shape)):
                        list_couple[i][idx] = list_single[i][idx_single_man]
                # Debugging prints
                #print(f"Checking list_couple[{i}][{idx}] and list_single[{i}][{idx_single_woman}]")
                #print(f"Shapes: list_couple[{i}].shape = {list_couple[i].shape}, list_single[{i}].shape = {list_single[i].shape}")
                #print(f"indices: idx_single_woman = {idx_single_woman}, idx_single_man = {idx_single_man}, idx = {idx}")
                
                # Ensure indices are within valid range
                if isinstance(idx_single_woman, tuple) and isinstance(idx, tuple):
                    if all(x < s for x, s in zip(idx_single_woman, list_single[i].shape)) and all(x < s for x, s in zip(idx, list_couple[i].shape)):
                        list_couple[i][idx] = list_single[i][idx_single_woman]
                    if all(x < s for x, s in zip(idx_single_man, list_single[i].shape)) and all(x < s for x, s in zip(idx, list_couple[i].shape)):
                        list_couple[i][idx] = list_single[i][idx_single_man]

            if all(x < s for x, s in zip(idx, power_idx.shape)):
                power_idx[idx] = -1
                power[idx] = -1.0
            if all(x < s for x, s in zip(idx, power_idx.shape)):
                power_idx[idx] = -1
                power[idx] = -1.0

    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 1 #0 # in case there is no crossing, this will be the correct value
        Low_m = par.num_power-1-1 #par.num_power-1 # in case there is no crossing, this will be the correct value
        for iP in range(par.num_power-1):
            if (Sw[iP]<0) & (Sw[iP+1]>=0):
                Low_w = iP+1
                
            if (Sm[iP]>=0) & (Sm[iP+1]<0):
                Low_m = iP

        # b. interpolate the surplus of each member at indifference points
        # women indifference
        id = Low_w-1
        denom = (par.grid_power[id+1] - par.grid_power[id])
        ratio_w = (Sw[id+1] - Sw[id])/denom
        ratio_m = (Sm[id+1] - Sm[id])/denom
        power_at_zero_w = par.grid_power[id] - Sw[id]/ratio_w
        Sm_at_zero_w = Sm[id] + ratio_m*( power_at_zero_w - par.grid_power[id] )

        # men indifference
        id = Low_m
        denom = (par.grid_power[id+1] - par.grid_power[id])
        ratio_w = (Sw[id+1] - Sw[id])/denom
        ratio_m = (Sm[id+1] - Sm[id])/denom
        power_at_zero_m = par.grid_power[id] - Sm[id]/ratio_m
        Sw_at_zero_m = Sw[id] + ratio_w*( power_at_zero_m - par.grid_power[id] )

        # update the outcomes
        for iP in range(par.num_power):
            idx = idx_couple(iP)
    
            # woman wants to leave
            if iP < Low_w: 
                if Sm_at_zero_w > 0: # man happy to shift some bargaining power
                    for i, key in enumerate(list_couple):
                        if iP == 0:
                            list_couple[i][idx] = interp_1d(par.grid_power, list_raw[i], power_at_zero_w)#, Low_w-1) 
                        else:
                            list_couple[i][idx] = list_couple[i][idx_couple(0)]; # re-use that the interpolated values are identical

                    if all(x < s for x, s in zip(idx, power_idx.shape)):
                        power_idx[idx] = Low_w
                        power[idx] = power_at_zero_w
                    
                else: # divorce
                    for i, key in enumerate(list_couple):
                        list_couple[i][idx] = list_single[i][idx_single_woman]
                        list_couple[i][idx] = list_single[i][idx_single_man]
                    
                    if all(x < s for x, s in zip(idx, power_idx.shape)):
                        power_idx[idx] = -1
                        power[idx] = -1.0
                
            # man wants to leave
            elif iP > Low_m: 
                if Sw_at_zero_m > 0: # woman happy to shift some bargaining power
                    for i, key in enumerate(list_couple):
                        if (iP == Low_m + 1):
                            list_couple[i][idx] = interp_1d(par.grid_power, list_raw[i], power_at_zero_m, Low_m) 
                        else:
                            list_couple[i][idx] = list_couple[i][idx_couple(Low_m + 1)]; # re-use that the interpolated values are identical

                    if all(x < s for x, s in zip(idx, power_idx.shape)):
                        power_idx[idx] = Low_m
                        power[idx] = power_at_zero_m
                    
                else: # divorce
                    for i, key in enumerate(list_couple):
                        list_couple[i][idx] = list_single[i][idx_single_woman]
                        list_couple[i][idx] = list_single[i][idx_single_man]

                    if all(x < s for x, s in zip(idx, power_idx.shape)):
                        power_idx[idx] = -1
                        power[idx] = -1.0

            else: # no-one wants to leave
                for i, key in enumerate(list_couple):
                    list_couple[i][idx] = list_raw[i][iP]

                if all(x < s for x, s in zip(idx, power_idx.shape)):
                    power_idx[idx] = iP
                    power[idx] = par.grid_power[iP]









def update_bargaining_index(Sw,Sm,iP, par):
    
    # check the participation constraints. Array
    min_Sw = np.min(Sw)
    min_Sm = np.min(Sm)
    max_Sw = np.max(Sw)
    max_Sm = np.max(Sm)

    if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
        return iP

    elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
        return -1

    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 0 # in case there is no crossing, this will be the correct value
        Low_m = par.num_power-1 # in case there is no crossing, this will be the correct value
        for _iP in range(par.num_power-1):
            if (Sw[_iP]<0) & (Sw[_iP+1]>=0):
                Low_w = _iP+1
                
            if (Sm[_iP]>=0) & (Sm[_iP+1]<0):
                Low_m = iP

        # update the outcomes
        # woman wants to leave
        if iP<Low_w: 
            if Sm[Low_w] > 0: # man happy to shift some bargaining power
                return Low_w
                
            else: # divorce
                return -1
            
        # man wants to leave
        elif iP>Low_m: 
            if Sw[Low_m] > 0: # woman happy to shift some bargaining power
                return Low_m
                
            else: # divorce
                return -1

        else: # no-one wants to leave
            return iP

def wage_func(self,capital,gender):
    # before tax wage rate
    par = self.par

    if gender == woman:
        constant = par.wage_const_1
        return_K = par.wage_K_1
    else:
        constant = par.wage_const_2
        return_K = par.wage_K_2
    return np.exp(constant + return_K * capital)

def child_tran(self,hours1,hours2,income_hh,kids):
    par = self.par
    if kids<1:
        return 0.0
    
    else:
        C1 = par.uncon_uni                           #unconditional, universal transfer (>0)
        C2 = np.fmax(par.means_level - par.means_slope*income_hh , 0.0) #means-tested transfer (>0)
        # child-care related (net-of-subsidy costs)
        both_work = (hours1>0) * (hours2>0)
        C3 = par.cond*both_work                      #all working couples has this net cost (<0)
        C4 = par.cond_high*both_work*(income_hh>0.5) #low-income couples do not have this net-cost (<0)

    return C1+C2+C3+C4

def resources_couple(self,par,A, capital1, capital2, hours1, hours2):
    par = par.self
    return par.R*A + wage_func(self,capital1,sex = 1)*hours1 + wage_func(self, capital2, sex = 2)*hours2

def resources_single(self,capital, hours, A, gender,par):
    income = wage_func(self, capital, sex = 1)*hours
    if gender == man:
        income = wage_func(self, capital, sex = 2)*hours

    return par.R*A + income

def obj_last_single(self,hours,assets,capital,gender,kids): #remember to add kids!
    par = self.par
    conspriv = cons_last_single(self,hours,assets,capital,gender)[0]
    conspub = cons_last_single(self,hours,assets,capital,gender)[1]
    
    return - usr.util(conspriv,conspub, hours, gender, kids, par)

def obj_last_couple(self, hours_man, hours_women, assets, capital1, capital2, power, love, kids):
    par = self.par

    return - (power*usr.util(conspriv1,conspub1,kids) + (1.0-power)*usr.util(conspriv2,conspub2,kids))

def cons_last_single(self,hours,assets,capital, gender):
    #This returns C_priv, C_pub for singles in the last period
    par = self.par
    income = wage_func(self,capital,gender)*hours
    #Consume everything in the last period
    C_tot = assets + income
    conspriv = usr.cons_priv_single(C_tot, gender, par)
    conspub = C_tot - conspriv

    return conspriv, conspub
    
