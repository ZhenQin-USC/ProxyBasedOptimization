import sys

# sys.path.append(r'/')
from Function.source_func import call_simulation, moving_files
from pyDOE2 import *
from copy import deepcopy
from scipy.optimize import Bounds, LinearConstraint, minimize


def sample_optimizer(x00, bounds, lcon, verbose):
    def objfun(x, x00=x00):
        return 0.5 * (x - x00) @ (x - x00)

    def gradfun(x, x00=x00):
        return x - x00

    res = minimize(objfun, x00, method='trust-constr', jac=gradfun, hess=None,
                   bounds=bounds, constraints=lcon, tol=None,
                   options={'xtol': 1e-12, 'gtol': 1e-12,
                            'barrier_tol': 1e-08, 'sparse_jacobian': None,
                            'maxiter': 1e3, 'verbose': verbose, 'finite_diff_rel_step': None,
                            'initial_constr_penalty': 1.0, 'initial_tr_radius': 1.0,
                            'initial_barrier_parameter': 0.1, 'initial_barrier_tolerance': 0.1,
                            'factorization_method': None, 'disp': False})
    return res


def sample_mapping(x00, upper, lower, lcon):
    res = {'x': None, 'constr_violation': None}
    dQ = lcon.A @ x00 - lcon.lb  # delta Q need to be re-allocate to each well
    X_dimen = x00 * (upper - lower) + lower  # rate variable in original unit
    dlowers = X_dimen - lower  # lower available adjustable range
    duppers = upper - X_dimen  # upper available adjustable range
    weights = np.zeros_like(lcon.A)  # weights indicates well types (prod/injt)
    weights[lcon.A > 0] = 1
    mapped_sample = np.zeros_like(X_dimen)  # mapped samples in original unit
    for j in range(len(dQ)):
        if dQ[j] > 0:
            x_delta = deepcopy(dlowers)
        elif dQ[j] < 0:
            x_delta = deepcopy(duppers)
        dQp = weights[j] * x_delta * dQ[j] / (x_delta @ weights[j])
        temp_data = deepcopy(X_dimen - dQp)
        mapped_sample[weights[j] == 1] = temp_data[weights[j] == 1]
    mapped_x = (mapped_sample - lower) / (upper - lower)
    # check the bound constraints
    above = mapped_sample - upper
    below = lower - mapped_sample
    above = above[above > 0].tolist()
    below = below[below > 0].tolist()
    if len(above) is not 0:
        max_above = 10 * np.max(above)
    else:
        max_above = 0
    if len(below) is not 0:
        max_below = 10 * np.max(below)
    else:
        max_below = 0
    res['x'] = mapped_x
    res['constr_violation'] = max(lcon.A @ mapped_x - lcon.lb)
    res['constr_violation'] = max(res['constr_violation'], max_above, max_below)
    return res


def convert_x(history_x, nwell, nstage, pbound_proxy=None, ibound_proxy=None, pbound_sim=None, ibound_sim=None):
    niter = history_x.shape[0]
    if pbound_proxy is None:
        pbound_proxy = np.array([[324., 540., 540., 540., 300., 540.],
                                 [175.56, 290.4, 82.5, 82.5, 155.1, 82.5]])
    if ibound_proxy is None:
        ibound_proxy = np.array([[560., 740., 740., 740., 740., 560., 200., 740.],
                                 [0., 0., 0., 0., 0., 0., 0., 0.]])
    if pbound_sim is None:
        pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                               [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
    if ibound_sim is None:
        ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
    # de-normalize
    upper = np.repeat(np.concatenate((pbound_proxy[0, :], ibound_proxy[0, :]), axis=0), repeats=nstage)
    lower = np.repeat(np.concatenate((pbound_proxy[1, :], ibound_proxy[1, :]), axis=0), repeats=nstage)
    #X_opt = history_x[-1, :] * (upper - lower) + lower
    X_opt = history_x * (upper - lower) + lower
    # normalize
    upper = np.repeat(np.concatenate((pbound_sim[0, :], ibound_sim[0, :]), axis=0), repeats=nstage)
    lower = np.repeat(np.concatenate((pbound_sim[1, :], ibound_sim[1, :]), axis=0), repeats=nstage)
    x0_norm = ((X_opt - lower) / (upper - lower)).reshape(niter, nwell, nstage).transpose(0,2,1)

    return x0_norm


class GlobalSampler:
    def __init__(self, nstage, totaln, pbound_sim=None, ibound_sim=None, QTOTOMASS=1850, eps=0.03):
        if pbound_sim is None:
            pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                                   [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
        if ibound_sim is None:
            ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                                   [0, 0, 0, 0, 0, 0, 0, 0]])
        # default values
        self.eps = eps  # noise std
        self.Goals = ['Sampling']
        self.verbose = 0
        # input variables
        self.nstage = nstage  # number of control stages
        self.totaln = totaln  # total number of data points
        # self.history = history  # historical data
        self.pbound_sim = pbound_sim  # simulator's bound for production rate
        self.ibound_sim = ibound_sim  # simulator's bound for injection rate
        self.QTOTOMASS = QTOTOMASS  # total mass flow rate (default is 1850 kkg/hr)
        self.nprod = pbound_sim.shape[1]  #
        self.ninjt = ibound_sim.shape[1]  #
        self.nwell = self.nprod + self.ninjt  #
        # inner variables defined by sampler
        self.nsample = None  # number of samples to be generated
        self.Prod_bounds = None  # bounds for production rate
        self.Injt_bounds = None  # bounds for injection rate
        self.Enth_bounds = None  # bounds for specific enthalpy
        self.res = None  # results of optimization
        self.RATE_sim = None  # simulated production rate
        self.RATEi_sim = None  # simulated injection rate
        self.ENTH_sim = None  # simulated specific enthalpy
        self.y = None  # simulated net power generation
        self.mask = None  # mask for mapping
        self.ypred_mask = None  # mask filter for plotting ypred
        self.Samples = None  # filtered samples that is sent to "simulator" (call simulation)
        self.temp_sample = None  # temporary sample, not to be stored (need to be mapped back to feasible region)
        self.upper = None  # upper bound in original dimension
        self.lower = None  # lower bound in original dimension
        self.bounds = None  # bound constraint for mapping
        self.bounds_ = None  # truncated bound constraint for mapping
        self.lcon = None  # linear constraint for mapping
        self.lcon_ = None  # truncated linear constraint for mapping
        self.pre_samples = None  #

    def sampling(self, samples=None, keep=None, nsample=50):
        if keep is None:
            keep = []
        self.nsample = nsample
        self.keep = keep
        if samples is None:  # constrained LHS
            print('lhs')
            Samples, Well_Data = self.pre_sampling()
        else:  # augmented constrained LHS
            print('Aug_lhs')
            self.pre_samples = samples
            Samples, Well_Data = self.aug_sampling()
        # print(Samples.shape)
        return Samples, Well_Data

    def aug_sampling(self):
        pre_samples = self.pre_samples  # old_samples
        nsample_aug = self.nsample  # new_nsamples
        nsample_old = pre_samples.shape[0]
        nsample_tot = nsample_aug + nsample_old
        well_data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store mapped samples
        Well_Data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store raw samples
        # aug_lhs
        for i in range(self.nstage):
            for j in range(self.nwell):
                # create a list of interval
                # interval = 1 / nsample_tot
                interval_array = np.array([k / nsample_tot for k in range(0, nsample_tot)])
                # argmin broadcast(interval_array - samples)
                delta_x = np.abs(np.repeat(np.expand_dims(interval_array, axis=0), nsample_old, axis=0) -
                                 np.expand_dims(pre_samples[:, i, j], axis=1))  # ndim of interval x ndim of samples
                removed_index = np.argmin(delta_x, axis=-1)
                # 3D augmented sample for each stage and well
                aug_sample0 = np.random.permutation(list(set(interval_array) - set(interval_array[removed_index])))[
                              :nsample_aug]
                well_data00 = aug_sample0  # + interval*np.random.rand(nsample_aug,) # 3D noisy sample for single stage and well
                Well_Data[:, i, j] = well_data00
            self.well_data00 = Well_Data[:, i, :]  # self.well_data00 is stored and sent to mapping function
            well_data[:, i, :] = self.partial_mapping()  # return mapped samples --> self.samples --> sent to filter
        self.Samples = well_data
        return self.Samples, Well_Data

    def pre_sampling(self):
        # self.nsample = nsample
        well_data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store mapped samples
        Well_Data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store raw samples
        for i in range(self.nstage):
            noise = self.eps * np.random.rand(self.nsample, self.nwell) - self.eps / 2
            well_data00 = lhs(self.nwell, samples=self.nsample, criterion='centermaximin')
            well_data00 += noise
            well_data00[well_data00 > 1] = 1
            well_data00[well_data00 < 0] = 0
            self.well_data00 = well_data00  # well_data00 is stored and sent to mapping function
            samples = self.partial_mapping()  # return mapped samples --> self.samples --> sent to filter
            well_data[:, i, :] = samples  # mapped samples
            Well_Data[:, i, :] = well_data00  # raw samples
        self.Samples = well_data  # .transpose((0,2,1))
        return self.Samples, Well_Data

    def partial_mapping(self):
        temp_sample = deepcopy(self.well_data00)
        # print(temp_sample.shape)
        keep = self.keep
        QTOTOMASS = self.QTOTOMASS
        pbound_sim = self.pbound_sim
        ibound_sim = self.ibound_sim
        self.upper = np.concatenate((pbound_sim[0], ibound_sim[0]), axis=0)
        self.lower = np.concatenate((pbound_sim[1], ibound_sim[1]), axis=0)
        # constraints
        lb = np.zeros_like(temp_sample[0])
        ub = np.ones_like(temp_sample[0])
        kp = pbound_sim[0, :] - pbound_sim[1, :]
        bp = np.expand_dims((QTOTOMASS - sum(pbound_sim[1, :])), axis=0)
        ki = ibound_sim[0, :] - ibound_sim[1, :]
        bi = np.expand_dims((QTOTOMASS - sum(ibound_sim[1, :])), axis=0)
        Ap = np.expand_dims(np.concatenate((kp, np.zeros(self.ninjt, )), axis=0), axis=1)  # production rate
        Ai = np.expand_dims(np.concatenate((np.zeros(self.nprod, ), ki), axis=0), axis=1)  # injection rate
        Aeq = np.concatenate((Ap, Ai), axis=1).T
        beq = np.concatenate((bp, bi), axis=0)
        self.lcon = LinearConstraint(Aeq, beq, beq, keep_feasible=True)  # equality: production constraints
        self.bounds = Bounds(lb, ub, keep_feasible=True)  # bound

        # reset based on mask
        self.mask = list(set(range(len(temp_sample[0]))) - set(keep))  # mask
        # reset bound constraint
        self.bounds_ = deepcopy(self.bounds)
        self.bounds_.lb = self.bounds.lb[self.mask]
        self.bounds_.ub = self.bounds.ub[self.mask]
        # reset linear constraint
        self.lcon_ = deepcopy(self.lcon)
        self.lcon_.A = self.lcon.A[:, self.mask]
        # reset starting point
        samples = np.zeros_like(temp_sample)
        samples0 = deepcopy(temp_sample[:, self.mask])

        # mapping to feasible region
        for i in range(len(temp_sample)):
            self.lcon_.lb = self.lcon.lb - self.lcon.A[:, keep] @ temp_sample[i][keep]
            self.lcon_.ub = self.lcon.ub - self.lcon.A[:, keep] @ temp_sample[i][keep]
            try:
                res = sample_mapping(samples0[i], self.upper[self.mask], self.lower[self.mask], lcon=self.lcon_)
                # print('Mapping w Mask: No.{}'.format(i)) #print(res['constr_violation'])
                if res['constr_violation'] >= 1:
                    raise Exception("constraint violation is too large!")
                samples[i, self.mask] = res['x']
                samples[i, self.keep] = temp_sample[i, self.keep]
            except:
                res = sample_mapping(temp_sample[i], self.upper, self.lower, lcon=self.lcon)
                # print('Mapping wo Mask: No.{}'.format(i))
                samples[i, :] = res['x']
        self.Samples = deepcopy(samples)
        return self.Samples

    def simulator(self, samples, DATE, lsSTEP):
        self.y, self.ypred_mask, self.ENTH_sim, self.PBHP_sim, self.IBHP_sim, \
        self.PPump_sim, self.IPump_sim, self.RATE_sim, self.RATEi_sim = call_simulation(
            samples, DATE, lsSTEP, Prod_bounds=self.pbound_sim, Injt_bounds=self.ibound_sim, Goals=self.Goals)
        return self.y, self.ypred_mask, self.ENTH_sim, self.PBHP_sim, self.IBHP_sim, self.RATE_sim, self.RATEi_sim


class Sampler:
    def __init__(self, nstage, pbound_sim=None, ibound_sim=None, QTOTOMASS=None, eps=0.03):
        if pbound_sim is None:
            pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                                   [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
        if ibound_sim is None:
            ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                                   [0, 0, 0, 0, 0, 0, 0, 0]])
        # default values
        self.eps = eps  # noise std
        self.Goals = ['Sampling']
        self.verbose = 0
        # input variables
        self.nstage = nstage  # number of control stages
        # self.totaln = totaln  # total number of data points
        # self.history = history  # historical data
        self.pbound_sim = pbound_sim  # simulator's bound for production rate
        self.ibound_sim = ibound_sim  # simulator's bound for injection rate
        if QTOTOMASS is None:
            self.QTOTOMASS = 1850 * np.ones(nstage) # total mass flow rate (default is 1850 kkg/hr)
        else:
            self.QTOTOMASS = QTOTOMASS
        self.nprod = pbound_sim.shape[1]  #
        self.ninjt = ibound_sim.shape[1]  #
        self.nwell = self.nprod + self.ninjt  #
        # inner variables defined by sampler
        self.nsample = None  # number of samples to be generated
        self.Prod_bounds = None  # bounds for production rate
        self.Injt_bounds = None  # bounds for injection rate
        self.Enth_bounds = None  # bounds for specific enthalpy
        self.res = None  # results of optimization
        self.RATE_sim = None  # simulated production rate
        self.RATEi_sim = None  # simulated injection rate
        self.ENTH_sim = None  # simulated specific enthalpy
        self.y = None  # simulated net power generation
        self.mask = None  # mask for mapping
        self.ypred_mask = None  # mask filter for plotting ypred
        self.Samples = None  # filtered samples that is sent to "simulator" (call simulation)
        self.temp_sample = None  # temporary sample, not to be stored (need to be mapped back to feasible region)
        self.upper = None  # upper bound in original dimension
        self.lower = None  # lower bound in original dimension
        self.bounds = None  # bound constraint for mapping
        self.bounds_ = None  # truncated bound constraint for mapping
        self.lcon = None  # linear constraint for mapping
        self.lcon_ = None  # truncated linear constraint for mapping
        self.pre_samples = None  #

    def globalsampling(self, samples=None, keep=None, nsample=50):
        if keep is None:
            keep = []
        self.nsample = nsample
        self.keep = keep
        if samples is None:  # only constrained LHS
            print('lhs')
            Samples, Well_Data = self.pre_sampling()
        else:  # augmented constrained LHS
            print('Aug_lhs')
            self.pre_samples = samples
            Samples, Well_Data = self.aug_sampling()
        # print(Samples.shape)
        return Samples, Well_Data


    def localsampling(self, x0_norm=None, samples=None, keep=None, nsample=10, radius=0.1):
        # x0_norm is the control of current iteration
        # samples is the sample pool for retraining
        # keep index
        if keep is None:
            keep = []
        self.keep = keep

        # Creat upper & lower based on current iteration
        upper = x0_norm + radius
        lower = x0_norm - radius
        upper[upper > 1] = 1
        lower[lower < 0] = 0

        # Retain samples within the range
        retain_index = np.logical_and(samples >= lower, samples <= upper).all(axis=(1, 2))
        # retain_index[0:2] = True
        retain_index = np.where(retain_index == True)[0]

        # local sampling is always augmented lhs
        print('Aug_lhs')
        self.pre_samples = np.concatenate((samples[retain_index, :], x0_norm[None, :, :]), axis=0)
        self.nsample = nsample - self.pre_samples.shape[0]
        Samples, Well_Data = self.aug_sampling(lower=lower, upper=upper)

        return Samples, self.pre_samples, Well_Data

    def aug_sampling(self, lower=None, upper=None):
        if lower is None:
            lower = np.zeros(self.nstage)
        if upper is None:
            upper = np.ones(self.nstage)

        pre_samples = self.pre_samples  # old_samples
        nsample_aug = self.nsample  # new_nsamples
        nsample_old = pre_samples.shape[0]
        nsample_tot = nsample_aug + nsample_old
        well_data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store mapped samples
        Well_Data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store raw samples
        # aug_lhs
        for i in range(self.nstage):
            for j in range(self.nwell):
                # create a list of interval
                # interval = 1 / nsample_tot
                interval_array = np.array([k / nsample_tot for k in range(0, nsample_tot)])*(upper[i,j]-lower[i,j])+lower[i,j]

                # argmin broadcast(interval_array - samples)
                delta_x = np.abs(np.repeat(np.expand_dims(interval_array, axis=0), nsample_old, axis=0) -
                                 np.expand_dims(pre_samples[:, i, j], axis=1))  # ndim of interval x ndim of samples
                removed_index = np.argmin(delta_x, axis=-1)

                # 3D augmented sample for each stage and well
                aug_sample0 = np.random.permutation(list(set(interval_array) - set(interval_array[removed_index])))[
                              :nsample_aug]
                well_data00 = aug_sample0  # + interval*np.random.rand(nsample_aug,) # 3D noisy sample for single stage and well
                Well_Data[:, i, j] = well_data00
            self.well_data00 = Well_Data[:, i, :]  # self.well_data00 is stored and sent to mapping function
            well_data[:, i, :] = self.partial_mapping()  # return mapped samples --> self.samples --> sent to filter
        self.Samples = well_data
        return self.Samples, Well_Data

    def pre_sampling(self, lower=None, upper=None):
        # self.nsample = nsample
        if lower is None:
            lower = np.zeros(self.nstage)
        if upper is None:
            upper = np.ones(self.nstage)
        well_data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store mapped samples
        Well_Data = np.zeros((self.nsample, self.nstage, self.nwell))  # to store raw samples
        for i in range(self.nstage):
            noise = self.eps * np.random.rand(self.nsample, self.nwell) - self.eps / 2
            well_data00 = lhs(self.nwell, samples=self.nsample, criterion='centermaximin')*(upper[i]-lower[i])+lower[i]
            well_data00 += noise
            well_data00[well_data00 > 1] = 1
            well_data00[well_data00 < 0] = 0
            self.well_data00 = well_data00  # well_data00 is stored and sent to mapping function
            samples = self.partial_mapping(stage=i)  # return mapped samples --> self.samples --> sent to filter
            well_data[:, i, :] = samples  # mapped samples
            Well_Data[:, i, :] = well_data00  # raw samples
        self.Samples = well_data  # .transpose((0,2,1))
        return self.Samples, Well_Data

    def partial_mapping(self, stage=0):
        temp_sample = deepcopy(self.well_data00)
        # print(temp_sample.shape)
        keep = self.keep
        QTOTOMASS = self.QTOTOMASS[stage]
        pbound_sim = self.pbound_sim
        ibound_sim = self.ibound_sim
        self.upper = np.concatenate((pbound_sim[0], ibound_sim[0]), axis=0)
        self.lower = np.concatenate((pbound_sim[1], ibound_sim[1]), axis=0)
        # constraints
        lb = np.zeros_like(temp_sample[0])
        ub = np.ones_like(temp_sample[0])
        kp = pbound_sim[0, :] - pbound_sim[1, :]
        bp = np.expand_dims((QTOTOMASS - sum(pbound_sim[1, :])), axis=0)
        ki = ibound_sim[0, :] - ibound_sim[1, :]
        bi = np.expand_dims((QTOTOMASS - sum(ibound_sim[1, :])), axis=0)
        Ap = np.expand_dims(np.concatenate((kp, np.zeros(self.ninjt, )), axis=0), axis=1)  # production rate
        Ai = np.expand_dims(np.concatenate((np.zeros(self.nprod, ), ki), axis=0), axis=1)  # injection rate
        Aeq = np.concatenate((Ap, Ai), axis=1).T
        beq = np.concatenate((bp, bi), axis=0)
        self.lcon = LinearConstraint(Aeq, beq, beq, keep_feasible=True)  # equality: production constraints
        self.bounds = Bounds(lb, ub, keep_feasible=True)  # bound

        # reset based on mask
        self.mask = list(set(range(len(temp_sample[0]))) - set(keep))  # mask
        # reset bound constraint
        self.bounds_ = deepcopy(self.bounds)
        self.bounds_.lb = self.bounds.lb[self.mask]
        self.bounds_.ub = self.bounds.ub[self.mask]
        # reset linear constraint
        self.lcon_ = deepcopy(self.lcon)
        self.lcon_.A = self.lcon.A[:, self.mask]
        # reset starting point
        samples = np.zeros_like(temp_sample)
        samples0 = deepcopy(temp_sample[:, self.mask])

        # mapping to feasible region
        for i in range(len(temp_sample)):
            self.lcon_.lb = self.lcon.lb - self.lcon.A[:, keep] @ temp_sample[i][keep]
            self.lcon_.ub = self.lcon.ub - self.lcon.A[:, keep] @ temp_sample[i][keep]
            try:
                res = sample_mapping(samples0[i], self.upper[self.mask], self.lower[self.mask], lcon=self.lcon_)
                # print('Mapping w Mask: No.{}'.format(i)) #print(res['constr_violation'])
                if res['constr_violation'] >= 1:
                    raise Exception("constraint violation is too large!")
                samples[i, self.mask] = res['x']
                samples[i, self.keep] = temp_sample[i, self.keep]
            except:
                res = sample_mapping(temp_sample[i], self.upper, self.lower, lcon=self.lcon)
                # print('Mapping wo Mask: No.{}'.format(i))
                samples[i, :] = res['x']
        self.Samples = deepcopy(samples)
        return self.Samples

    def simulator(self, samples, DATE, lsSTEP):
        self.y, self.mask, self.ENTH_sim, self.PBHP_sim, self.IBHP_sim, \
        self.PPump_sim, self.IPump_sim, self.RATE_sim, self.RATEi_sim = call_simulation(
            samples, DATE, lsSTEP, Prod_bounds=self.pbound_sim, Injt_bounds=self.ibound_sim, Goals=self.Goals)
        moving_files()
        return self.y, self.ypred_mask, self.ENTH_sim, self.PBHP_sim, self.IBHP_sim, self.RATE_sim, self.RATEi_sim
