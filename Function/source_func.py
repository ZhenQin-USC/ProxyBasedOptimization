from __future__ import absolute_import, division, print_function, unicode_literals
import os
import scipy.io as sio
import numpy as np
from datetime import date, timedelta
import json
import subprocess
import tensorflow as tf
from scipy.optimize import Bounds, LinearConstraint, minimize


def datefun(year0, year1, month0=1, day0=1, month1=1, day1=1):
    DATE = []
    lsSTEP = []
    date0 = date.fromisoformat('{year}-{month:02d}-{day:02d}'.format(year=year0, month=month0, day=day0))
    date1 = date.fromisoformat('{year}-{month:02d}-{day:02d}'.format(year=year1, month=month1, day=day1))
    dT = date1 - date0
    dt = timedelta(days=7)
    nstep = int(np.ceil(dT.days / 7))
    nweek = 0
    for i in range(nstep):
        DATE.append(date0 + dt * i)
        nweek += 1
        if i >= 1:
            if DATE[i].year == DATE[i - 1].year:
                pass
            else:
                lsSTEP.append(nweek)
                nweek = 0
    if DATE[-1].year == year1:
        pass
    else:
        lsSTEP.append(nweek)
    nstage = DATE[-1].year - DATE[0].year + 1
    STEP = np.asarray(lsSTEP)
    return nstep, STEP, lsSTEP, DATE, nstage


def load_dict(path_to_dict):
    return np.load(path_to_dict, allow_pickle='TRUE').item()


def OptInitializer(year0=2021, year1=2033, month0=1, day0=1, nprod=6, ninjt=8, nstage=None, twindow=6):
    [nstep_, STEP_, lsSTEP_, DATE, nstage_] = datefun(year0, year1, month0=month0, day0=day0)
    if nstage is None or nstage <= 0:
        nstage = nstage_  # annually changed
    lsSTEP = []
    dyear = nstage_ / nstage
    t0 = 0
    index = []
    # sum up steps into nstage groups:
    for i in range(nstage):
        index.append(list(range(int(t0), int(t0 + dyear))))
        t0 += dyear
        # print(i)
        if i == nstage - 1:
            if index[-1][-1] is not nstage_ - 1:
                index[-1].append(nstage_ - 1)
        lsSTEP.append(int(sum(STEP_[index[i]])))
    nstep = sum(lsSTEP) - 1
    totaln = nstep + twindow
    # coefficients for objective function:
    coeff = tf.constant(0.13 / (3.6 * 1000), dtype='float32')
    coeffh = tf.constant(245.4050, dtype='float32')
    coeff2 = tf.constant(1 / nstep, dtype='float32')
    Pratio = tf.constant(1 / 0.4 / 1000, dtype='float32')
    Iratio = tf.constant(1 / 4.5 / 1000, dtype='float32')
    coeffs = [coeff, coeffh, coeff2, Pratio, Iratio]
    # fixed input controls:
    tvar = tf.constant(np.expand_dims(np.asarray(range(6, totaln)), axis=(0, 2)) / totaln, dtype='float32')
    itemp2 = 0.5 * tf.ones((1, nstep, 5))
    itemp3 = 0.5 * tf.ones((1, nstep, 2))
    return nstep, nstep_, nprod, ninjt, coeffs, nstage, lsSTEP, lsSTEP_, DATE, tvar, itemp2, itemp3


def Monitoring(x_hist, DATE, lsSTEP, Prod_bounds, Injt_bounds):
    Goals=['Monitoring']
    DATE_= [DATE[j-1] for j in np.concatenate(([1],np.cumsum(lsSTEP)),axis=0)]
    lsSTEP_ = [1 for j in range(len(lsSTEP))]
    fval, mask, ENTH_sim, PBHP_sim, IBHP_sim, PPump_sim, IPump_sim, RATE_sim, RATEi_sim = call_simulation(
        x_hist, DATE_, lsSTEP_, Prod_bounds, Injt_bounds, Goals=Goals)
    moving_files()
    return fval, mask, ENTH_sim, PBHP_sim, IBHP_sim, PPump_sim, IPump_sim, RATE_sim, RATEi_sim


def call_simulation(x_norm, DATE, lsSTEP, Prod_bounds, Injt_bounds, Goals):
    Main_dir = os.getcwd()
    print(Main_dir)
    Matlab_workpath = os.path.join(Main_dir, 'ReservoirModel')
    os.chdir(Matlab_workpath)
    print(os.getcwd())
    # save input data as json
    input_array = np.asarray(x_norm).tolist()
    with open('input_array.json', 'w') as f:
        json.dump(input_array, f)
    with open('Prod_bounds.json', 'w') as f:
        json.dump(Prod_bounds.tolist(), f)
    with open('Injt_bounds.json', 'w') as f:
        json.dump(Injt_bounds.tolist(), f)
    with open('Goals.json', 'w') as f:
        json.dump(Goals, f)
    DATE_json = json.dumps(DATE, sort_keys=True, default=str)
    with open('DATE.json', 'w') as f:
        json.dump(DATE_json, f)
    with open('lsSTEP.json', 'w') as f:
        json.dump(lsSTEP, f)
    # run matlab
    matlab = [r'C:\Program Files\MATLAB\R2021a\bin\matlab.exe']
    options = ['-nodesktop', '-nosplash', '-wait', '-r']
    command = ['Forward_Simulation; quit']
    command2= ['Moving_Files; quit']
    args = matlab + options + command
    # args1 = matlab + options + command2
    print('start subprocess')
    subprocess.run(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    # simulation run is finished
    # get objective function values
    fval = sio.loadmat('output_array.mat')['NPower']
    mask = sio.loadmat('output_array.mat')['mask']
    ENTH_sim = sio.loadmat('output_array.mat')['ENTH']
    PBHP_sim = sio.loadmat('output_array.mat')['PBHP']
    IBHP_sim = sio.loadmat('output_array.mat')['IBHP']
    RATE_sim = sio.loadmat('output_array.mat')['RATE']
    RATEi_sim= sio.loadmat('output_array.mat')['RATEi']
    PPump_sim= sio.loadmat('output_array.mat')['PPump']
    IPump_sim= sio.loadmat('output_array.mat')['IPump']
    # moving files
    # subprocess.run(args1, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    print('finish subprocess')
    print('matlab terminated')
    os.chdir(Main_dir)
    return fval, mask, ENTH_sim, PBHP_sim, IBHP_sim, PPump_sim, IPump_sim, RATE_sim, RATEi_sim


def moving_files():
    Main_dir = os.getcwd()
    print(Main_dir)
    Matlab_workpath = os.path.join(Main_dir, 'ReservoirModel')
    os.chdir(Matlab_workpath)
    print(os.getcwd())
    matlab = [r'C:\Program Files\MATLAB\R2021a\bin\matlab.exe']
    options = ['-nodesktop', '-nosplash', '-wait', '-r']
    command2 = ['Moving_Files; quit']
    args = matlab + options + command2
    print('start subprocess')
    subprocess.run(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    print('finish subprocess')
    print('matlab terminated')
    os.chdir(Main_dir)


def initializer(totaln, ndata, twindow=6, twindow2=104, nbsize=20, predictwindow=0, frequency=1, validsplit=0):
    return totaln, ndata, twindow, twindow2, nbsize, predictwindow, frequency, validsplit


def ConstraintSetup(rate0, nstage, pbound_proxy, ibound_proxy, pbound_sim, ibound_sim, QTOTOMASS=1850, nprod=6, ninjt=8,
                    x0_type='equal'):
    # Production well: ["21-28", "78-20", "88-19", "77A-19","21-19", "77-19"]
    # Injection well: ["44-21", "23-17", "36-17", "37-17", "85-19", "38-21", "36A-15", "2429I"]

    repeat_func = lambda x: np.repeat(x, repeats=nstage)
    pubs = repeat_func(pbound_sim[0, :])  # bounds for simulation
    plbs = repeat_func(pbound_sim[1, :])
    iubs = repeat_func(ibound_sim[0, :])
    ilbs = repeat_func(ibound_sim[1, :])
    pubp = repeat_func(pbound_proxy[0, :])  # bounds for proxy model
    plbp = repeat_func(pbound_proxy[1, :])
    iubp = repeat_func(ibound_proxy[0, :])
    ilbp = repeat_func(ibound_proxy[1, :])
    # store information into 'control'
    control = {'ProdBound': pbound_proxy,
               'InjtBound': ibound_proxy,
               'QTOTOMASS': QTOTOMASS,
               'nprod': nprod,
               'ninjt': ninjt,
               'ncontrol': nprod + ninjt,
               'QTOTMASS-Equality': 1}

    # normalize initial control based on proxy bound
    xp00 = (rate0['ProdRate'] - plbp) / (pubp - plbp)
    xi00 = (rate0['InjtRate'] - ilbp) / (iubp - ilbp)
    x00 = np.concatenate((xp00, xi00), axis=0)

    # normalize sim bounds based on proxy bound
    pubn = (pubs - plbp) / (pubp - plbp)
    plbn = (plbs - plbp) / (pubp - plbp)
    iubn = (iubs - ilbp) / (iubp - ilbp)
    ilbn = (ilbs - ilbp) / (iubp - ilbp)

    # get bounds for
    ub = np.concatenate((pubn, iubn), axis=0)
    lb = np.concatenate((plbn, ilbn), axis=0)

    # get mapped constraints and initial control
    x0, Aeq, beq, A, b, ub, lb = ConstraintFunc(x00, nstage, control, x0_type, ub, lb);
    bounds = Bounds(lb, ub, keep_feasible=True)  # bound constraints
    Alb = np.zeros_like(b)
    c1 = LinearConstraint(A, Alb, b, keep_feasible=True)  # inequality constraints
    c2 = LinearConstraint(Aeq, beq, beq, keep_feasible=True)  # equality constraints
    if b.shape[0]:
        lcon = [c1, c2]
    else:
        lcon = c2
    return x0, bounds, lcon, x00, Aeq, beq, A, b, ub, lb


def ConstraintFunc(x00, nstage, control, x0_type=None, ub=None, lb=None):
    def objfun(x, x00=x00):
        return 0.5 * (x - x00) @ (x - x00)

    def gradfun(x, x00=x00):
        return x - x00

    repeat_func = lambda x: np.repeat(x, repeats=nstage)
    ncontrol = control['ncontrol']
    ndim = ncontrol * nstage

    if ub is None:
        ub = np.ones((ndim, 1)) - 1e-4
    if lb is None:
        lb = np.zeros((ndim, 1)) + 1e-4
    if len(x00) is 0:
        x00 = 0.5
    if x0_type is not None:
        x0_type = "equal"
    if np.isscalar(x00):
        if x0_type == "equal":
            x00 = x00 * np.ones((ndim, 1))
        elif x0_type == "random":
            x00 = x00 * np.random.rand(ndim, 1)
    else:
        if x00.shape[0] == ndim:
            x00 = x00
        else:
            raise TypeError("Dimension of x00 has to be ndim-by-1")

    # calculate constraints
    Dprod = repeat_func(-np.diff(control['ProdBound'], axis=0))
    Dinjt = repeat_func(-np.diff(control['InjtBound'], axis=0))
    LBProd = repeat_func(control['ProdBound'][1, :])
    LBInjt = repeat_func(control['InjtBound'][1, :])
    D = np.diag(np.concatenate((Dprod, Dinjt), axis=0))
    LB = np.concatenate((LBProd, LBInjt), axis=0)
    #
    A = []
    b = []
    A1_prod = np.tile(np.eye(nstage), (1, control['nprod']))
    A1_injt = -1 * np.tile(np.eye(nstage), (1, control['ninjt']))
    A2_injt = np.tile(np.zeros((nstage, nstage)), (1, control['ninjt']))
    A1_ipt = np.tile(np.zeros((nstage, nstage)), (1, 2 * control['ninjt']))
    A2_ipt = np.tile(np.zeros((nstage, nstage)), (1, 2 * control['ninjt']))

    if control['ncontrol'] is (control['nprod'] + 3 * control['ninjt']):
        A1_ = np.concatenate((A1_prod, A1_injt, A1_ipt), axis=1)
        A2_ = np.concatenate((A1_prod, A2_injt, A2_ipt), axis=1)
    elif control['ncontrol'] is (control['nprod'] + control['ninjt']):
        A1_ = np.concatenate((A1_prod, A1_injt), axis=1)
        A2_ = np.concatenate((A1_prod, A2_injt), axis=1)

    b1_ = np.zeros((nstage, 1))  # mass balance constraint
    b2_ = control['QTOTOMASS'] * np.ones((nstage, 1))  # total rate constraint
    Aeq_ = np.concatenate((A1_, A2_), axis=0)
    beq_ = np.concatenate((b1_, b2_), axis=0)[:, 0]
    Aeq = Aeq_ @ D
    beq = beq_ - Aeq_ @ LB

    # Set total rate constraint as equality [1] or Inequality [0]?
    if control['QTOTMASS-Equality'] is 1:
        print('Total rate is equality constraint.')
        lcon = LinearConstraint(Aeq, beq, beq, keep_feasible=True)  # equality constraints
    else:
        print('Total rate is inequality constraint.')
        A = Aeq[-nstage:, :]
        b = beq[-nstage:]
        Aeq = Aeq[:nstage, :]
        beq = beq[:nstage, ]
        Alb = np.zeros_like(b)
        c1 = LinearConstraint(A, Alb, b, keep_feasible=True)  # inequality constraints
        c2 = LinearConstraint(Aeq, beq, beq, keep_feasible=True)  # equality constraints
        lcon = [c1, c2]
    bounds = Bounds(lb, ub, keep_feasible=True)  # bound constraints
    # print('mapping')
    # map x00 into feasible region
    res = minimize(objfun, x00, method='trust-constr', jac=gradfun, hess=None,
                   bounds=bounds, constraints=lcon, tol=None,
                   options={'xtol': 1e-12, 'gtol': 1e-12,
                            'barrier_tol': 1e-08, 'sparse_jacobian': None,
                            'maxiter': 1e3, 'verbose': 0, 'finite_diff_rel_step': None,
                            'initial_constr_penalty': 1.0, 'initial_tr_radius': 1.0,
                            'initial_barrier_parameter': 0.1, 'initial_barrier_tolerance': 0.1,
                            'factorization_method': None, 'disp': False})
    x0 = res['x']
    return np.asarray(x0), np.asarray(Aeq), np.asarray(beq), np.asarray(A), np.asarray(b), np.asarray(ub), np.asarray(lb)


