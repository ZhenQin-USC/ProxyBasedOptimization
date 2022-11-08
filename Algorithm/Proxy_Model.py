import sys
sys.path.append(r'D:\Users\qinzh\Google Drive USC\MATLAB Local\Proxy Opt')
from Function.source_func import OptInitializer, ConstraintSetup, Monitoring
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
# from pyDOE2 import *
#import os

def ConvertToScaler(ProxyBoundEnth):
    # Prod["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"];
    # Fault:  3        m        2        2         1        2
    jp1 = [4]
    jp2 = [2, 3, 5]
    jp3 = [0]
    jpm = [1]
    # Inj:["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
    # Fault:  3       2       2       2       2       3       -        2
    ji2 = [1, 2, 3, 4, 7]
    ji3 = [0, 5]
    ji_ = [6]
    s1 = np.array([ProxyBoundEnth['Prod_bounds'][1, jp1][0], ProxyBoundEnth['Prod_bounds'][0, jp1][0],
                   ProxyBoundEnth['Enth_bounds'][0, jp1][0], ProxyBoundEnth['Enth_bounds'][0, jp1][0]])
    s2 = np.array([ProxyBoundEnth['Prod_bounds'][1, jp2][0], ProxyBoundEnth['Prod_bounds'][0, jp2][0],
                   ProxyBoundEnth['Injt_bounds'][1, ji2][0], ProxyBoundEnth['Injt_bounds'][0, ji2][0],
                   ProxyBoundEnth['Enth_bounds'][1, jp2][0], ProxyBoundEnth['Enth_bounds'][0, jp2][0]])
    sm = np.array([ProxyBoundEnth['Prod_bounds'][1, jpm][0], ProxyBoundEnth['Prod_bounds'][0, jpm][0],
                   ProxyBoundEnth['Enth_bounds'][1, jpm][0], ProxyBoundEnth['Enth_bounds'][0, jpm][0]])
    s3 = np.array([ProxyBoundEnth['Prod_bounds'][1, jp3][0], ProxyBoundEnth['Prod_bounds'][0, jp3][0],
                   ProxyBoundEnth['Injt_bounds'][1, ji3][0], ProxyBoundEnth['Injt_bounds'][0, ji3][0],
                   ProxyBoundEnth['Enth_bounds'][1, jp3][0], ProxyBoundEnth['Enth_bounds'][0, jp3][0]])
    s_ = np.array([ProxyBoundEnth['Injt_bounds'][1, ji_][0], ProxyBoundEnth['Injt_bounds'][0, ji_][0]])
    scaler = [tf.constant(s1, dtype='float32'),
              tf.constant(s2, dtype='float32'),
              tf.constant(sm, dtype='float32'),
              tf.constant(s3, dtype='float32'),
              tf.constant(s_, dtype='float32')]
    return scaler


# without bhp-related pump cost
class proxy_model:
    def __init__(self, models, history, scaler, coeffs, lsSTEP, it2n, it3n, tvar, nprod, nstage):
        self.models = models
        self.history = history
        self.coeffs = coeffs
        self.lsSTEP = lsSTEP
        self.itemp2 = it2n
        self.itemp3 = it3n
        self.tvar = tvar
        self.s = scaler
        self.nprod = nprod
        self.nstage = nstage
        self.fval = None
        self.GPower = None
        self.PPump = None
        self.IPump = None
        self.enth1 = None
        self.enth2a = None
        self.enth2b = None
        self.enth2c = None
        self.enthm = None
        self.enth3 = None
        self.history_x = []
        self.history_f = []
        self.historyFval = {'GPower': [], 'PPump': [], 'IPump': []}
        self.historyEnth = {'enth1': [], 'enth2a': [], 'enth2b': [], 'enth2c': [], 'enthm': [], 'enth3': []}

    def ReturnResults(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        fval, NPower, GPower, PPump, IPump, enth1, enth2a, enth2b, enth2c, enthm, enth3 = self.computeFval(control)
        return fval, NPower, GPower, PPump, IPump, enth1, enth2a, enth2b, enth2c, enthm, enth3

    def objfun(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        self.fval, self.NPower, self.GPower, self.PPump, self.IPump, self.enth1, self.enth2a, self.enth2b, self.enth2c, self.enthm, self.enth3 = self.computeFval(
            control)
        return self.fval.numpy()

    def gradfun(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        grad = self.computegrad(control).numpy()
        return grad

    def savedata(self, x, res):
        historyGPower = self.historyFval['GPower']
        historyPPump = self.historyFval['PPump']
        historyIPump = self.historyFval['IPump']
        historyEnth1 = self.historyEnth['enth1']
        historyEnth2a = self.historyEnth['enth2a']
        historyEnth2b = self.historyEnth['enth2b']
        historyEnth2c = self.historyEnth['enth2c']
        historyEnthm = self.historyEnth['enthm']
        historyEnth3 = self.historyEnth['enth3']

        historyGPower.append(self.GPower.numpy())
        historyPPump.append(self.PPump.numpy())
        historyIPump.append(self.IPump.numpy())
        historyEnth1.append(self.enth1.numpy())
        historyEnth2a.append(self.enth2a.numpy())
        historyEnth2b.append(self.enth2b.numpy())
        historyEnth2c.append(self.enth2c.numpy())
        historyEnthm.append(self.enthm.numpy())
        historyEnth3.append(self.enth3.numpy())

        self.history_x.append(x)
        self.history_f.append(self.fval.numpy())
        self.historyFval = {'GPower': historyGPower,
                            'PPump': historyPPump,
                            'IPump': historyIPump}
        self.historyEnth = {'enth1': historyEnth1,
                            'enth2a': historyEnth2a,
                            'enth2b': historyEnth2b,
                            'enth2c': historyEnth2c,
                            'enthm': historyEnthm,
                            'enth3': historyEnth3}

    def Optimizer(self, x0, bounds, lcon, constr_penalty=1.0, tr_radius=1.0, barrier_para=0.1, barrier_tol=0.1):
        fun = self.objfun
        gfun = self.gradfun
        res = minimize(fun, x0, method='trust-constr', jac=gfun, hess=None,
                       bounds=bounds, constraints=lcon, tol=None,
                       callback=self.savedata,
                       options={'xtol': 1e-06, 'gtol': 1e-06,
                                'barrier_tol': 1e-08, 'sparse_jacobian': None,
                                'maxiter': 1e3, 'verbose': 2, 'finite_diff_rel_step': None,
                                'initial_constr_penalty': constr_penalty,
                                'initial_tr_radius': tr_radius,
                                'initial_barrier_parameter': barrier_para,
                                'initial_barrier_tolerance': barrier_tol,
                                'factorization_method': None, 'disp': True})
        return res, self.history_x, self.history_f, self.historyFval, self.historyEnth

    @tf.function
    def computegrad(self, control):
        output = self.Func_Val(control)
        grad = tf.reshape(tf.gradients(output, control), [-1])
        return grad

    @tf.function
    def Func_Val(self, control):
        # models  = self.models
        output = self.computeFval(control)[0]
        return output

    @tf.function
    def computeFval(self, control):
        coeffs = self.coeffs
        coeff = coeffs[0]
        coeffh = coeffs[1]
        coeff2 = coeffs[2]
        Pratio = coeffs[3]
        Iratio = coeffs[4]

        [enth1, enth2a, enth2b, enth2c, enthm, enth3,
         p1, p2a, p2b, p2c, pm, p3,
         i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.computeEnth(control)
        prate = self.sum_array([p1, p2a, p2b, p2c, pm, p3])
        irate = self.sum_array([i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_])
        PPump = self.reduce_sum(prate, coeff2 * Pratio)
        IPump = self.reduce_sum(irate, coeff2 * Iratio)
        TPump = PPump + IPump

        ENTH1 = tf.multiply(p1, enth1)
        ENTH2a = tf.multiply(p2a, enth2a)
        ENTH2b = tf.multiply(p2b, enth2b)
        ENTH2c = tf.multiply(p2c, enth2c)
        ENTHm = tf.multiply(pm, enthm)
        ENTH3 = tf.multiply(p3, enth3)
        ENTH = self.sum_array([ENTH1, ENTH2a, ENTH2b, ENTH2c, ENTHm, ENTH3])
        ENTMWe = self.reduce_sum(ENTH, coeff2 * coeff)
        iENTH = coeffh * irate
        iENTMWe = self.reduce_sum(iENTH, coeff2 * coeff)
        GPower = ENTMWe - iENTMWe
        NPower = GPower - TPump
        fval = - NPower
        return fval, NPower, GPower, PPump, IPump, enth1, enth2a, enth2b, enth2c, enthm, enth3

    @tf.function
    def computeEnth(self, control):
        models = self.models
        # history = self.history

        f1 = models[0]
        f2 = models[1]
        fm = models[2]
        f3 = models[3]
        [h1, h2, hm, h3] = self.splitHist()
        [c1, c2, cm, c3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.Expanders(control)
        state1 = np.zeros((1, 1))  # 1
        state2 = np.zeros((1, 3))  # 3
        is_training = False * np.ones(1)
        # Compute normalized enthalpy
        enth1n = f1([h1, c1, state1, is_training])[0]
        enth2n = f2([h2, c2, state2, is_training])[0]
        enthmn = fm([hm, cm, state1, is_training])[0]
        enth3n = f3([h3, c3, state1, is_training])[0]
        [enth1, enth2a, enth2b, enth2c, enthm, enth3] = self.backnormENTH(enth1n, enth2n, enthmn, enth3n)
        return enth1, enth2a, enth2b, enth2c, enthm, enth3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def Expanders(self, controln):
        # it2n = self.itemp2
        # it3n = self.itemp3
        tvar = self.tvar
        [pcontroln, icontroln] = self.spliter_PI(controln)
        # production rate
        [p1n, p2an, p2bn, p2cn, pmn, p3n] = self.ExpandProd(pcontroln)
        [p1, p2a, p2b, p2c, pm, p3] = self.backnormPC(p1n, p2an, p2bn, p2cn, pmn, p3n)
        # injection rate
        [i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n] = self.ExpandInjt(icontroln)
        [i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.backnormIC(i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n)
        # concatenate controls for fault zone 2 and 3

        # with time variable
        # c1 = layers.Concatenate(axis=-1)([tvar, p1n])  # 1 + 1
        # # c2 = layers.Concatenate(axis=-1)([tvar,p2an,p2bn,p2cn,i2an,i2bn,i2cn,i2dn,i2en,it2n])  # 13 + 1  with temperature
        # c2 = layers.Concatenate(axis=-1)([tvar, p2an, p2bn, p2cn, i2an, i2bn, i2cn, i2dn, i2en])  # 13 + 1  without temperature
        # cm = layers.Concatenate(axis=-1)([tvar, pmn])  # 1 + 1
        # # c3 = layers.Concatenate(axis=-1)([tvar,p3n,i3an,i3bn,it3n])  # 5 + 1 with temperature
        # c3 = layers.Concatenate(axis=-1)([tvar, p3n, i3an, i3bn])  # 5 + 1 without temperature

        # without time variable
        c1 = p1n
        c2 = layers.Concatenate(axis=-1)([p2an, p2bn, p2cn, i2an, i2bn, i2cn, i2dn, i2en])
        cm = pmn
        c3 = layers.Concatenate(axis=-1)([p3n, i3an, i3bn])
        return c1, c2, cm, c3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def backnormPC(self, p1n, p2an, p2bn, p2cn, pmn, p3n):  # Map normalized rate and enthalpy to original space
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        p1 = self.backnorm(p1n, s1[0], s1[1])
        p2a = self.backnorm(p2an, s2[0], s2[1])
        p2b = self.backnorm(p2bn, s2[0], s2[1])
        p2c = self.backnorm(p2cn, s2[0], s2[1])
        pm = self.backnorm(pmn, sm[0], sm[1])
        p3 = self.backnorm(p3n, s3[0], s3[1])
        return p1, p2a, p2b, p2c, pm, p3

    @tf.function
    def backnormIC(self, i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n):
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        i2a = self.backnorm(i2an, s2[2], s2[3])
        i2b = self.backnorm(i2bn, s2[2], s2[3])
        i2c = self.backnorm(i2cn, s2[2], s2[3])
        i2d = self.backnorm(i2dn, s2[2], s2[3])
        i2e = self.backnorm(i2en, s2[2], s2[3])
        i3a = self.backnorm(i3an, s3[2], s3[3])
        i3b = self.backnorm(i3bn, s3[2], s3[3])
        i_ = self.backnorm(i_n, s_[0], s_[1])
        return i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def backnormENTH(self, enth1n, enth2n, enthmn, enth3n):
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        enth1 = self.backnorm(enth1n, s1[-2], s1[-1])
        enth2a = self.backnorm(enth2n[:, :, 0:1], s2[-2], s2[-1])
        enth2b = self.backnorm(enth2n[:, :, 1:2], s2[-2], s2[-1])
        enth2c = self.backnorm(enth2n[:, :, 2:3], s2[-2], s2[-1])
        enthm = self.backnorm(enthmn, sm[-2], sm[-1])
        enth3 = self.backnorm(enth3n, s3[-2], s3[-1])
        return enth1, enth2a, enth2b, enth2c, enthm, enth3

    @tf.function
    def spliter_PI(self, control):  # split production wells and injection wells
        nprod = self.nprod
        nstage = self.nstage
        K = nprod * nstage
        pcontrol = layers.Lambda(lambda x: x[:, :K])(control)
        icontrol = layers.Lambda(lambda x: x[:, K:])(control)
        return pcontrol, icontrol

    @tf.function
    def ExpandProd(self, pcontrol):  # extend production control
        [p1_, p2a_, p2b_, p2c_, pm_, p3_] = self.splitProd(pcontrol)
        p1 = self.repmatv(p1_)
        p2a = self.repmatv(p2a_)
        p2b = self.repmatv(p2b_)
        p2c = self.repmatv(p2c_)
        pm = self.repmatv(pm_)
        p3 = self.repmatv(p3_)
        return p1, p2a, p2b, p2c, pm, p3

    @tf.function
    def ExpandInjt(self, icontrol):  # extend injection control
        [i2a_, i2b_, i2c_, i2d_, i2e_, i3a_, i3b_, i__] = self.splitInjt(icontrol)
        i2a = self.repmatv(i2a_)
        i2b = self.repmatv(i2b_)
        i2c = self.repmatv(i2c_)
        i2d = self.repmatv(i2d_)
        i2e = self.repmatv(i2e_)
        i3a = self.repmatv(i3a_)
        i3b = self.repmatv(i3b_)
        i_ = self.repmatv(i__)
        return i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def splitProd(self, pcontrol):  # split production wells
        nstage = self.nstage
        p3_ = layers.Lambda(lambda x: x[:, 0 * nstage: 1 * nstage])(pcontrol)
        pm_ = layers.Lambda(lambda x: x[:, 1 * nstage: 2 * nstage])(pcontrol)
        p2a_ = layers.Lambda(lambda x: x[:, 2 * nstage: 3 * nstage])(pcontrol)
        p2b_ = layers.Lambda(lambda x: x[:, 3 * nstage: 4 * nstage])(pcontrol)
        p1_ = layers.Lambda(lambda x: x[:, 4 * nstage: 5 * nstage])(pcontrol)
        p2c_ = layers.Lambda(lambda x: x[:, 5 * nstage: 6 * nstage])(pcontrol)
        return p1_, p2a_, p2b_, p2c_, pm_, p3_

    @tf.function
    def splitInjt(self, icontrol):
        nstage = self.nstage
        i3a_ = layers.Lambda(lambda x: x[:, 0 * nstage: 1 * nstage])(icontrol)  # 44-21
        i2a_ = layers.Lambda(lambda x: x[:, 1 * nstage: 2 * nstage])(icontrol)  # 23-17
        i2b_ = layers.Lambda(lambda x: x[:, 2 * nstage: 3 * nstage])(icontrol)  # 36-17
        i2c_ = layers.Lambda(lambda x: x[:, 3 * nstage: 4 * nstage])(icontrol)  # 37-17
        i2d_ = layers.Lambda(lambda x: x[:, 4 * nstage: 5 * nstage])(icontrol)  # 85-19
        i3b_ = layers.Lambda(lambda x: x[:, 5 * nstage: 6 * nstage])(icontrol)  # 38-21
        i__ = layers.Lambda(lambda x: x[:, 6 * nstage: 7 * nstage])(icontrol)  # 36A-15
        i2e_ = layers.Lambda(lambda x: x[:, 7 * nstage: 8 * nstage])(icontrol)  # 2429I
        return i2a_, i2b_, i2c_, i2d_, i2e_, i3a_, i3b_, i__

    @tf.function
    def splitHist(self):  # split history and control:
        history = self.history
        return history[0], history[1], history[2], history[3]

    @tf.function
    def splitScaler(self):
        s = self.s
        return s[0], s[1], s[2], s[3], s[4]

    @tf.function
    def repmatv(self, x):  # vertically repeat control based on nweeks per year
        STEP = self.lsSTEP
        return tf.repeat(x, repeats=STEP, axis=1)

    @tf.function
    def sum_array(self, x):
        return layers.Add()(x)

    @tf.function
    def backnorm(self, x, xmin, xmax):
        return tf.add(tf.multiply(x, (xmax - xmin)), xmin)

    @tf.function
    def reduce_sum(self, x, a=1):
        return a * tf.reduce_sum(x, [0, 1, 2])


# with bhp-related pump cost, four bhp models: fz12, fz12i, fz3m, fz3mi
class proxy_model0:
    def __init__(self, models_enth, models_bhp, history_enth, history_bhp, ProxyBoundEnth, ProxyBoundBHP):
        # defaults: basic setups
        nstep, nstep_, nprod, ninjt, coeffs, nstage, lsSTEP, lsSTEP_, DATE, _, itemp2, itemp3 = OptInitializer(
            year0=2021, year1=2033, month0=5, day0=21)
        self.nprod = nprod
        self.ninjt = ninjt
        self.coeffs = coeffs
        self.nstage = nstage
        self.lsSTEP = lsSTEP
        self.DATE = DATE
        self.itemp2 = itemp2
        self.itemp3 = itemp3

        # initialization
        scaler = ConvertToScaler(ProxyBoundEnth)
        self.s = scaler
        self.models_enth = models_enth
        self.models_bhp = models_bhp
        self.history_enth = history_enth
        self.history_bhp = history_bhp
        self.fval = None
        self.GPower = None
        self.PPump = None
        self.IPump = None
        self.enth1 = None
        self.enth2a = None
        self.enth2b = None
        self.enth2c = None
        self.enthm = None
        self.enth3 = None
        self.history_x = []
        self.history_f = []
        self.historyFval = {'GPower': [], 'PPump': [], 'IPump': []}
        self.historyEnth = {'enth1': [], 'enth2a': [], 'enth2b': [], 'enth2c': [], 'enthm': [], 'enth3': []}

        self.prate_proxy_enth= tf.cast(ProxyBoundEnth['Prod_bounds'], dtype='float32')
        self.irate_proxy_enth= tf.cast(ProxyBoundEnth['Injt_bounds'], dtype='float32')
        self.prate_proxy_bhp = tf.cast(ProxyBoundBHP['Prod_bounds'], dtype='float32')
        self.irate_proxy_bhp = tf.cast(ProxyBoundBHP['Injt_bounds'], dtype='float32')
        self.ppred_proxy_bhp = tf.cast(ProxyBoundBHP['PBHP_bounds'], dtype='float32')
        self.ipred_proxy_bhp = tf.cast(ProxyBoundBHP['IBHP_bounds'], dtype='float32')
        self.lower_enth= tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_enth[1,:], self.irate_proxy_enth[1,:]]), repeats=nstage)[None,:,None]
        self.upper_enth= tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_enth[0,:], self.irate_proxy_enth[0,:]]), repeats=nstage)[None,:,None]
        self.lower_bhp = tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_bhp[1,:], self.irate_proxy_bhp[1,:]]), repeats=nstage)[None,:,None]
        self.upper_bhp = tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_bhp[0,:], self.irate_proxy_bhp[0,:]]), repeats=nstage)[None,:,None]

        # coefficients for pump power
        self.rho_injt = 1000
        self.rho_prod = 900
        self.pump_eff = 0.75
        psi = 6.89476
        kPa = 1
        gravity = 9.81
        TVD_prod = tf.cast(np.array([2405, 2055, 2637, 2685, 2713, 2685]), dtype='float32')
        TVD_injt = tf.cast(np.array([2.3 * 2481, 1.6 * 2637, 1.35 * 2715, 3312, 1.4 * 2321, 3044, 1.1 * 2737, 2103]),
                           dtype='float32')
        pgrad_prod = 22.6206 * 0.052 * self.rho_prod / 119.83  # kPa/m
        pgrad_injt = 22.6206 * 0.052 * self.rho_injt / 119.83  # kPa/m
        WH_inlet = 150 * psi / kPa
        WH_outlet0 = 150 * psi / kPa
        WH_outlet = 200 * psi / kPa

        self.dP_injt0 = WH_outlet - WH_outlet0
        self.BackPres_prod = TVD_prod * pgrad_prod + WH_inlet  # kPa
        self.BackPres_injt = TVD_injt * pgrad_injt + WH_outlet  # kPa

    def monitoring(self, history_x, interval = 1):
        N = history_x.shape[0] # total number of iteration
        x_filter = list(range(0, N, interval))
        if x_filter[-1] != N - 1:
            x_filter.append(N - 1)
        x_hist = history_x[x_filter, :]
        print(x_hist.shape)
        DATE = self.DATE
        lsSTEP = self.lsSTEP
        Prod_bounds = self.prate_proxy_enth.numpy()
        Injt_bounds = self.irate_proxy_enth.numpy()
        fval_sim, mask, ENTH_sim, PBHP_sim, IBHP_sim, PPump_sim, IPump_sim, RATE_sim, RATEi_sim = Monitoring(
            x_hist, DATE, lsSTEP, Prod_bounds, Injt_bounds)
        return fval_sim, mask, ENTH_sim, PBHP_sim, IBHP_sim, PPump_sim, IPump_sim, RATE_sim, RATEi_sim, x_filter

    def ReturnResult(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        PumpPower, PPump, IPump, IPump0 = self.computePumpPW(control)
        fval, NPower, GPower, _, _, enth1, enth2a, enth2b, enth2c, enthm, enth3 = self.computeFval(control)
        pbhp, ibhp, prate, irate = self.computeRateBHP(control)
        return fval, NPower, GPower, PumpPower, PPump, IPump, IPump0, \
               enth1, enth2a, enth2b, enth2c, enthm, enth3, pbhp, ibhp, prate, irate

    def Init_Optimizer(self, rate0, pbound_sim=None, ibound_sim=None, x0_type='equal'):
        if pbound_sim is None:
            pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                                   [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
        if ibound_sim is None:
            ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                                   [  0,   0,   0,   0,   0,   0,   0,   0]])

        x0, bounds, lcon, x00, Aeq, beq, A, b, ub, lb = ConstraintSetup(
            rate0, self.nstage, self.prate_proxy_enth, self.irate_proxy_enth, pbound_sim, ibound_sim, QTOTOMASS=1850,
            nprod=self.nprod, ninjt=self.ninjt, x0_type=x0_type)
        return x0, bounds, lcon, x00, Aeq, beq, A, b, ub, lb

    def objfun(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        self.fval, self.NPower, self.GPower, self.PPump, self.IPump, self.enth1, self.enth2a, self.enth2b, self.enth2c, self.enthm, self.enth3 = self.computeFval(
            control)
        return self.fval.numpy()

    def gradfun(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        grad = self.computegrad(control).numpy()
        return grad

    def savedata(self, x, res):
        historyGPower = self.historyFval['GPower']
        historyPPump = self.historyFval['PPump']
        historyIPump = self.historyFval['IPump']
        historyEnth1 = self.historyEnth['enth1']
        historyEnth2a = self.historyEnth['enth2a']
        historyEnth2b = self.historyEnth['enth2b']
        historyEnth2c = self.historyEnth['enth2c']
        historyEnthm = self.historyEnth['enthm']
        historyEnth3 = self.historyEnth['enth3']

        historyGPower.append(self.GPower.numpy())
        historyPPump.append(self.PPump.numpy())
        historyIPump.append(self.IPump.numpy())
        historyEnth1.append(self.enth1.numpy())
        historyEnth2a.append(self.enth2a.numpy())
        historyEnth2b.append(self.enth2b.numpy())
        historyEnth2c.append(self.enth2c.numpy())
        historyEnthm.append(self.enthm.numpy())
        historyEnth3.append(self.enth3.numpy())

        self.history_x.append(x)
        self.history_f.append(self.fval.numpy())
        self.historyFval = {'GPower': historyGPower,
                            'PPump': historyPPump,
                            'IPump': historyIPump}
        self.historyEnth = {'enth1': historyEnth1,
                            'enth2a': historyEnth2a,
                            'enth2b': historyEnth2b,
                            'enth2c': historyEnth2c,
                            'enthm': historyEnthm,
                            'enth3': historyEnth3}

    def Optimizer(self, x0, bounds, lcon, constr_penalty=1.0, tr_radius=1.0, barrier_para=0.1, barrier_tol=0.1):
        fun = self.objfun
        gfun = self.gradfun
        res = minimize(fun, x0, method='trust-constr', jac=gfun, hess=None,
                       bounds=bounds, constraints=lcon, tol=None,
                       callback=self.savedata,
                       options={'xtol': 1e-06, 'gtol': 1e-06,
                                'barrier_tol': 1e-08, 'sparse_jacobian': None,
                                'maxiter': 1e3, 'verbose': 2, 'finite_diff_rel_step': None,
                                'initial_constr_penalty': constr_penalty,
                                'initial_tr_radius': tr_radius,
                                'initial_barrier_parameter': barrier_para,
                                'initial_barrier_tolerance': barrier_tol,
                                'factorization_method': None, 'disp': True})
        return res, self.history_x, self.history_f, self.historyFval, self.historyEnth

    @tf.function
    def computegrad(self, control):
        output = self.Func_Val(control)
        grad = tf.reshape(tf.gradients(output, control), [-1])
        return grad

    @tf.function
    def Func_Val(self, control):
        # models  = self.models
        output = self.computeFval(control)[0]
        return output

    @tf.function
    def computeFval(self, control):
        coeffs = self.coeffs
        coeff = coeffs[0]
        coeffh = coeffs[1]
        coeff2 = coeffs[2]
        Pratio = coeffs[3]
        Iratio = coeffs[4]

        [enth1, enth2a, enth2b, enth2c, enthm, enth3,
         p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.computeEnth(control)
        prate = self.sum_array([p1, p2a, p2b, p2c, pm, p3])
        irate = self.sum_array([i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_])
        PumpPower, PPump, IPump, IPump0 = self.computePumpPW(control)

        ENTH1 = tf.multiply(p1, enth1)
        ENTH2a = tf.multiply(p2a, enth2a)
        ENTH2b = tf.multiply(p2b, enth2b)
        ENTH2c = tf.multiply(p2c, enth2c)
        ENTHm = tf.multiply(pm, enthm)
        ENTH3 = tf.multiply(p3, enth3)
        ENTH = self.sum_array([ENTH1, ENTH2a, ENTH2b, ENTH2c, ENTHm, ENTH3])
        ENTMWe = self.reduce_sum(ENTH, coeff2 * coeff)
        iENTH = coeffh * irate
        iENTMWe = self.reduce_sum(iENTH, coeff2 * coeff)
        GPower = ENTMWe - iENTMWe
        NPower = GPower - PumpPower
        fval = - NPower
        return fval, NPower, GPower, PPump, IPump, enth1, enth2a, enth2b, enth2c, enthm, enth3

    @tf.function
    def computeEnth(self, control):
        models_enth = self.models_enth
        # history = self.history
        f1 = models_enth[0]
        f2 = models_enth[1]
        fm = models_enth[2]
        f3 = models_enth[3]
        [h1, h2, hm, h3] = self.splitHist()
        [c1, c2, cm, c3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.Expanders(control)
        state1 = np.zeros((1, 1))  # 1
        state2 = np.zeros((1, 3))  # 3
        is_training = False * np.ones(1)
        # Compute normalized enthalpy
        enth1n = f1([h1, c1, state1, is_training])[0]
        enth2n = f2([h2, c2, state2, is_training])[0]
        enthmn = fm([hm, cm, state1, is_training])[0]
        enth3n = f3([h3, c3, state1, is_training])[0]
        [enth1, enth2a, enth2b, enth2c, enthm, enth3] = self.backnormENTH(enth1n, enth2n, enthmn, enth3n)
        return enth1, enth2a, enth2b, enth2c, enthm, enth3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def computePumpPW(self, control):
        # get bhp and rate
        pbhp, ibhp, prate, irate = self.computeRateBHP(control)

        # calculate dp
        dp_prod = tf.keras.activations.relu(self.BackPres_prod - pbhp)
        dp_injt = tf.keras.activations.relu(ibhp - self.BackPres_injt)
        dp_injt0 = self.dP_injt0

        # calculate hydraulic power for each step and well
        Epump_prod = self.HydraulicPower(dp_prod, prate, self.rho_prod)  # MWe
        Epump_injt = self.HydraulicPower(dp_injt, irate, self.rho_injt)
        Epump_injt0 = self.HydraulicPower(dp_injt0, irate, self.rho_injt)

        # calculate average pump power
        IPump0 = tf.reduce_mean(tf.reduce_sum(Epump_injt0, axis=1), axis=0) / self.pump_eff
        PPump = tf.reduce_mean(tf.reduce_sum(Epump_prod, axis=1), axis=0) / self.pump_eff
        IPump = tf.reduce_mean(tf.reduce_sum(Epump_injt, axis=1), axis=0) / self.pump_eff + IPump0
        PumpPower = PPump + IPump
        return PumpPower, PPump, IPump, IPump0

    @tf.function
    def HydraulicPower(self, dp, rate, rho):
        return dp*(rate/rho)/(3.6e3)

    @tf.function
    def computeRateBHP(self, control):
        # define model
        f12p = self.models_bhp[0]
        f3mp = self.models_bhp[1]
        f12i= self.models_bhp[2]
        f3mi= self.models_bhp[3]
        # split control and history for two models
        h12p, h3mp, h12i, h3mi = self.SplitHistBHP()
        c12, c3m, praten, iraten, rate0_sim = self.SplitRateBHP(control)

        # backnorm rate
        rate0 = self.repmatv(tf.transpose(tf.reshape(rate0_sim, [1, -1, self.nstage]), perm=(0, 2, 1)))
        prate = rate0[0, :, :self.nprod]
        irate = rate0[0, :, self.nprod:]

        # return predicted normalized bhp from two models
        state12p = np.zeros((1, 4))  # section 19
        state12i = np.zeros((1, 5))  # five injectors
        state3mp = np.zeros((1, 2))  # 78-20 and 21-28
        state3mi = np.zeros((1, 3))  # 44-21, 38-21, and 36A-15
        is_training = False * np.ones(1)
        bhp12pn = f12p([h12p, c12, state12p, is_training])[0]
        bhp12in = f12i([h12i, c12, state12i, is_training])[0]
        bhp3mpn = f3mp([h3mp, c3m, state3mp, is_training])[0]
        bhp3min = f3mi([h3mi, c3m, state3mi, is_training])[0]
        bhp12n = layers.Concatenate(axis=-1)([bhp12pn, bhp12in])
        bhp3mn = layers.Concatenate(axis=-1)([bhp3mpn, bhp3min])

        # backnorm bhp
        pbhp, ibhp = self.backnormBHP(bhp12n, bhp3mn)
        return pbhp, ibhp, prate, irate

    @tf.function
    def SplitRateBHP(self, control):
        rate0_sim = control*(self.upper_enth-self.lower_enth)+self.lower_enth
        x0_bhp00 = (rate0_sim-self.lower_bhp)/(self.upper_bhp-self.lower_bhp)
        x0_bhp01 = tf.reshape(x0_bhp00,(1,-1, self.nstage))
        x0_bhp11 = tf.transpose(x0_bhp01, perm=(0,2,1))
        x0_bhp = self.repmatv(x0_bhp11)
        praten = x0_bhp[:,:,:self.nprod]
        iraten = x0_bhp[:,:,self.nprod:]
        p12 = layers.Concatenate(axis=-1)([praten[:,:,2:3],praten[:,:,3:4],praten[:,:,4:5],praten[:,:,5:6]]) # jp12 = [2, 3, 4, 5]
        i12 = layers.Concatenate(axis=-1)([iraten[:,:,1:2],iraten[:,:,2:3],iraten[:,:,3:4],iraten[:,:,4:5],iraten[:,:,7:8]]) # ji12 = [1, 2, 3, 4, 7]
        p3m = layers.Concatenate(axis=-1)([praten[:,:,0:1],praten[:,:,1:2]]) # jp3m = [0, 1]
        i3m = layers.Concatenate(axis=-1)([iraten[:,:,0:1],iraten[:,:,5:6],iraten[:,:,6:7]]) # ji3m = [0, 5, 6]
        c12 = layers.Concatenate(axis=-1)([p12, i12])
        c3m = layers.Concatenate(axis=-1)([p3m, i3m])
        return c12, c3m, praten, iraten, rate0_sim

    @tf.function
    def SplitHistBHP(self):
        return self.history_bhp[0], self.history_bhp[1], self.history_bhp[2], self.history_bhp[3]

    @tf.function
    def backnormBHP(self, bhp12n, bhp3mn):
        # Fault zones:       3m       3m       12       12        12       12
        # Prod_name  =  ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"];
        # Fault zones:     3m      12      12      12      12      3m      3m      12
        # Inj_name  =  ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        ppred_proxy_bhp = self.ppred_proxy_bhp
        ipred_proxy_bhp = self.ipred_proxy_bhp
        jp12 = [2, 3, 4, 5]
        ji12 = [1, 2, 3, 4, 7]
        jp3m = [0, 1]
        ji3m = [0, 5, 6]
        ymax12 = ppred_proxy_bhp[0, jp12[0]]
        ymin12 = ppred_proxy_bhp[1, jp12[0]]
        ymax3m = ppred_proxy_bhp[0, jp3m[0]]
        ymin3m = ppred_proxy_bhp[1, jp3m[0]]
        ymax12i = ipred_proxy_bhp[0, ji12[0]]
        ymin12i = ipred_proxy_bhp[1, ji12[0]]
        ymax3mi = ipred_proxy_bhp[0, ji3m[0]]
        ymin3mi = ipred_proxy_bhp[1, ji3m[0]]

        pbhp12n = bhp12n[0, :, :len(jp12)]  # production bhp in fault zones 1 and 2
        pbhp12 = pbhp12n * (ymax12 - ymin12) + ymin12
        ibhp12n = bhp12n[0, :, len(jp12):]  # injection bhp in fault zones 1 and 2
        ibhp12 = ibhp12n * (ymax12i - ymin12i) + ymin12i

        pbhp3mn = bhp3mn[0, :, :len(jp3m)]  # production bhp in fault zones 3 and middle
        pbhp3m = pbhp3mn * (ymax3m - ymin3m) + ymin3m
        ibhp3mn = bhp3mn[0, :, len(jp3m):]  # injection bhp in fault zones 3 and middle and 36A-15
        ibhp3m = ibhp3mn * (ymax3mi - ymin3mi) + ymin3mi

        pbhp = layers.Concatenate(axis=-1)([pbhp3m, pbhp12])
        ibhp = layers.Concatenate(axis=-1)([ibhp3m[:, :1], ibhp12[:, 0:4], ibhp3m[:, 1:], ibhp12[:, 4:]])
        return pbhp, ibhp

    @tf.function
    def Expanders(self, controln):
        # it2n = self.itemp2
        # it3n = self.itemp3
        # tvar = self.tvar
        [pcontroln, icontroln] = self.spliter_PI(controln)
        # production rate
        [p1n, p2an, p2bn, p2cn, pmn, p3n] = self.ExpandProd(pcontroln)
        [p1, p2a, p2b, p2c, pm, p3] = self.backnormPC(p1n, p2an, p2bn, p2cn, pmn, p3n)
        # injection rate
        [i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n] = self.ExpandInjt(icontroln)
        [i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.backnormIC(i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n)
        # concatenate controls for fault zone 2 and 3

        # without time variable
        c1 = p1n
        c2 = layers.Concatenate(axis=-1)([p2an, p2bn, p2cn, i2an, i2bn, i2cn, i2dn, i2en])
        cm = pmn
        c3 = layers.Concatenate(axis=-1)([p3n, i3an, i3bn])
        return c1, c2, cm, c3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def backnormPC(self, p1n, p2an, p2bn, p2cn, pmn, p3n):  # Map normalized rate and enthalpy to original space
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        p1 = self.backnorm(p1n, s1[0], s1[1])
        p2a = self.backnorm(p2an, s2[0], s2[1])
        p2b = self.backnorm(p2bn, s2[0], s2[1])
        p2c = self.backnorm(p2cn, s2[0], s2[1])
        pm = self.backnorm(pmn, sm[0], sm[1])
        p3 = self.backnorm(p3n, s3[0], s3[1])
        return p1, p2a, p2b, p2c, pm, p3

    @tf.function
    def backnormIC(self, i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n):
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        i2a = self.backnorm(i2an, s2[2], s2[3])
        i2b = self.backnorm(i2bn, s2[2], s2[3])
        i2c = self.backnorm(i2cn, s2[2], s2[3])
        i2d = self.backnorm(i2dn, s2[2], s2[3])
        i2e = self.backnorm(i2en, s2[2], s2[3])
        i3a = self.backnorm(i3an, s3[2], s3[3])
        i3b = self.backnorm(i3bn, s3[2], s3[3])
        i_ = self.backnorm(i_n, s_[0], s_[1])
        return i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def backnormENTH(self, enth1n, enth2n, enthmn, enth3n):
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        enth1 = self.backnorm(enth1n, s1[-2], s1[-1])
        enth2a = self.backnorm(enth2n[:, :, 0:1], s2[-2], s2[-1])
        enth2b = self.backnorm(enth2n[:, :, 1:2], s2[-2], s2[-1])
        enth2c = self.backnorm(enth2n[:, :, 2:3], s2[-2], s2[-1])
        enthm = self.backnorm(enthmn, sm[-2], sm[-1])
        enth3 = self.backnorm(enth3n, s3[-2], s3[-1])
        return enth1, enth2a, enth2b, enth2c, enthm, enth3

    @tf.function
    def spliter_PI(self, control):  # split production wells and injection wells
        nprod = self.nprod
        nstage = self.nstage
        K = nprod * nstage
        pcontrol = layers.Lambda(lambda x: x[:, :K])(control)
        icontrol = layers.Lambda(lambda x: x[:, K:])(control)
        return pcontrol, icontrol

    @tf.function
    def ExpandProd(self, pcontrol):  # extend production control
        [p1_, p2a_, p2b_, p2c_, pm_, p3_] = self.splitProd(pcontrol)
        p1 = self.repmatv(p1_)
        p2a = self.repmatv(p2a_)
        p2b = self.repmatv(p2b_)
        p2c = self.repmatv(p2c_)
        pm = self.repmatv(pm_)
        p3 = self.repmatv(p3_)
        return p1, p2a, p2b, p2c, pm, p3

    @tf.function
    def ExpandInjt(self, icontrol):  # extend injection control
        [i2a_, i2b_, i2c_, i2d_, i2e_, i3a_, i3b_, i__] = self.splitInjt(icontrol)
        i2a = self.repmatv(i2a_)
        i2b = self.repmatv(i2b_)
        i2c = self.repmatv(i2c_)
        i2d = self.repmatv(i2d_)
        i2e = self.repmatv(i2e_)
        i3a = self.repmatv(i3a_)
        i3b = self.repmatv(i3b_)
        i_ = self.repmatv(i__)
        return i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def splitProd(self, pcontrol):  # split production wells
        nstage = self.nstage
        p3_ = layers.Lambda(lambda x: x[:, 0 * nstage: 1 * nstage])(pcontrol)
        pm_ = layers.Lambda(lambda x: x[:, 1 * nstage: 2 * nstage])(pcontrol)
        p2a_ = layers.Lambda(lambda x: x[:, 2 * nstage: 3 * nstage])(pcontrol)
        p2b_ = layers.Lambda(lambda x: x[:, 3 * nstage: 4 * nstage])(pcontrol)
        p1_ = layers.Lambda(lambda x: x[:, 4 * nstage: 5 * nstage])(pcontrol)
        p2c_ = layers.Lambda(lambda x: x[:, 5 * nstage: 6 * nstage])(pcontrol)
        return p1_, p2a_, p2b_, p2c_, pm_, p3_

    @tf.function
    def splitInjt(self, icontrol):
        nstage = self.nstage
        i3a_ = layers.Lambda(lambda x: x[:, 0 * nstage: 1 * nstage])(icontrol)  # 44-21
        i2a_ = layers.Lambda(lambda x: x[:, 1 * nstage: 2 * nstage])(icontrol)  # 23-17
        i2b_ = layers.Lambda(lambda x: x[:, 2 * nstage: 3 * nstage])(icontrol)  # 36-17
        i2c_ = layers.Lambda(lambda x: x[:, 3 * nstage: 4 * nstage])(icontrol)  # 37-17
        i2d_ = layers.Lambda(lambda x: x[:, 4 * nstage: 5 * nstage])(icontrol)  # 85-19
        i3b_ = layers.Lambda(lambda x: x[:, 5 * nstage: 6 * nstage])(icontrol)  # 38-21
        i__ = layers.Lambda(lambda x: x[:, 6 * nstage: 7 * nstage])(icontrol)  # 36A-15
        i2e_ = layers.Lambda(lambda x: x[:, 7 * nstage: 8 * nstage])(icontrol)  # 2429I
        return i2a_, i2b_, i2c_, i2d_, i2e_, i3a_, i3b_, i__

    @tf.function
    def splitHist(self):  # split history and control:
        history = self.history_enth
        return history[0], history[1], history[2], history[3]

    @tf.function
    def splitScaler(self):
        s = self.s
        return s[0], s[1], s[2], s[3], s[4]

    @tf.function
    def repmatv(self, x):  # vertically repeat control based on nweeks per year
        STEP = self.lsSTEP
        return tf.repeat(x, repeats=STEP, axis=1)

    @tf.function
    def sum_array(self, x):
        return layers.Add()(x)

    @tf.function
    def backnorm(self, x, xmin, xmax):
        return tf.add(tf.multiply(x, (xmax - xmin)), xmin)

    @tf.function
    def reduce_sum(self, x, a=1):
        return a * tf.reduce_sum(x, [0, 1, 2])


# with bhp-related pump cost, one model for one fault zone including prod and injt
class proxy_model1:
    def __init__(self, models_enth, models_bhp, history_enth, history_bhp, ProxyBoundEnth, ProxyBoundBHP):
        # defaults: basic setups
        nstep, nstep_, nprod, ninjt, coeffs, nstage, lsSTEP, lsSTEP_, DATE, _, itemp2, itemp3 = OptInitializer(
            year0=2021, year1=2033, month0=5, day0=21)
        self.nprod = nprod
        self.ninjt = ninjt
        self.coeffs = coeffs
        self.nstage = nstage
        self.lsSTEP = lsSTEP
        self.DATE = DATE
        self.itemp2 = itemp2
        self.itemp3 = itemp3

        # initialization
        scaler = ConvertToScaler(ProxyBoundEnth)
        self.s = scaler
        self.models_enth = models_enth
        self.models_bhp = models_bhp
        self.history_enth = history_enth
        self.history_bhp = history_bhp
        self.fval = None
        self.GPower = None
        self.PPump = None
        self.IPump = None
        self.enth1 = None
        self.enth2a = None
        self.enth2b = None
        self.enth2c = None
        self.enthm = None
        self.enth3 = None
        self.history_x = []
        self.history_f = []
        self.historyFval = {'GPower': [], 'PPump': [], 'IPump': []}
        self.historyEnth = {'enth1': [], 'enth2a': [], 'enth2b': [], 'enth2c': [], 'enthm': [], 'enth3': []}

        self.prate_proxy_enth= tf.cast(ProxyBoundEnth['Prod_bounds'], dtype='float32')
        self.irate_proxy_enth= tf.cast(ProxyBoundEnth['Injt_bounds'], dtype='float32')
        self.prate_proxy_bhp = tf.cast(ProxyBoundBHP['Prod_bounds'], dtype='float32')
        self.irate_proxy_bhp = tf.cast(ProxyBoundBHP['Injt_bounds'], dtype='float32')
        self.ppred_proxy_bhp = tf.cast(ProxyBoundBHP['PBHP_bounds'], dtype='float32')
        self.ipred_proxy_bhp = tf.cast(ProxyBoundBHP['IBHP_bounds'], dtype='float32')
        self.lower_enth= tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_enth[1,:], self.irate_proxy_enth[1,:]]), repeats=nstage)[None,:,None]
        self.upper_enth= tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_enth[0,:], self.irate_proxy_enth[0,:]]), repeats=nstage)[None,:,None]
        self.lower_bhp = tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_bhp[1,:], self.irate_proxy_bhp[1,:]]), repeats=nstage)[None,:,None]
        self.upper_bhp = tf.repeat(layers.Concatenate(axis=0)(
            [self.prate_proxy_bhp[0,:], self.irate_proxy_bhp[0,:]]), repeats=nstage)[None,:,None]

        # coefficients for pump power
        self.rho_injt = 1000
        self.rho_prod = 900
        self.pump_eff = 0.75
        psi = 6.89476
        kPa = 1
        gravity = 9.81
        TVD_prod = tf.cast(np.array([2405, 2055, 2637, 2685, 2713, 2685]), dtype='float32')
        TVD_injt = tf.cast(np.array([2.3 * 2481, 1.6 * 2637, 1.35 * 2715, 3312, 1.4 * 2321, 3044, 1.1 * 2737, 2103]),
                           dtype='float32')
        pgrad_prod = 22.6206 * 0.052 * self.rho_prod / 119.83  # kPa/m
        pgrad_injt = 22.6206 * 0.052 * self.rho_injt / 119.83  # kPa/m
        WH_inlet = 150 * psi / kPa
        WH_outlet0 = 150 * psi / kPa
        WH_outlet = 200 * psi / kPa

        self.dP_injt0 = WH_outlet - WH_outlet0
        self.BackPres_prod = TVD_prod * pgrad_prod + WH_inlet  # kPa
        self.BackPres_injt = TVD_injt * pgrad_injt + WH_outlet  # kPa

    def monitoring(self, history_x, interval = 1):
        N = history_x.shape[0] # total number of iteration
        x_filter = list(range(0, N, interval))
        if x_filter[-1] != N - 1:
            x_filter.append(N - 1)
        x_hist = history_x[x_filter, :]
        print(x_hist.shape)
        DATE = self.DATE
        lsSTEP = self.lsSTEP
        Prod_bounds = self.prate_proxy_enth.numpy()
        Injt_bounds = self.irate_proxy_enth.numpy()
        fval_sim, mask, ENTH_sim, PBHP_sim, IBHP_sim, PPump_sim, IPump_sim, RATE_sim, RATEi_sim = Monitoring(
            x_hist, DATE, lsSTEP, Prod_bounds, Injt_bounds)
        return fval_sim, mask, ENTH_sim, PBHP_sim, IBHP_sim, PPump_sim, IPump_sim, RATE_sim, RATEi_sim, x_filter

    def ReturnResult(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        PumpPower, PPump, IPump, IPump0 = self.computePumpPW(control)
        fval, NPower, GPower, _, _, enth1, enth2a, enth2b, enth2c, enthm, enth3 = self.computeFval(control)
        pbhp, ibhp, prate, irate = self.computeRateBHP(control)
        return fval, NPower, GPower, PumpPower, PPump, IPump, IPump0, \
               enth1, enth2a, enth2b, enth2c, enthm, enth3, pbhp, ibhp, prate, irate

    def Init_Optimizer(self, rate0, pbound_sim=None, ibound_sim=None, x0_type='equal'):
        if pbound_sim is None:
            pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                                   [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
        if ibound_sim is None:
            ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                                   [  0,   0,   0,   0,   0,   0,   0,   0]])

        x0, bounds, lcon, x00, Aeq, beq, A, b, ub, lb = ConstraintSetup(
            rate0, self.nstage, self.prate_proxy_enth, self.irate_proxy_enth, pbound_sim, ibound_sim, QTOTOMASS=1850,
            nprod=self.nprod, ninjt=self.ninjt, x0_type=x0_type)

        return x0, bounds, lcon, x00, Aeq, beq, A, b, ub, lb

    def objfun(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        self.fval, self.NPower, self.GPower, self.PPump, self.IPump, self.enth1, self.enth2a, self.enth2b, self.enth2c, self.enthm, self.enth3 = self.computeFval(
            control)
        return self.fval.numpy()

    def gradfun(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        grad = self.computegrad(control).numpy()
        return grad

    def savedata(self, x, res):
        historyGPower = self.historyFval['GPower']
        historyPPump = self.historyFval['PPump']
        historyIPump = self.historyFval['IPump']
        historyEnth1 = self.historyEnth['enth1']
        historyEnth2a = self.historyEnth['enth2a']
        historyEnth2b = self.historyEnth['enth2b']
        historyEnth2c = self.historyEnth['enth2c']
        historyEnthm = self.historyEnth['enthm']
        historyEnth3 = self.historyEnth['enth3']

        historyGPower.append(self.GPower.numpy())
        historyPPump.append(self.PPump.numpy())
        historyIPump.append(self.IPump.numpy())
        historyEnth1.append(self.enth1.numpy())
        historyEnth2a.append(self.enth2a.numpy())
        historyEnth2b.append(self.enth2b.numpy())
        historyEnth2c.append(self.enth2c.numpy())
        historyEnthm.append(self.enthm.numpy())
        historyEnth3.append(self.enth3.numpy())

        self.history_x.append(x)
        self.history_f.append(self.fval.numpy())
        self.historyFval = {'GPower': historyGPower,
                            'PPump': historyPPump,
                            'IPump': historyIPump}
        self.historyEnth = {'enth1': historyEnth1,
                            'enth2a': historyEnth2a,
                            'enth2b': historyEnth2b,
                            'enth2c': historyEnth2c,
                            'enthm': historyEnthm,
                            'enth3': historyEnth3}

    def Optimizer(self, x0, bounds, lcon, constr_penalty=1.0, tr_radius=1.0, barrier_para=0.1, barrier_tol=0.1):
        fun = self.objfun
        gfun = self.gradfun
        res = minimize(fun, x0, method='trust-constr', jac=gfun, hess=None,
                       bounds=bounds, constraints=lcon, tol=None,
                       callback=self.savedata,
                       options={'xtol': 1e-06, 'gtol': 1e-06,
                                'barrier_tol': 1e-08, 'sparse_jacobian': None,
                                'maxiter': 1e3, 'verbose': 2, 'finite_diff_rel_step': None,
                                'initial_constr_penalty': constr_penalty,
                                'initial_tr_radius': tr_radius,
                                'initial_barrier_parameter': barrier_para,
                                'initial_barrier_tolerance': barrier_tol,
                                'factorization_method': None, 'disp': True})
        return res, self.history_x, self.history_f, self.historyFval, self.historyEnth

    @tf.function
    def computegrad(self, control):
        output = self.Func_Val(control)
        grad = tf.reshape(tf.gradients(output, control), [-1])
        return grad

    @tf.function
    def Func_Val(self, control):
        # models  = self.models
        output = self.computeFval(control)[0]
        return output

    @tf.function
    def computeFval(self, control):
        coeffs = self.coeffs
        coeff = coeffs[0]
        coeffh = coeffs[1]
        coeff2 = coeffs[2]
        Pratio = coeffs[3]
        Iratio = coeffs[4]

        [enth1, enth2a, enth2b, enth2c, enthm, enth3,
         p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.computeEnth(control)
        prate = self.sum_array([p1, p2a, p2b, p2c, pm, p3])
        irate = self.sum_array([i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_])
        PumpPower, PPump, IPump, IPump0 = self.computePumpPW(control)

        ENTH1 = tf.multiply(p1, enth1)
        ENTH2a = tf.multiply(p2a, enth2a)
        ENTH2b = tf.multiply(p2b, enth2b)
        ENTH2c = tf.multiply(p2c, enth2c)
        ENTHm = tf.multiply(pm, enthm)
        ENTH3 = tf.multiply(p3, enth3)
        ENTH = self.sum_array([ENTH1, ENTH2a, ENTH2b, ENTH2c, ENTHm, ENTH3])
        ENTMWe = self.reduce_sum(ENTH, coeff2 * coeff)
        iENTH = coeffh * irate
        iENTMWe = self.reduce_sum(iENTH, coeff2 * coeff)
        GPower = ENTMWe - iENTMWe
        NPower = GPower - PumpPower
        fval = - NPower
        return fval, NPower, GPower, PPump, IPump, enth1, enth2a, enth2b, enth2c, enthm, enth3

    @tf.function
    def computeEnth(self, control):
        models_enth = self.models_enth
        # history = self.history
        f1 = models_enth[0]
        f2 = models_enth[1]
        fm = models_enth[2]
        f3 = models_enth[3]
        [h1, h2, hm, h3] = self.splitHist()
        [c1, c2, cm, c3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.Expanders(control)
        state1 = np.zeros((1, 1))  # 1
        state2 = np.zeros((1, 3))  # 3
        is_training = False * np.ones(1)
        # Compute normalized enthalpy
        enth1n = f1([h1, c1, state1, is_training])[0]
        enth2n = f2([h2, c2, state2, is_training])[0]
        enthmn = fm([hm, cm, state1, is_training])[0]
        enth3n = f3([h3, c3, state1, is_training])[0]
        [enth1, enth2a, enth2b, enth2c, enthm, enth3] = self.backnormENTH(enth1n, enth2n, enthmn, enth3n)
        return enth1, enth2a, enth2b, enth2c, enthm, enth3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def computePumpPW(self, control):
        # get bhp and rate
        pbhp, ibhp, prate, irate = self.computeRateBHP(control)

        # calculate dp
        dp_prod = tf.keras.activations.relu(self.BackPres_prod - pbhp)
        dp_injt = tf.keras.activations.relu(ibhp - self.BackPres_injt)
        dp_injt0 = self.dP_injt0

        # calculate hydraulic power for each step and well
        Epump_prod = self.HydraulicPower(dp_prod, prate, self.rho_prod)  # MWe
        Epump_injt = self.HydraulicPower(dp_injt, irate, self.rho_injt)
        Epump_injt0 = self.HydraulicPower(dp_injt0, irate, self.rho_injt)

        # calculate average pump power
        IPump0 = tf.reduce_mean(tf.reduce_sum(Epump_injt0, axis=1), axis=0) / self.pump_eff
        PPump = tf.reduce_mean(tf.reduce_sum(Epump_prod, axis=1), axis=0) / self.pump_eff
        IPump = tf.reduce_mean(tf.reduce_sum(Epump_injt, axis=1), axis=0) / self.pump_eff + IPump0
        PumpPower = PPump + IPump
        return PumpPower, PPump, IPump, IPump0

    @tf.function
    def HydraulicPower(self, dp, rate, rho):
        return dp*(rate/rho)/(3.6e3)

    @tf.function
    def computeRateBHP(self, control):
        # define model
        f12 = self.models_bhp[0]
        f3m = self.models_bhp[1]

        # split control and history for two models
        h12, h3m = self.SplitHistBHP()
        c12, c3m, praten, iraten, rate0_sim = self.SplitRateBHP(control)

        # backnorm rate
        rate0 = self.repmatv(tf.transpose(tf.reshape(rate0_sim, [1, -1, self.nstage]), perm=(0, 2, 1)))
        prate = rate0[0, :, :self.nprod]
        irate = rate0[0, :, self.nprod:]

        # return predicted normalized bhp from two models
        state12 = np.zeros((1, 9))  # 9
        state3m = np.zeros((1, 5))  # 5
        is_training = False * np.ones(1)
        bhp12n = f12([h12, c12, state12, is_training])[0]
        bhp3mn = f3m([h3m, c3m, state3m, is_training])[0]

        # backnorm bhp
        pbhp, ibhp = self.backnormBHP(bhp12n, bhp3mn)
        return pbhp, ibhp, prate, irate

    @tf.function
    def SplitRateBHP(self, control):
        rate0_sim = control*(self.upper_enth-self.lower_enth)+self.lower_enth
        x0_bhp00 = (rate0_sim-self.lower_bhp)/(self.upper_bhp-self.lower_bhp)
        x0_bhp01 = tf.reshape(x0_bhp00,(1,-1, self.nstage))
        x0_bhp11 = tf.transpose(x0_bhp01, perm=(0,2,1))
        x0_bhp = self.repmatv(x0_bhp11)
        praten = x0_bhp[:,:,:self.nprod]
        iraten = x0_bhp[:,:,self.nprod:]
        p12 = layers.Concatenate(axis=-1)([praten[:,:,2:3],praten[:,:,3:4],praten[:,:,4:5],praten[:,:,5:6]]) # jp12 = [2, 3, 4, 5]
        i12 = layers.Concatenate(axis=-1)([iraten[:,:,1:2],iraten[:,:,2:3],iraten[:,:,3:4],iraten[:,:,4:5],iraten[:,:,7:8]]) # ji12 = [1, 2, 3, 4, 7]
        p3m = layers.Concatenate(axis=-1)([praten[:,:,0:1],praten[:,:,1:2]]) # jp3m = [0, 1]
        i3m = layers.Concatenate(axis=-1)([iraten[:,:,0:1],iraten[:,:,5:6],iraten[:,:,6:7]]) # ji3m = [0, 5, 6]
        c12 = layers.Concatenate(axis=-1)([p12, i12])
        c3m = layers.Concatenate(axis=-1)([p3m, i3m])
        return c12, c3m, praten, iraten, rate0_sim

    @tf.function
    def SplitHistBHP(self):
        return self.history_bhp[0], self.history_bhp[1]

    @tf.function
    def backnormBHP(self, bhp12n, bhp3mn):
        # Fault zones:       3m       3m       12       12        12       12
        # Prod_name  =  ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"];
        # Fault zones:     3m      12      12      12      12      3m      3m      12
        # Inj_name  =  ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        ppred_proxy_bhp = self.ppred_proxy_bhp
        ipred_proxy_bhp = self.ipred_proxy_bhp
        jp12 = [2, 3, 4, 5]
        ji12 = [1, 2, 3, 4, 7]
        jp3m = [0, 1]
        ji3m = [0, 5, 6]
        ymax12 = ppred_proxy_bhp[0, jp12[0]]
        ymin12 = ppred_proxy_bhp[1, jp12[0]]
        ymax3m = ppred_proxy_bhp[0, jp3m[0]]
        ymin3m = ppred_proxy_bhp[1, jp3m[0]]
        ymax12i = ipred_proxy_bhp[0, ji12[0]]
        ymin12i = ipred_proxy_bhp[1, ji12[0]]
        ymax3mi = ipred_proxy_bhp[0, ji3m[0]]
        ymin3mi = ipred_proxy_bhp[1, ji3m[0]]

        pbhp12n = bhp12n[0, :, :len(jp12)]  # production bhp in fault zones 1 and 2
        pbhp12 = pbhp12n * (ymax12 - ymin12) + ymin12
        ibhp12n = bhp12n[0, :, len(jp12):]  # injection bhp in fault zones 1 and 2
        ibhp12 = ibhp12n * (ymax12i - ymin12i) + ymin12i

        pbhp3mn = bhp3mn[0, :, :len(jp3m)]  # production bhp in fault zones 3 and middle
        pbhp3m = pbhp3mn * (ymax3m - ymin3m) + ymin3m
        ibhp3mn = bhp3mn[0, :, len(jp3m):]  # injection bhp in fault zones 3 and middle and 36A-15
        ibhp3m = ibhp3mn * (ymax3mi - ymin3mi) + ymin3mi

        pbhp = layers.Concatenate(axis=-1)([pbhp3m, pbhp12])
        ibhp = layers.Concatenate(axis=-1)([ibhp3m[:, :1], ibhp12[:, 0:4], ibhp3m[:, 1:], ibhp12[:, 4:]])
        return pbhp, ibhp

    @tf.function
    def Expanders(self, controln):
        # it2n = self.itemp2
        # it3n = self.itemp3
        # tvar = self.tvar
        [pcontroln, icontroln] = self.spliter_PI(controln)
        # production rate
        [p1n, p2an, p2bn, p2cn, pmn, p3n] = self.ExpandProd(pcontroln)
        [p1, p2a, p2b, p2c, pm, p3] = self.backnormPC(p1n, p2an, p2bn, p2cn, pmn, p3n)
        # injection rate
        [i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n] = self.ExpandInjt(icontroln)
        [i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_] = self.backnormIC(i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n)
        # concatenate controls for fault zone 2 and 3

        # without time variable
        c1 = p1n
        c2 = layers.Concatenate(axis=-1)([p2an, p2bn, p2cn, i2an, i2bn, i2cn, i2dn, i2en])
        cm = pmn
        c3 = layers.Concatenate(axis=-1)([p3n, i3an, i3bn])
        return c1, c2, cm, c3, p1, p2a, p2b, p2c, pm, p3, i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def backnormPC(self, p1n, p2an, p2bn, p2cn, pmn, p3n):  # Map normalized rate and enthalpy to original space
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        p1 = self.backnorm(p1n, s1[0], s1[1])
        p2a = self.backnorm(p2an, s2[0], s2[1])
        p2b = self.backnorm(p2bn, s2[0], s2[1])
        p2c = self.backnorm(p2cn, s2[0], s2[1])
        pm = self.backnorm(pmn, sm[0], sm[1])
        p3 = self.backnorm(p3n, s3[0], s3[1])
        return p1, p2a, p2b, p2c, pm, p3

    @tf.function
    def backnormIC(self, i2an, i2bn, i2cn, i2dn, i2en, i3an, i3bn, i_n):
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        i2a = self.backnorm(i2an, s2[2], s2[3])
        i2b = self.backnorm(i2bn, s2[2], s2[3])
        i2c = self.backnorm(i2cn, s2[2], s2[3])
        i2d = self.backnorm(i2dn, s2[2], s2[3])
        i2e = self.backnorm(i2en, s2[2], s2[3])
        i3a = self.backnorm(i3an, s3[2], s3[3])
        i3b = self.backnorm(i3bn, s3[2], s3[3])
        i_ = self.backnorm(i_n, s_[0], s_[1])
        return i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def backnormENTH(self, enth1n, enth2n, enthmn, enth3n):
        # s = self.s
        [s1, s2, sm, s3, s_] = self.splitScaler()
        enth1 = self.backnorm(enth1n, s1[-2], s1[-1])
        enth2a = self.backnorm(enth2n[:, :, 0:1], s2[-2], s2[-1])
        enth2b = self.backnorm(enth2n[:, :, 1:2], s2[-2], s2[-1])
        enth2c = self.backnorm(enth2n[:, :, 2:3], s2[-2], s2[-1])
        enthm = self.backnorm(enthmn, sm[-2], sm[-1])
        enth3 = self.backnorm(enth3n, s3[-2], s3[-1])
        return enth1, enth2a, enth2b, enth2c, enthm, enth3

    @tf.function
    def spliter_PI(self, control):  # split production wells and injection wells
        nprod = self.nprod
        nstage = self.nstage
        K = nprod * nstage
        pcontrol = layers.Lambda(lambda x: x[:, :K])(control)
        icontrol = layers.Lambda(lambda x: x[:, K:])(control)
        return pcontrol, icontrol

    @tf.function
    def ExpandProd(self, pcontrol):  # extend production control
        [p1_, p2a_, p2b_, p2c_, pm_, p3_] = self.splitProd(pcontrol)
        p1 = self.repmatv(p1_)
        p2a = self.repmatv(p2a_)
        p2b = self.repmatv(p2b_)
        p2c = self.repmatv(p2c_)
        pm = self.repmatv(pm_)
        p3 = self.repmatv(p3_)
        return p1, p2a, p2b, p2c, pm, p3

    @tf.function
    def ExpandInjt(self, icontrol):  # extend injection control
        [i2a_, i2b_, i2c_, i2d_, i2e_, i3a_, i3b_, i__] = self.splitInjt(icontrol)
        i2a = self.repmatv(i2a_)
        i2b = self.repmatv(i2b_)
        i2c = self.repmatv(i2c_)
        i2d = self.repmatv(i2d_)
        i2e = self.repmatv(i2e_)
        i3a = self.repmatv(i3a_)
        i3b = self.repmatv(i3b_)
        i_ = self.repmatv(i__)
        return i2a, i2b, i2c, i2d, i2e, i3a, i3b, i_

    @tf.function
    def splitProd(self, pcontrol):  # split production wells
        nstage = self.nstage
        p3_ = layers.Lambda(lambda x: x[:, 0 * nstage: 1 * nstage])(pcontrol)
        pm_ = layers.Lambda(lambda x: x[:, 1 * nstage: 2 * nstage])(pcontrol)
        p2a_ = layers.Lambda(lambda x: x[:, 2 * nstage: 3 * nstage])(pcontrol)
        p2b_ = layers.Lambda(lambda x: x[:, 3 * nstage: 4 * nstage])(pcontrol)
        p1_ = layers.Lambda(lambda x: x[:, 4 * nstage: 5 * nstage])(pcontrol)
        p2c_ = layers.Lambda(lambda x: x[:, 5 * nstage: 6 * nstage])(pcontrol)
        return p1_, p2a_, p2b_, p2c_, pm_, p3_

    @tf.function
    def splitInjt(self, icontrol):
        nstage = self.nstage
        i3a_ = layers.Lambda(lambda x: x[:, 0 * nstage: 1 * nstage])(icontrol)  # 44-21
        i2a_ = layers.Lambda(lambda x: x[:, 1 * nstage: 2 * nstage])(icontrol)  # 23-17
        i2b_ = layers.Lambda(lambda x: x[:, 2 * nstage: 3 * nstage])(icontrol)  # 36-17
        i2c_ = layers.Lambda(lambda x: x[:, 3 * nstage: 4 * nstage])(icontrol)  # 37-17
        i2d_ = layers.Lambda(lambda x: x[:, 4 * nstage: 5 * nstage])(icontrol)  # 85-19
        i3b_ = layers.Lambda(lambda x: x[:, 5 * nstage: 6 * nstage])(icontrol)  # 38-21
        i__ = layers.Lambda(lambda x: x[:, 6 * nstage: 7 * nstage])(icontrol)  # 36A-15
        i2e_ = layers.Lambda(lambda x: x[:, 7 * nstage: 8 * nstage])(icontrol)  # 2429I
        return i2a_, i2b_, i2c_, i2d_, i2e_, i3a_, i3b_, i__

    @tf.function
    def splitHist(self):  # split history and control:
        history = self.history_enth
        return history[0], history[1], history[2], history[3]

    @tf.function
    def splitScaler(self):
        s = self.s
        return s[0], s[1], s[2], s[3], s[4]

    @tf.function
    def repmatv(self, x):  # vertically repeat control based on nweeks per year
        STEP = self.lsSTEP
        return tf.repeat(x, repeats=STEP, axis=1)

    @tf.function
    def sum_array(self, x):
        return layers.Add()(x)

    @tf.function
    def backnorm(self, x, xmin, xmax):
        return tf.add(tf.multiply(x, (xmax - xmin)), xmin)

    @tf.function
    def reduce_sum(self, x, a=1):
        return a * tf.reduce_sum(x, [0, 1, 2])


# only for pump power debugging
class proxy_model_bhp:
    def __init__(self, models_bhp, history_bhp, ProxyBoundEnth, ProxyBoundBHP, nstage, lsSTEP):
        self.prate_proxy_enth = tf.cast(ProxyBoundEnth['Prod_bounds'], dtype='float32')
        self.irate_proxy_enth = tf.cast(ProxyBoundEnth['Injt_bounds'], dtype='float32')
        self.prate_proxy_bhp = tf.cast(ProxyBoundBHP['Prod_bounds'], dtype='float32')
        self.irate_proxy_bhp = tf.cast(ProxyBoundBHP['Injt_bounds'], dtype='float32')
        self.ppred_proxy_bhp = tf.cast(ProxyBoundBHP['PBHP_bounds'], dtype='float32')
        self.ipred_proxy_bhp = tf.cast(ProxyBoundBHP['IBHP_bounds'], dtype='float32')

        self.lower_enth = tf.repeat(layers.Concatenate(axis=0)([self.prate_proxy_enth[1, :],
                                                                self.irate_proxy_enth[1, :]]), repeats=nstage)[None, :,
                          None]
        self.upper_enth = tf.repeat(layers.Concatenate(axis=0)([self.prate_proxy_enth[0, :],
                                                                self.irate_proxy_enth[0, :]]), repeats=nstage)[None, :,
                          None]
        self.lower_bhp = tf.repeat(layers.Concatenate(axis=0)([self.prate_proxy_bhp[1, :],
                                                               self.irate_proxy_bhp[1, :]]), repeats=nstage)[None, :,
                         None]
        self.upper_bhp = tf.repeat(layers.Concatenate(axis=0)([self.prate_proxy_bhp[0, :],
                                                               self.irate_proxy_bhp[0, :]]), repeats=nstage)[None, :,
                         None]

        self.nprod = 6
        self.ninjt = 8
        self.nstage = nstage
        self.lsSTEP = lsSTEP
        self.history_bhp = history_bhp
        self.models_bhp = models_bhp

        # coefficients for pump power
        self.rho_injt = 1000
        self.rho_prod = 900
        self.pump_eff = 0.75
        psi = 6.89476
        kPa = 1
        gravity = 9.81
        TVD_prod = tf.cast(np.array([2405, 2055, 2637, 2685, 2713, 2685]), dtype='float32')
        TVD_injt = tf.cast(np.array([2.3 * 2481, 1.6 * 2637, 1.35 * 2715, 3312, 1.4 * 2321, 3044, 1.1 * 2737, 2103]),
                           dtype='float32')
        pgrad_prod = 22.6206 * 0.052 * self.rho_prod / 119.83  # kPa/m
        pgrad_injt = 22.6206 * 0.052 * self.rho_injt / 119.83  # kPa/m
        WH_inlet = 150 * psi / kPa
        WH_outlet0 = 150 * psi / kPa
        WH_outlet = 200 * psi / kPa

        self.dP_injt0 = WH_outlet - WH_outlet0
        self.BackPres_prod = TVD_prod * pgrad_prod + WH_inlet  # kPa
        self.BackPres_injt = TVD_injt * pgrad_injt + WH_outlet  # kPa

    def gradfun(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        grad = self.computegrad(control).numpy()
        return grad

    def ReturnResult(self, x0):
        control = tf.cast(x0.reshape((1, -1, 1)), dtype='float32')
        pumppower, ppump, ipump, ipump0 = self.computePumpPW(control)
        return pumppower, ppump, ipump, ipump0

    @tf.function
    def computegrad(self, control):
        output = self.computeFval(control)
        grad = tf.reshape(tf.gradients(output, control), [-1])
        return grad

    @tf.function
    def computeFval(self, control):
        pumppower, ppump, ipump, ipump0 = self.computePumpPW(control)
        return pumppower

    @tf.function
    def computePumpPW(self, control):
        # get bhp and rate
        pbhp, ibhp, prate, irate = self.computeRateBHP(control)

        # calculate dp
        dp_prod = tf.keras.activations.relu(self.BackPres_prod - pbhp)
        dp_injt = tf.keras.activations.relu(ibhp - self.BackPres_injt)
        dp_injt0 = self.dP_injt0

        # calculate hydraulic power for each step and well
        Epump_prod = self.HydraulicPower(dp_prod, prate, self.rho_prod)  # MWe
        Epump_injt = self.HydraulicPower(dp_injt, irate, self.rho_injt)
        Epump_injt0 = self.HydraulicPower(dp_injt0, irate, self.rho_injt)

        # calculate average pump power
        ipump0 = tf.reduce_mean(tf.reduce_sum(Epump_injt0, axis=1), axis=0) / self.pump_eff
        ppump = tf.reduce_mean(tf.reduce_sum(Epump_prod, axis=1), axis=0) / self.pump_eff
        ipump = tf.reduce_mean(tf.reduce_sum(Epump_injt, axis=1), axis=0) / self.pump_eff + ipump0
        pumppower = ppump + ipump + ipump0

        return pumppower, ppump, ipump, ipump0

    @tf.function
    def HydraulicPower(self, dp, rate, rho):
        return dp * (rate / rho) / (3.6e3)

    @tf.function
    def computeRateBHP(self, control):
        # define model
        f12 = self.models_bhp[0]
        f3m = self.models_bhp[1]

        # split control and history for two models
        h12, h3m = self.SplitHistBHP()
        c12, c3m, praten, iraten, rate0_sim = self.SplitRateBHP(control)

        # backnorm rate
        rate0 = self.repmatv(tf.transpose(tf.reshape(rate0_sim, [1, -1, self.nstage]), perm=(0, 2, 1)))
        prate = rate0[0, :, :self.nprod]
        irate = rate0[0, :, self.nprod:]

        # return predicted normalized bhp from two models
        state12 = np.zeros((1, 9))  # 9
        state3m = np.zeros((1, 5))  # 5
        is_training = False * np.ones(1)
        bhp12n = f12([h12, c12, state12, is_training])[0]
        bhp3mn = f3m([h3m, c3m, state3m, is_training])[0]

        # backnorm bhp
        pbhp, ibhp = self.backnormBHP(bhp12n, bhp3mn)
        return pbhp, ibhp, prate, irate

    @tf.function
    def SplitRateBHP(self, control):
        rate0_sim = control * (self.upper_enth - self.lower_enth) + self.lower_enth
        x0_bhp00 = (rate0_sim - self.lower_bhp) / (self.upper_bhp - self.lower_bhp)
        x0_bhp01 = tf.reshape(x0_bhp00, (1, -1, self.nstage))
        x0_bhp11 = tf.transpose(x0_bhp01, perm=(0, 2, 1))
        x0_bhp = self.repmatv(x0_bhp11)
        praten = x0_bhp[:, :, :self.nprod]
        iraten = x0_bhp[:, :, self.nprod:]
        p12 = layers.Concatenate(axis=-1)(
            [praten[:, :, 2:3], praten[:, :, 3:4], praten[:, :, 4:5], praten[:, :, 5:6]])  # jp12 = [2, 3, 4, 5]
        i12 = layers.Concatenate(axis=-1)([iraten[:, :, 1:2], iraten[:, :, 2:3], iraten[:, :, 3:4], iraten[:, :, 4:5],
                                           iraten[:, :, 7:8]])  # ji12 = [1, 2, 3, 4, 7]
        p3m = layers.Concatenate(axis=-1)([praten[:, :, 0:1], praten[:, :, 1:2]])  # jp3m = [0, 1]
        i3m = layers.Concatenate(axis=-1)([iraten[:, :, 0:1], iraten[:, :, 5:6], iraten[:, :, 6:7]])  # ji3m = [0, 5, 6]
        c12 = layers.Concatenate(axis=-1)([p12, i12])
        c3m = layers.Concatenate(axis=-1)([p3m, i3m])
        return c12, c3m, praten, iraten, rate0_sim

    @tf.function
    def SplitHistBHP(self):
        return self.history_bhp[0], self.history_bhp[1]

    @tf.function
    def backnormBHP(self, bhp12n, bhp3mn):
        # Fault zones:       3m       3m       12       12        12       12
        # Prod_name  =  ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"];
        # Fault zones:     3m      12      12      12      12      3m      3m      12
        # Inj_name  =  ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        ppred_proxy_bhp = self.ppred_proxy_bhp
        ipred_proxy_bhp = self.ipred_proxy_bhp
        jp12 = [2, 3, 4, 5]
        ji12 = [1, 2, 3, 4, 7]
        jp3m = [0, 1]
        ji3m = [0, 5, 6]
        ymax12 = ppred_proxy_bhp[0, jp12[0]]
        ymin12 = ppred_proxy_bhp[1, jp12[0]]
        ymax3m = ppred_proxy_bhp[0, jp3m[0]]
        ymin3m = ppred_proxy_bhp[1, jp3m[0]]
        ymax12i = ipred_proxy_bhp[0, ji12[0]]
        ymin12i = ipred_proxy_bhp[1, ji12[0]]
        ymax3mi = ipred_proxy_bhp[0, ji3m[0]]
        ymin3mi = ipred_proxy_bhp[1, ji3m[0]]

        pbhp12n = bhp12n[0, :, :len(jp12)]  # production bhp in fault zones 1 and 2
        pbhp12 = pbhp12n * (ymax12 - ymin12) + ymin12
        ibhp12n = bhp12n[0, :, len(jp12):]  # injection bhp in fault zones 1 and 2
        ibhp12 = ibhp12n * (ymax12i - ymin12i) + ymin12i

        pbhp3mn = bhp3mn[0, :, :len(jp3m)]  # production bhp in fault zones 3 and middle
        pbhp3m = pbhp3mn * (ymax3m - ymin3m) + ymin3m
        ibhp3mn = bhp3mn[0, :, len(jp3m):]  # injection bhp in fault zones 3 and middle and 36A-15
        ibhp3m = ibhp3mn * (ymax3mi - ymin3mi) + ymin3mi

        pbhp = layers.Concatenate(axis=-1)([pbhp3m, pbhp12])
        ibhp = layers.Concatenate(axis=-1)([ibhp3m[:, :1], ibhp12[:, 0:4], ibhp3m[:, 1:], ibhp12[:, 4:]])
        return pbhp, ibhp

    @tf.function
    def repmatv(self, x):  # vertically repeat control based on nweeks per year
        STEP = self.lsSTEP
        return tf.repeat(x, repeats=STEP, axis=1)

    @tf.function
    def reduce_sum(self, x, a=1):
        return a * tf.reduce_sum(x, [0, 1, 2])