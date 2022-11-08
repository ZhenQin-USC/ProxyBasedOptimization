import sys

sys.path.append(r'D:\Users\qinzh\Google Drive USC\MATLAB Local\Proxy Opt')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from os.path import join
import scipy.io as sio
import numpy as np
import tensorflow as tf


def load_samples(path):
    ENTH_sim = sio.loadmat(path)['ENTH']
    PTAV_sim = sio.loadmat(path)['PTAV']
    PPAV_sim = sio.loadmat(path)['PPAV']
    PBHP_sim = sio.loadmat(path)['PBHP']
    IBHP_sim = sio.loadmat(path)['IBHP']
    RATE_sim = sio.loadmat(path)['RATE']
    RATEi_sim= sio.loadmat(path)['RATEi']
    return ENTH_sim, PTAV_sim, PPAV_sim, PBHP_sim, IBHP_sim, RATE_sim, RATEi_sim


def normalization(prop):
    # prop = block_reduce(prop, block_size = (24,1), func = np.mean)
    # prop = scipy.signal.savgol_filter(prop[:-1], 1001, 2, mode = 'mirror',axis = 0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return np.reshape(scaler.fit_transform(prop.reshape([-1, 1])), [-1, prop.shape[1]])


def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])
    if limit is not None:
        invalid = ~valid
        for n in range(1, limit + 1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan
    return filled


def GenerateSets(x, y, frequency, twindow, twindow2, predictwindow):
    yin = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow, y.shape[1]))
    xin = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow, x.shape[1]))
    yout = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow2, y.shape[1]))
    control = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, twindow2, x.shape[1]))
    obs = np.zeros((x.shape[0] - (twindow + twindow2) * frequency + 1, y.shape[1]))
    for i in range(x.shape[0] - (twindow + twindow2) * frequency + 1):
        for j in range(twindow):
            yin[i, j] = y[i + j * frequency]
            xin[i, j] = x[i + j * frequency]
        for j in range(twindow2):
            yout[i, j] = y[twindow * frequency + i + j * frequency]
            control[i, j] = x[twindow * frequency + i + j * frequency]
        obs[i] = y[twindow * frequency + i]
    history = np.concatenate((xin, yin), axis=2)
    n = int(yout.shape[0] / twindow2 / frequency)
    traintesthistory = np.zeros((n, twindow, x.shape[1] + y.shape[1]))
    traintestcontrol = np.zeros((n, twindow2, x.shape[1]))
    for i in range(n):
        traintesthistory[i] = history[i * twindow2 * frequency]
        traintestcontrol[i] = control[i * twindow2 * frequency]
    stepsize = 1
    initialwindow = yout.shape[0] - predictwindow
    index = np.random.permutation(initialwindow)
    historyt = (history[:initialwindow])[index]
    controlt = (control[:initialwindow])[index]
    youtt = (yout[:initialwindow])[index]
    return history, control, yout, historyt, controlt, youtt, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin


def split_enth(History_Data, twindow=0):
    controls_Inj_Rate = History_Data['controls_Inj_Rate']
    controls_Prod_Rate = History_Data['controls_Prod_Rate']
    predicts_Prod_Enth = History_Data['predicts_Prod_Enth']
    params = {'ncontrol1': [], 'nfeature1': [],
              'ncontrol2': [], 'nfeature2': [],
              'ncontrolm': [], 'nfeaturem': [],
              'ncontrol3': [], 'nfeature3': [],
              'ncontrol_': [], 'nfeature_': []}
    # fz1
    y1 = np.abs(predicts_Prod_Enth[:, 4:5])
    x1 = np.abs(controls_Prod_Rate[:, 4:5])
    params['ncontrol1'] = x1.shape[1]
    params['nfeature1'] = y1.shape[1]
    # fz2
    jp = [2, 3, 5]
    ji = [1, 2, 3, 4, 7]
    y2 = np.abs(predicts_Prod_Enth[:, jp])
    x2a = np.abs(controls_Prod_Rate[:, jp])
    x2b = np.abs(controls_Inj_Rate[:, ji])
    # x2c= np.abs(controls_Inj_Temp[:, ji])
    x2 = np.concatenate((x2a, x2b), axis=1)
    params['ncontrol2'] = x2.shape[1]
    params['nfeature2'] = y2.shape[1]
    # fzm
    ym = np.abs(predicts_Prod_Enth[:, 1:2])
    xm = np.abs(controls_Prod_Rate[:, 1:2])
    params['ncontrolm'] = xm.shape[1]
    params['nfeaturem'] = ym.shape[1]
    # fz3
    ji = [0, 5]
    y3 = np.abs(predicts_Prod_Enth[:, 0:1])
    x3a = np.abs(controls_Prod_Rate[:, 0:1])
    x3b = np.abs(controls_Inj_Rate[:, ji])
    # x3c= np.abs(controls_Inj_Temp[:, ji])
    x3 = np.concatenate((x3a, x3b), axis=1)
    params['ncontrol3'] = x3.shape[1]
    params['nfeature3'] = y3.shape[1]
    # 36A-15
    x_ = np.abs(controls_Inj_Rate[:, 6:7])[-twindow:, :]
    params['ncontrol_'] = x_.shape[1]
    params['nfeature_'] = 0
    # history = [control, feature]
    h1 = np.concatenate((x1, y1), axis=1)[-twindow:, :]
    h2 = np.concatenate((x2, y2), axis=1)[-twindow:, :]
    hm = np.concatenate((xm, ym), axis=1)[-twindow:, :]
    h3 = np.concatenate((x3, y3), axis=1)[-twindow:, :]
    history = [h1, h2, hm, h3, x_]
    return history, params


# four models: FZ12, FZ12i, FZ3m, FZ3mi
def split_bhp(Dict_Data, twindow=0):
    controls_Prod_Rate = Dict_Data['controls_Prod_Rate']
    controls_Inj_Rate = Dict_Data['controls_Inj_Rate']
    predicts_Prod_BHP = Dict_Data['predicts_Prod_BHP']
    predicts_Inj_BHP = Dict_Data['predicts_Inj_BHP']
    params = {'ncontrol12': [], 'nfeature12': [], 'ncontrol12i': [], 'nfeature12i': [],
              'ncontrol3m': [], 'nfeature3m': [], 'ncontrol3mi': [], 'nfeature3mi': []}
    # fz12
    # Prod_name = ["21-28", "78-20", "88-19","77A-19", "21-19", "77-19"];
    # Fault zones:    3m      3m       12       12        12       12
    jp12 = [2, 3, 4, 5]  # [2,3,5] -- FZ2, [4] -- FZ1
    # Injt_name = ["44-21", "23-17", "36-17", "37-17", "85-19", "38-21","36A-15","2429I"];
    # Fault zones:    3m       12       12       12       12       3m       3m      12
    ji12 = [1, 2, 3, 4, 7]
    x12p= np.abs(controls_Prod_Rate[:, jp12])  # ProdRate
    y12p= np.abs(predicts_Prod_BHP[:, jp12])  # ProdBHP
    x12i= np.abs(controls_Inj_Rate[:, ji12])  # InjtRate
    y12i= np.abs(predicts_Inj_BHP[:, ji12])  # InjtBHP
    x12 = np.concatenate((x12p, x12i), axis=1)  # ProdRate + InjtRate
    params['ncontrol12'] = x12.shape[1]
    params['nfeature12'] = y12p.shape[1]
    params['ncontrol12i'] = x12.shape[1]
    params['nfeature12i'] = y12i.shape[1]

    # fz3m_
    # Prod_name = ["21-28", "78-20", "88-19","77A-19", "21-19", "77-19"];
    # Fault zones:    3m      3m       12       12        12       12
    jp3m = [0, 1]  # [0] -- FZ3, [1] -- FZm
    # Injt_name = ["44-21", "23-17", "36-17", "37-17", "85-19", "38-21","36A-15","2429I"];
    # Fault zones:    3m       12       12       12       12       3m       3m      12
    ji3m = [0, 5, 6]  # [0,5] -- fz3, [6] -- 36A-15
    y3mp= np.abs(predicts_Prod_BHP[:, jp3m])  # ProdBHP
    y3mi= np.abs(predicts_Inj_BHP[:, ji3m])  # InjtBHP
    x3mp= np.abs(controls_Prod_Rate[:, jp3m])  # ProdRate
    x3mi= np.abs(controls_Inj_Rate[:, ji3m])  # InjtRate
    x3m = np.concatenate((x3mp, x3mi), axis=1)  # ProdRate + InjtRate
    params['ncontrol3m'] = x3m.shape[1]
    params['nfeature3m'] = y3mp.shape[1]
    params['ncontrol3mi'] = x3m.shape[1]
    params['nfeature3mi'] = y3mi.shape[1]

    # history = [control, feature]
    h12p = np.concatenate((x12, y12p), axis=1)[-twindow:, :]
    h3mp = np.concatenate((x3m, y3mp), axis=1)[-twindow:, :]
    h12i = np.concatenate((x12, y12i), axis=1)[-twindow:, :]
    h3mi = np.concatenate((x3m, y3mi), axis=1)[-twindow:, :]
    history = [h12p, h3mp, h12i, h3mi]
    return history, params

# only two models, each model include both production and injection BHP
def split_bhp2(Dict_Data, twindow=0):
    controls_Prod_Rate = Dict_Data['controls_Prod_Rate']
    controls_Inj_Rate = Dict_Data['controls_Inj_Rate']
    predicts_Prod_BHP = Dict_Data['predicts_Prod_BHP']
    predicts_Inj_BHP = Dict_Data['predicts_Inj_BHP']
    params = {'ncontrol12': [], 'nfeature12': [],
              'ncontrol3m_': [], 'nfeature3m_': []}
    # fz12
    # Prod_name = ["21-28", "78-20", "88-19","77A-19", "21-19", "77-19"];
    # Fault zones:    3m      3m       12       12        12       12
    jp12 = [2, 3, 4, 5]  # [2,3,5] -- FZ2, [4] -- FZ1
    # Injt_name = ["44-21", "23-17", "36-17", "37-17", "85-19", "38-21","36A-15","2429I"];
    # Fault zones:    3m       12       12       12       12       3m       3m      12
    ji12 = [1, 2, 3, 4, 7]
    y2a = np.abs(predicts_Prod_BHP[:, jp12])  # ProdBHP
    y2b = np.abs(predicts_Inj_BHP[:, ji12])  # InjtBHP
    x2a = np.abs(controls_Prod_Rate[:, jp12])  # ProdRate
    x2b = np.abs(controls_Inj_Rate[:, ji12])  # InjtRate
    x12 = np.concatenate((x2a, x2b), axis=1) # ProdRate + InjtRate
    y12 = np.concatenate((y2a, y2b), axis=1) # ProdBHP + InjtBHP
    params['ncontrol12'] = x12.shape[1]
    params['nfeature12'] = y12.shape[1]
    # fz3m_
    # Prod_name = ["21-28", "78-20", "88-19","77A-19", "21-19", "77-19"];
    # Fault zones:    3m      3m       12       12        12       12
    jp3m = [0, 1]  # [0] -- FZ3, [1] -- FZm
    # Injt_name = ["44-21", "23-17", "36-17", "37-17", "85-19", "38-21","36A-15","2429I"];
    # Fault zones:    3m       12       12       12       12       3m       3m      12
    ji3m = [0, 5, 6]  # [0,5] -- fz3, [6] -- 36A-15
    y3a = np.abs(predicts_Prod_BHP[:, jp3m])  # ProdBHP
    y3b = np.abs(predicts_Inj_BHP[:, ji3m])  # InjtBHP
    x3a = np.abs(controls_Prod_Rate[:, jp3m])  # ProdRate
    x3b = np.abs(controls_Inj_Rate[:, ji3m])  # InjtRate
    y3m = np.concatenate((y3a, y3b), axis=1)  # ProdBHP + InjtBHP
    x3m = np.concatenate((x3a, x3b), axis=1)  # ProdRate + InjtRate
    params['ncontrol3m'] = x3m.shape[1]
    params['nfeature3m'] = y3m.shape[1]
    # history = [control, feature]
    h12 = np.concatenate((x12, y12), axis=1)[-twindow:, :]
    h3m = np.concatenate((x3m, y3m), axis=1)[-twindow:, :]
    history = [h12, h3m]
    return history, params


def split_bhp3(History_Data, twindow=0):
    controls_Prod_Rate = History_Data['controls_Prod_Rate']
    controls_Inj_Rate = History_Data['controls_Inj_Rate']
    predicts_Prod_BHP = History_Data['predicts_Prod_BHP']
    predicts_Inj_BHP = History_Data['predicts_Inj_BHP']
    params = {'ncontrol1': [], 'nfeature1': [],
              'ncontrol2': [], 'nfeature2': [],
              'ncontrol3m': [], 'nfeature3m': [],
              'ncontrol_': [], 'nfeature_': []}
    # fz1
    y1 = np.abs(predicts_Prod_BHP[:, 4:5])
    x1 = np.abs(controls_Prod_Rate[:, 4:5])
    params['ncontrol1'] = x1.shape[1]
    params['nfeature1'] = y1.shape[1]
    # fz2
    jp = [2, 3, 5]
    ji = [1, 2, 3, 4, 7]
    y2a = np.abs(predicts_Prod_BHP[:, jp])
    y2b = np.abs(predicts_Inj_BHP[:, ji])
    x2a = np.abs(controls_Prod_Rate[:, jp])
    x2b = np.abs(controls_Inj_Rate[:, ji])
    x2 = np.concatenate((x2a, x2b), axis=1)
    y2 = np.concatenate((y2a, y2b), axis=1)
    params['ncontrol2'] = x2.shape[1]
    params['nfeature2'] = y2.shape[1]
    # fz3m
    ym = np.abs(predicts_Prod_BHP[:, 1:2])
    xm = np.abs(controls_Prod_Rate[:, 1:2])
    ji = [0, 5]  # fz3
    y3a = np.abs(predicts_Prod_BHP[:, 0:1])
    y3b = np.abs(predicts_Inj_BHP[:, ji])
    x3a = np.abs(controls_Prod_Rate[:, 0:1])
    x3b = np.abs(controls_Inj_Rate[:, ji])
    y3 = np.concatenate((y3a, y3b), axis=1)
    x3 = np.concatenate((x3a, x3b), axis=1)
    x3m = np.concatenate((xm, x3), axis=1)
    y3m = np.concatenate((ym, y3), axis=1)
    params['ncontrol3m'] = x3m.shape[1]
    params['nfeature3m'] = y3m.shape[1]
    # 36A-15
    x_ = np.abs(controls_Inj_Rate[:, 6:7])
    y_ = np.abs(predicts_Inj_BHP[:, 6:7])
    params['ncontrol_'] = x_.shape[1]
    params['nfeature_'] = y_.shape[1]
    # history = [control, feature]
    h1 = np.concatenate((x1, y1), axis=1)[-twindow:, :]
    h2 = np.concatenate((x2, y2), axis=1)[-twindow:, :]
    h3m = np.concatenate((x3m, y3m), axis=1)[-twindow:, :]
    h_ = np.concatenate((x_, y_), axis=1)[-twindow:, :]
    history = [h1, h2, h3m, h_]
    return history, params


def process_enth(history_enth, predict_enth, params, twindow=6, pbound_sim=None, ibound_sim=None, Enth_bounds=None):
    if pbound_sim is None:
        # Prod_name  =        ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"];
        # Fault zones:             3        m        2        2         1        2
        pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                               [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
    if ibound_sim is None:
        # Inj_name  =        ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        # Fault zones:            3       2       2       2       2       3       -        2
        ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
    if Enth_bounds is None:
        y1min = None
        y1max = None
        y2min = None
        y2max = None
        ymmin = None
        ymmax = None
        y3min = None
        y3max = None
    else:
        y1max = Enth_bounds[0, :][4]
        y1min = Enth_bounds[1, :][4]
        y2max = Enth_bounds[0, :][2]
        y2min = Enth_bounds[1, :][2]
        ymmax = Enth_bounds[0, :][1]
        ymmin = Enth_bounds[1, :][1]
        y3max = Enth_bounds[0, :][0]
        y3min = Enth_bounds[1, :][0]

    x3max = pbound_sim[0, 0]
    x3min = pbound_sim[1, 0]
    xmmax = pbound_sim[0, 1]
    xmmin = pbound_sim[1, 1]
    jp = [2, 3, 5]
    x2max = max(pbound_sim[0, jp])
    x2min = min(pbound_sim[1, jp])
    x1max = pbound_sim[0, 4]
    x1min = pbound_sim[1, 4]

    ji = [0, 5]
    x3maxi = max(ibound_sim[0, ji])
    x3mini = min(ibound_sim[1, ji])
    ji = [1, 2, 3, 4, 7]
    x2maxi = max(ibound_sim[0, ji])
    x2mini = min(ibound_sim[1, ji])

    x_maxi = ibound_sim[0, 6]
    x_mini = ibound_sim[1, 6]
    # ========== loading data ==========
    ncontrols = [params['ncontrol1'], params['ncontrol2'], params['ncontrolm'], params['ncontrol3']]
    nfeatures = [params['nfeature1'], params['nfeature2'], params['nfeaturem'], params['nfeature3']]
    fz_pwell = nfeatures  # No. of production wells in each fault zone
    fz1_enth = np.concatenate((history_enth[0], predict_enth[0]), axis=0)
    fz2_enth = np.concatenate((history_enth[1], predict_enth[1]), axis=0)
    fzm_enth = np.concatenate((history_enth[2], predict_enth[2]), axis=0)
    fz3_enth = np.concatenate((history_enth[3], predict_enth[3]), axis=0)
    fz__enth = np.concatenate((history_enth[4], predict_enth[4]), axis=0)  # 36A-15
    # print(fz1_enth.shape, fz2_enth.shape, fzm_enth.shape, fz3_enth.shape, fz__enth.shape)

    # ========== fault zone 1 ==========
    ncontrol1 = params['ncontrol1']
    X1 = fz1_enth[:, :ncontrol1, :]
    Y1 = fz1_enth[:, ncontrol1:, :]
    X1n = (X1 - x1min) / (x1max - x1min)
    if y1min is None:
        y1min = Y1.min()
        y1max = Y1.max()
    Y1n = (Y1 - y1min) / (y1max - y1min)
    s1 = np.asarray([x1min, x1max, y1min, y1max])

    # ========== fault zone 2 ==========
    np2 = fz_pwell[1]
    ncontrol2 = params['ncontrol2']
    X2p = fz2_enth[:, :ncontrol2, :][:, :np2, :]  # production rate
    X2i = fz2_enth[:, :ncontrol2, :][:, np2:, :]  # injection rate
    # X2 = fz2_enth[:, :ncontrol2, :]  # control
    Y2 = fz2_enth[:, ncontrol2:, :]  # predict
    X2pn = (X2p - x2min) / (x2max - x2min)
    X2in = (X2i - x2mini) / (x2maxi - x2mini)
    X2n = np.concatenate((X2pn, X2in), axis=1)
    if y2min is None:
        y2min = Y2.min()
        y2max = Y2.max()
    Y2n = (Y2 - y2min) / (y2max - y2min)
    s2 = np.asarray([x2min, x2max, x2mini, x2maxi, 40, 80, y2min, y2max])

    # ========== fault zone m ==========
    ncontrolm = params['ncontrolm']
    Xm = fzm_enth[:, :ncontrolm, :]
    Ym = fzm_enth[:, ncontrolm:, :]
    Xmn = (Xm - xmmin) / (xmmax - xmmin)
    if ymmin is None:
        ymmin = Ym.min()
        ymmax = Ym.max()
    Ymn = (Ym - ymmin) / (ymmax - ymmin)
    sm = np.asarray([xmmin, xmmax, ymmin, ymmax])

    # ========== fault zone 3 ==========
    np3 = fz_pwell[3]
    ncontrol3 = params['ncontrol3']
    # X3 = fz3_enth[:, :ncontrol3, :]
    X3p = fz3_enth[:, :ncontrol3, :][:, :np3, :]
    X3i = fz3_enth[:, :ncontrol3, :][:, np3:, :]
    X3pn = (X3p - x3min) / (x3max - x3min)
    X3in = (X3i - x3mini) / (x3maxi - x3mini)
    X3n = np.concatenate((X3pn, X3in), axis=1)
    Y3 = fz3_enth[:, ncontrol3:, :]
    if y3min is None:
        y3min = Y3.min()
        y3max = Y3.max()
    Y3n = (Y3 - y3min) / (y3max - y3min)
    s3 = np.asarray([x3min, x3max, x3mini, x3maxi, 40, 80, y3min, y3max])

    # ========== 36A-15 ==========
    X_ = fz__enth
    X_n = (X_ - x_mini) / (x_maxi - x_mini)
    s_ = np.asarray([x_mini, x_maxi])

    # ========== combine scalers ==========
    s = [tf.constant(s1, dtype='float32'), tf.constant(s2, dtype='float32'), tf.constant(sm, dtype='float32'),
         tf.constant(s3, dtype='float32'), tf.constant(s_, dtype='float32')]
    #  ========== generate bounds ==========
    # Prod_name  =         ["21-28","78-20","88-19","77A-19","21-19","77-19"];
    # Fault zones:             3       m       2       2        1       2
    Prod_bounds = np.array([[s3[1], sm[1], s2[1], s2[1], s1[1], s2[1]],
                            [s3[0], sm[0], s2[0], s2[0], s1[0], s2[0]]])
    Enth_bounds = np.array([[s3[-1], sm[-1], s2[-1], s2[-1], s1[-1], s2[-1]],
                            [s3[-2], sm[-2], s2[-2], s2[-2], s1[-2], s2[-2]]])
    # Inj_name  =          ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
    # Fault zones:             3       2       2       2       2       3       -        2
    Injt_bounds = np.array([[s3[3], s2[3], s2[3], s2[3], s2[3], s3[3], s_[1], s2[3]],
                            [s3[2], s2[2], s2[2], s2[2], s2[2], s3[2], s_[0], s2[2]]])
    # ========== generate history ==========
    h1 = np.concatenate((X1n[:twindow, :, :1], Y1n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h2 = np.concatenate((X2n[:twindow, :, :1], Y2n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    hm = np.concatenate((Xmn[:twindow, :, :1], Ymn[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h3 = np.concatenate((X3n[:twindow, :, :1], Y3n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h_ = X_n[:twindow, :, :1].transpose(2, 0, 1)
    History = {"Fault 1": h1, "Fault 2": h2, "Fault m": hm, "Fault 3": h3, "36A-15": h_}
    Dataset = {"Fault 1": [X1n, Y1n], "Fault 2": [X2n, Y2n], "Fault m": [Xmn, Ymn], "Fault 3": [X3n, Y3n],
               "36A-15": [X_n]}
    ProxyBounds = {"Prod_bounds": Prod_bounds, "Injt_bounds": Injt_bounds, "Enth_bounds": Enth_bounds}
    return History, Dataset, s, nfeatures, ProxyBounds


def process_bhp(history_bhp, predict_bhp, params, twindow=6, pbound_sim=None, ibound_sim=None, PBHP_bounds=None,
                IBHP_bounds=None):
    if pbound_sim is None:
        # Prod_name  =         ["21-28", "78-20", "88-19","77A-19", "21-19", "77-19"];
        # Fault zones:             3m      3m       12       12        12       12
        pbound_sim = np.array([[324.00,  540.00,  492.00,  540.00,  300.00,  168.00],
                               [175.56,  290.40,  264.00,  290.40,  155.10,  82.50]])
    if ibound_sim is None:
        # Injt_name  =          ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        # Fault zones:             3m      12      12      12      12      3m      3m      12
        ibound_sim = np.array([[   560,    350,    520,    300,    740,    200,    200,    200],
                               [     0,      0,      0,      0,      0,      0,      0,      0]])
    if PBHP_bounds is None:
        y12min = None
        y12max = None
        y3mmin = None
        y3mmax = None
    else:
        # PBHP_bounds
        #      0        1        2        3        4        5
        # ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"],
        #      3m       3m       12       12        12       12
        y12min = PBHP_bounds[0, :][2]
        y12max = PBHP_bounds[1, :][2]
        y3mmin = PBHP_bounds[0, :][0]
        y3mmax = PBHP_bounds[1, :][0]
    if IBHP_bounds is None:
        y12mini = None
        y12maxi = None
        y3mmini = None
        y3mmaxi = None
    else:
        # IBHP_bounds
        #      0       1       2       3       4       5       6        7
        # ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        #      3m      12      12      12      12      3m      3m       12
        y12mini = IBHP_bounds[0, :][1]
        y12maxi = IBHP_bounds[1, :][1]
        y3mmini = IBHP_bounds[0, :][0]
        y3mmaxi = IBHP_bounds[1, :][0]

    jp = [0, 1]  # Fault Zone 3 and Middle Fault and 36A-15
    x3mmax = max(pbound_sim[0, jp])
    x3mmin = min(pbound_sim[1, jp])
    jp = [2, 3, 4, 5]  # Fault Zones 1 and 2
    x12max = max(pbound_sim[0, jp])
    x12min = min(pbound_sim[1, jp])
    ji = [0, 5, 6]  # Fault Zone 3 and Middle Fault and 36A-15
    x3mmaxi = max(ibound_sim[0, ji])
    x3mmini = min(ibound_sim[1, ji])
    ji = [1, 2, 3, 4, 7]  # Fault Zones 1 and 2
    x12maxi = max(ibound_sim[0, ji])
    x12mini = min(ibound_sim[1, ji])

    # ========== loading data ==========
    ncontrols = [params['ncontrol12'], params['ncontrol3m'], params['ncontrol12i'], params['ncontrol3mi']]
    nfeatures = [params['nfeature12'], params['nfeature3m'], params['nfeature12i'], params['nfeature3mi']]

    fz12pbhp = np.concatenate((history_bhp[0], predict_bhp[0]), axis=0)
    fz3mpbhp = np.concatenate((history_bhp[1], predict_bhp[1]), axis=0)
    fz12ibhp = np.concatenate((history_bhp[2], predict_bhp[2]), axis=0)
    fz3mibhp = np.concatenate((history_bhp[3], predict_bhp[3]), axis=0)

    # ========== production & injection: fault zone 1 and 2 ==========
    np12 = 4  # no. of production wells
    ncontrol12 = params['ncontrol12']
    ncontrol12i = params['ncontrol12i']
    X12p = fz12pbhp[:, :ncontrol12, :][:, :np12, :]  # production rate
    X12i = fz12pbhp[:, :ncontrol12, :][:, np12:, :]  # injection rate
    Y12p = fz12pbhp[:, ncontrol12:, :][:, :np12, :]  # production BHP
    Y12i = fz12ibhp[:, ncontrol12i:, :]  # injection BHP
    X12pn = (X12p - x12min) / (x12max - x12min)
    X12in = (X12i - x12mini) / (x12maxi - x12mini)
    X12n = np.concatenate((X12pn, X12in), axis=1)
    if y12min is None:
        y12min = Y12p.min()
        y12max = Y12p.max()
        y12mini = Y12i.min()
        y12maxi = Y12i.max()
    Y12pn = (Y12p - y12min) / (y12max - y12min)
    Y12in = (Y12i - y12mini) / (y12maxi - y12mini)
    s12 = np.asarray([x12min, x12max, x12mini, x12maxi, 40, 80, y12min, y12max, y12mini, y12maxi])

    # ========== fault zone 3m ==========
    np3m = 2  # no. of production wells
    ncontrol3m = params['ncontrol3m']
    ncontrol3mi = params['ncontrol3mi']
    X3p = fz3mpbhp[:, :ncontrol3m, :][:, :np3m, :]
    X3i = fz3mpbhp[:, :ncontrol3m, :][:, np3m:, :]
    X3pn = (X3p - x3mmin) / (x3mmax - x3mmin)
    X3in = (X3i - x3mmini) / (x3mmaxi - x3mmini)
    X3mn = np.concatenate((X3pn, X3in), axis=1)
    Y3p = fz3mpbhp[:, ncontrol3m:, :][:, :np3m, :]
    Y3i = fz3mibhp[:, ncontrol3mi:, :]#[:, np3m:, :]
    if y3mmin is None:
        y3mmin = Y3p.min()
        y3mmax = Y3p.max()
        y3mmini = Y3i.min()
        y3mmaxi = Y3i.max()
    Y3mpn = (Y3p - y3mmin) / (y3mmax - y3mmin)
    Y3min = (Y3i - y3mmini) / (y3mmaxi - y3mmini)
    s3m = np.asarray([x3mmin, x3mmax, x3mmini, x3mmaxi, 40, 80, y3mmin, y3mmax, y3mmini, y3mmaxi])

    # ========== combine scalers ==========
    scaler = [tf.constant(s12, dtype='float32'), tf.constant(s3m, dtype='float32')]

    #  ========== generate bounds ==========
    # Prod_name  =         ["21-28","78-20","88-19","77A-19","21-19","77-19"];
    # Fault zones:             3m       3m      12      12       12      12
    Prod_bounds = np.array([[x3mmax, x3mmax, x12max, x12max, x12max, x12max],
                            [x3mmin, x3mmin, x12min, x12min, x12min, x12min]])
    PBHP_bounds = np.array([[y3mmax, y3mmax, y12max, y12max, y12max, y12max],
                            [y3mmin, y3mmin, y12min, y12min, y12min, y12min]])
    # Inj_name  =          ["44-21", "23-17","36-17","37-17","85-19", "38-21","36A-15","2429I"];
    # Fault zones:              3m      12      12      12      12       3m      3m       12
    Injt_bounds = np.array([[x3mmaxi, x12maxi, x12maxi, x12maxi, x12maxi, x3mmaxi, x3mmaxi, x12maxi],
                            [x3mmini, x12mini, x12mini, x12mini, x12mini, x3mmini, x3mmini, x12mini]])
    IBHP_bounds = np.array([[y3mmaxi, y12maxi, y12maxi, y12maxi, y12maxi, y3mmaxi, y3mmaxi, y12maxi],
                            [y3mmini, y12mini, y12mini, y12mini, y12mini, y3mmini, y3mmini, y12mini]])

    # ========== generate dataset ==========
    h12p= np.concatenate((X12n[:twindow, :, :1], Y12pn[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h3mp= np.concatenate((X3mn[:twindow, :, :1], Y3mpn[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h12i= np.concatenate((X12n[:twindow, :, :1], Y12in[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h3mi= np.concatenate((X3mn[:twindow, :, :1], Y3min[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    History = {"Fault 12": h12p, "Fault 3m": h3mp,
               "Fault 12i": h12i, "Fault 3mi": h3mi}
    Dataset = {"Fault 12": [X12n, Y12pn], "Fault 3m": [X3mn, Y3mpn],
               "Fault 12i": [X12n, Y12in], "Fault 3mi": [X3mn, Y3min]}
    ProxyBounds = {"Prod_bounds": Prod_bounds, "Injt_bounds": Injt_bounds, "PBHP_bounds": PBHP_bounds, "IBHP_bounds": IBHP_bounds}
    return History, Dataset, scaler, nfeatures, ncontrols, ProxyBounds


def process_bhp2(history_bhp, predict_bhp, params, twindow=6, pbound_sim=None, ibound_sim=None, PBHP_bounds=None,
                IBHP_bounds=None):
    if pbound_sim is None:
        # Prod_name  =         ["21-28", "78-20", "88-19","77A-19", "21-19", "77-19"];
        # Fault zones:             3m      3m       12       12        12       12
        pbound_sim = np.array([[324.00,  540.00,  492.00,  540.00,  300.00,  168.00],
                               [175.56,  290.40,  264.00,  290.40,  155.10,  82.50]])
    if ibound_sim is None:
        # Injt_name  =          ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        # Fault zones:             3m      12      12      12      12      3m      3m      12
        ibound_sim = np.array([[   560,    350,    520,    300,    740,    200,    200,    200],
                               [     0,      0,      0,      0,      0,      0,      0,      0]])
    if PBHP_bounds is None:
        y12min = None
        y12max = None
        y3mmin = None
        y3mmax = None
    else:
        # PBHP_bounds
        #      0        1        2        3        4        5
        # ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"],
        #      3m       3m       12       12        12       12
        y12min = PBHP_bounds[0, :][2]
        y12max = PBHP_bounds[1, :][2]
        y3mmin = PBHP_bounds[0, :][0]
        y3mmax = PBHP_bounds[1, :][0]
    if IBHP_bounds is None:
        y12mini = None
        y12maxi = None
        y3mmini = None
        y3mmaxi = None
    else:
        # IBHP_bounds
        #      0       1       2       3       4       5       6        7
        # ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        #      3m      12      12      12      12      3m      3m       12
        y12mini = IBHP_bounds[0, :][1]
        y12maxi = IBHP_bounds[1, :][1]
        y3mmini = IBHP_bounds[0, :][0]
        y3mmaxi = IBHP_bounds[1, :][0]

    jp = [0, 1]  # Fault Zone 3 and Middle Fault and 36A-15
    x3mmax = max(pbound_sim[0, jp])
    x3mmin = min(pbound_sim[1, jp])
    jp = [2, 3, 4, 5]  # Fault Zones 1 and 2
    x12max = max(pbound_sim[0, jp])
    x12min = min(pbound_sim[1, jp])
    ji = [0, 5, 6]  # Fault Zone 3 and Middle Fault and 36A-15
    x3mmaxi = max(ibound_sim[0, ji])
    x3mmini = min(ibound_sim[1, ji])
    ji = [1, 2, 3, 4, 7]  # Fault Zones 1 and 2
    x12maxi = max(ibound_sim[0, ji])
    x12mini = min(ibound_sim[1, ji])

    # ========== loading data ==========
    ncontrols = [params['ncontrol12'], params['ncontrol3m']]
    nfeatures = [params['nfeature12'], params['nfeature3m']]
    fz_pwell = [1, 3, 2, 0]  # No. of production wells in each fault zone
    fz12_bhp = np.concatenate((history_bhp[0], predict_bhp[0]), axis=0)
    fz3m_bhp = np.concatenate((history_bhp[1], predict_bhp[1]), axis=0)

    # ========== fault zone 1 and 2 ==========
    np12 = 4  # no. of production wells
    ncontrol12 = params['ncontrol12']
    X12p = fz12_bhp[:, :ncontrol12, :][:, :np12, :]  # production rate
    X12i = fz12_bhp[:, :ncontrol12, :][:, np12:, :]  # injection rate
    Y12p = fz12_bhp[:, ncontrol12:, :][:, :np12, :]  # production BHP
    Y12i = fz12_bhp[:, ncontrol12:, :][:, np12:, :]  # injection BHP
    X12pn = (X12p - x12min) / (x12max - x12min)
    X12in = (X12i - x12mini) / (x12maxi - x12mini)
    X12n = np.concatenate((X12pn, X12in), axis=1)
    if y12min is None:
        y12min = Y12p.min()
        y12max = Y12p.max()
        y12mini = Y12i.min()
        y12maxi = Y12i.max()
    Y12pn = (Y12p - y12min) / (y12max - y12min)
    Y12in = (Y12i - y12mini) / (y12maxi - y12mini)
    Y12n = np.concatenate((Y12pn, Y12in), axis=1)
    s12 = np.asarray([x12min, x12max, x12mini, x12maxi, 40, 80, y12min, y12max, y12mini, y12maxi])

    # ========== fault zone 3m ==========
    np3m = 2  # no. of production wells
    ncontrol3m = params['ncontrol3m']
    X3p = fz3m_bhp[:, :ncontrol3m, :][:, :np3m, :]
    X3i = fz3m_bhp[:, :ncontrol3m, :][:, np3m:, :]
    X3pn = (X3p - x3mmin) / (x3mmax - x3mmin)
    X3in = (X3i - x3mmini) / (x3mmaxi - x3mmini)
    X3mn = np.concatenate((X3pn, X3in), axis=1)
    Y3p = fz3m_bhp[:, ncontrol3m:, :][:, :np3m, :]
    Y3i = fz3m_bhp[:, ncontrol3m:, :][:, np3m:, :]
    if y3mmin is None:
        y3mmin = Y3p.min()
        y3mmax = Y3p.max()
        y3mmini = Y3i.min()
        y3mmaxi = Y3i.max()
    Y3pn = (Y3p - y3mmin) / (y3mmax - y3mmin)
    Y3in = (Y3i - y3mmini) / (y3mmaxi - y3mmini)
    Y3mn = np.concatenate((Y3pn, Y3in), axis=1)
    s3m = np.asarray([x3mmin, x3mmax, x3mmini, x3mmaxi, 40, 80, y3mmin, y3mmax, y3mmini, y3mmaxi])

    # ========== combine scalers ==========
    scaler = [tf.constant(s12, dtype='float32'), tf.constant(s3m, dtype='float32')]
    #  ========== generate bounds ==========
    # Prod_name  =         ["21-28","78-20","88-19","77A-19","21-19","77-19"];
    # Fault zones:             3m       3m      12      12       12      12
    Prod_bounds = np.array([[x3mmax, x3mmax, x12max, x12max, x12max, x12max],
                            [x3mmin, x3mmin, x12min, x12min, x12min, x12min]])
    PBHP_bounds = np.array([[y3mmax, y3mmax, y12max, y12max, y12max, y12max],
                            [y3mmin, y3mmin, y12min, y12min, y12min, y12min]])
    # Inj_name  =          ["44-21", "23-17","36-17","37-17","85-19", "38-21","36A-15","2429I"];
    # Fault zones:              3m      12      12      12      12       3m      3m       12
    Injt_bounds = np.array([[x3mmaxi, x12maxi, x12maxi, x12maxi, x12maxi, x3mmaxi, x3mmaxi, x12maxi],
                            [x3mmini, x12mini, x12mini, x12mini, x12mini, x3mmini, x3mmini, x12mini]])
    IBHP_bounds = np.array([[y3mmaxi, y12maxi, y12maxi, y12maxi, y12maxi, y3mmaxi, y3mmaxi, y12maxi],
                            [y3mmini, y12mini, y12mini, y12mini, y12mini, y3mmini, y3mmini, y12mini]])
    # ========== generate history ==========
    h12 = np.concatenate((X12n[:twindow, :, :1], Y12n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h3m = np.concatenate((X3mn[:twindow, :, :1], Y3mn[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    History = {"Fault 12": h12, "Fault 3m": h3m}
    Dataset = {"Fault 12": [X12n, Y12n],"Fault 3m": [X3mn, Y3mn]}
    ProxyBounds = {"Prod_bounds": Prod_bounds, "Injt_bounds": Injt_bounds, "PBHP_bounds": PBHP_bounds,
                   "IBHP_bounds": IBHP_bounds}
    return History, Dataset, scaler, nfeatures, ncontrols, ProxyBounds


def process_bhp3(history_bhp, predict_bhp, params, twindow=6, pbound_sim=None, ibound_sim=None, PBHP_bounds=None,
                IBHP_bounds=None):
    if pbound_sim is None:
        # Prod_name  =        ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"];
        # Fault zones:             3        m        2        2         1        2
        pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                               [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
    if ibound_sim is None:
        # Inj_name  =       ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        # Fault zones:            3       2       2       2       2       3       -        2
        ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
    if PBHP_bounds is None:
        y1min = None
        y1max = None
        y2min = None
        y2max = None
        y3mmin = None
        y3mmax = None
    else:
        # PBHP_bounds
        #      0        1        2        3        4        5
        # ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"],
        #      3m       3m        2        2         1        2
        y1min = PBHP_bounds[0, :][4]
        y1max = PBHP_bounds[1, :][4]
        y2min = PBHP_bounds[0, :][2]
        y2max = PBHP_bounds[1, :][2]
        y3mmin = PBHP_bounds[0, :][0]
        y3mmax = PBHP_bounds[1, :][0]
    if IBHP_bounds is None:
        y2mini = None
        y2maxi = None
        y3mmini = None
        y3mmaxi = None
        y_mini = None
        y_maxi = None
    else:
        # IBHP_bounds
        #      0       1       2       3       4       5       6        7
        # ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        #      3m       2       2       2       2      3m      -        2
        y2mini = IBHP_bounds[0, :][1]
        y2maxi = IBHP_bounds[1, :][1]
        y3mmini = IBHP_bounds[0, :][0]
        y3mmaxi = IBHP_bounds[1, :][0]
        y_mini = IBHP_bounds[0, :][6]
        y_maxi = IBHP_bounds[1, :][6]

    jp = [0, 1]  # Fault Zone 3 and Middle Fault
    x3mmax = max(pbound_sim[0, jp])
    x3mmin = min(pbound_sim[1, jp])
    jp = [2, 3, 5]  # Fault Zone 2
    x2max = max(pbound_sim[0, jp])
    x2min = min(pbound_sim[1, jp])
    x1max = pbound_sim[0, 4]
    x1min = pbound_sim[1, 4]

    ji = [0, 5]  # Fault Zone 3 and Middle Fault
    x3mmaxi = max(ibound_sim[0, ji])
    x3mmini = min(ibound_sim[1, ji])
    ji = [1, 2, 3, 4, 7]  # Fault Zone 2
    x2maxi = max(ibound_sim[0, ji])
    x2mini = min(ibound_sim[1, ji])
    # 36A-15
    x_maxi = ibound_sim[0, 6]
    x_mini = ibound_sim[1, 6]

    # ========== loading data ==========
    ncontrols = [params['ncontrol1'], params['ncontrol2'], params['ncontrol3m'], params['ncontrol_']]
    nfeatures = [params['nfeature1'], params['nfeature2'], params['nfeature3m'], params['nfeature_']]
    fz_pwell = [1, 3, 2, 0]  # No. of production wells in each fault zone
    fz1_bhp = np.concatenate((history_bhp[0], predict_bhp[0]), axis=0)
    fz2_bhp = np.concatenate((history_bhp[1], predict_bhp[1]), axis=0)
    fz3m_bhp = np.concatenate((history_bhp[2], predict_bhp[2]), axis=0)
    fz__bhp = np.concatenate((history_bhp[3], predict_bhp[3]), axis=0)  # 36A-15
    # print(fz1_bhp.shape, fz2_bhp.shape, fzm_enth.shape, fz3_enth.shape, fz__enth.shape)

    # ========== fault zone 1 ==========
    ncontrol1 = params['ncontrol1']
    X1 = fz1_bhp[:, :ncontrol1, :]
    Y1 = fz1_bhp[:, ncontrol1:, :]
    X1n = (X1 - x1min) / (x1max - x1min)
    if y1min is None:
        y1min = Y1.min()
        y1max = Y1.max()
    Y1n = (Y1 - y1min) / (y1max - y1min)
    s1 = np.asarray([x1min, x1max, y1min, y1max])

    # ========== fault zone 2 ==========
    np2 = 3
    ncontrol2 = params['ncontrol2']
    X2p = fz2_bhp[:, :ncontrol2, :][:, :np2, :]  # production rate
    X2i = fz2_bhp[:, :ncontrol2, :][:, np2:, :]  # injection rate
    Y2p = fz2_bhp[:, ncontrol2:, :][:, :np2, :]  # production BHP
    Y2i = fz2_bhp[:, ncontrol2:, :][:, np2:, :]  # injection BHP
    X2pn = (X2p - x2min) / (x2max - x2min)
    X2in = (X2i - x2mini) / (x2maxi - x2mini)
    X2n = np.concatenate((X2pn, X2in), axis=1)
    if y2min is None:
        y2min = Y2p.min()
        y2max = Y2p.max()
        y2mini = Y2i.min()
        y2maxi = Y2i.max()
    Y2pn = (Y2p - y2min) / (y2max - y2min)
    Y2in = (Y2i - y2mini) / (y2maxi - y2mini)
    Y2n = np.concatenate((Y2pn, Y2in), axis=1)
    s2 = np.asarray([x2min, x2max, x2mini, x2maxi, 40, 80, y2min, y2max, y2mini, y2maxi])

    # ========== fault zone 3m ==========
    np3m = 2
    ncontrol3m = params['ncontrol3m']
    X3p = fz3m_bhp[:, :ncontrol3m, :][:, :np3m, :]
    X3i = fz3m_bhp[:, :ncontrol3m, :][:, np3m:, :]
    X3pn = (X3p - x3mmin) / (x3mmax - x3mmin)
    X3in = (X3i - x3mmini) / (x3mmaxi - x3mmini)
    X3n = np.concatenate((X3pn, X3in), axis=1)
    Y3p = fz3m_bhp[:, ncontrol3m:, :][:, :np3m, :]
    Y3i = fz3m_bhp[:, ncontrol3m:, :][:, np3m:, :]
    if y3mmin is None:
        y3mmin = Y3p.min()
        y3mmax = Y3p.max()
        y3mmini = Y3i.min()
        y3mmaxi = Y3i.max()
    Y3pn = (Y3p - y3mmin) / (y3mmax - y3mmin)
    Y3in = (Y3i - y3mmini) / (y3mmaxi - y3mmini)
    Y3n = np.concatenate((Y3pn, Y3in), axis=1)
    s3m = np.asarray([x3mmin, x3mmax, x3mmini, x3mmaxi, 40, 80, y3mmin, y3mmax, y3mmini, y3mmaxi])

    # ========== 36A-15 ==========
    ncontrol_ = params['ncontrol_']
    X_ = fz__bhp[:, :ncontrol_, :]
    Y_ = fz__bhp[:, ncontrol_:, :]
    X_n = (X_ - x_mini) / (x_maxi - x_mini)
    if y_mini is None:
        y_mini = Y_.min()
        y_maxi = Y_.max()
    Y_n = (Y_ - y_mini) / (y_maxi - y_mini)
    s_ = np.asarray([x_mini, x_maxi, 40, 80, y_mini, y_maxi])

    # ========== combine scalers ==========
    scaler = [tf.constant(s1, dtype='float32'), tf.constant(s2, dtype='float32'),
              tf.constant(s3m, dtype='float32'), tf.constant(s_, dtype='float32')]
    #  ========== generate bounds ==========
    # Prod_name  =         ["21-28","78-20","88-19","77A-19","21-19","77-19"];
    # Fault zones:             3m       3m       2       2        1       2
    Prod_bounds = np.array([[x3mmax, x3mmax, x2max, x2max, x1max, x2max],
                            [x3mmin, x3mmin, x2min, x2min, x1min, x2min]])
    PBHP_bounds = np.array([[y3mmax, y3mmax, y2max, y2max, y1max, y2max],
                            [y3mmin, y3mmin, y2min, y2min, y1min, y2min]])
    # Inj_name  =          ["44-21", "23-17","36-17","37-17","85-19", "38-21","36A-15","2429I"];
    # Fault zones:            3m       2       2       2       2       3m       -        2
    Injt_bounds = np.array([[x3mmaxi, x2maxi, x2maxi, x2maxi, x2maxi, x3mmaxi, x_maxi, x2maxi],
                            [x3mmini, x2mini, x2mini, x2mini, x2mini, x3mmini, x_mini, x2mini]])
    IBHP_bounds = np.array([[y3mmaxi, y2maxi, y2maxi, y2maxi, y2maxi, y3mmaxi, y_maxi, y2maxi],
                            [y3mmini, y2mini, y2mini, y2mini, y2mini, y3mmini, y_mini, y2mini]])
    # ========== generate history ==========
    h1 = np.concatenate((X1n[:twindow, :, :1], Y1n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h2 = np.concatenate((X2n[:twindow, :, :1], Y2n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h3m = np.concatenate((X3n[:twindow, :, :1], Y3n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h_ = X_n[:twindow, :, :1].transpose(2, 0, 1)
    History = {"Fault 1": h1, "Fault 2": h2, "Fault 3m": h3m, "36A-15": h_}
    Dataset = {"Fault 1": [X1n, Y1n], "Fault 2": [X2n, Y2n], "Fault 3m": [X3n, Y3n], "36A-15": [X_n, Y_n]}
    ProxyBounds = {"Prod_bounds": Prod_bounds, "Injt_bounds": Injt_bounds, "PBHP_bounds": PBHP_bounds,
                   "IBHP_bounds": IBHP_bounds}
    return History, Dataset, scaler, nfeatures, ncontrols, ProxyBounds


def process_enth2(history_enth, predict_enth, params, twindow=6, pbound_sim=None, ibound_sim=None):
    if pbound_sim is None:
        # Prod_name  =        ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"];
        # Fault zones:             3        m        2        2         1        2
        pbound_sim = np.array([[324.00, 540.00, 492.00, 540.00, 300.00, 168.00],
                               [175.56, 290.40, 264.00, 290.40, 155.10, 82.50]])
    if ibound_sim is None:
        # Inj_name  =        ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
        # Fault zones:            3       2       2       2       2       3       -        2
        ibound_sim = np.array([[560, 350, 520, 300, 740, 200, 200, 200],
                               [0, 0, 0, 0, 0, 0, 0, 0]])

    x3max = pbound_sim[0, 0]
    x3min = pbound_sim[1, 0]
    xmmax = pbound_sim[0, 1]
    xmmin = pbound_sim[1, 1]
    jp = [2, 3, 5]
    x2max = max(pbound_sim[0, jp])
    x2min = min(pbound_sim[1, jp])
    x1max = pbound_sim[0, 4]
    x1min = pbound_sim[1, 4]

    ji = [0, 5]
    x3maxi = max(ibound_sim[0, ji])
    x3mini = min(ibound_sim[1, ji])
    ji = [1, 2, 3, 4, 7]
    x2maxi = max(ibound_sim[0, ji])
    x2mini = min(ibound_sim[1, ji])

    x_maxi = ibound_sim[0, 6]
    x_mini = ibound_sim[1, 6]
    # ========== loading data ==========
    ncontrols = [params['ncontrol1'], params['ncontrol2'], params['ncontrolm'], params['ncontrol3']]
    nfeatures = [params['nfeature1'], params['nfeature2'], params['nfeaturem'], params['nfeature3']]
    fz_pwell = nfeatures  # No. of production wells in each fault zone
    fz1_enth = np.concatenate((history_enth[0], predict_enth[0]), axis=0)
    fz2_enth = np.concatenate((history_enth[1], predict_enth[1]), axis=0)
    fzm_enth = np.concatenate((history_enth[2], predict_enth[2]), axis=0)
    fz3_enth = np.concatenate((history_enth[3], predict_enth[3]), axis=0)
    fz__enth = np.concatenate((history_enth[4], predict_enth[4]), axis=0)  # 36A-15
    # print(fz1_enth.shape, fz2_enth.shape, fzm_enth.shape, fz3_enth.shape, fz__enth.shape)

    # ========== fault zone 1 ==========
    ncontrol1 = params['ncontrol1']
    X1 = fz1_enth[:, :ncontrol1, :]
    Y1 = fz1_enth[:, ncontrol1:, :]
    X1n = (X1 - x1min) / (x1max - x1min)
    Y1n = (Y1 - Y1.min()) / (Y1.max() - Y1.min())
    s1 = np.asarray([x1min, x1max, Y1.min(), Y1.max()])

    # ========== fault zone 2 ==========
    np2 = fz_pwell[1]
    ncontrol2 = params['ncontrol2']
    X2p = fz2_enth[:, :ncontrol2, :][:, :np2, :]  # production rate
    X2i = fz2_enth[:, :ncontrol2, :][:, np2:, :]  # injection rate
    # X2 = fz2_enth[:, :ncontrol2, :]  # control
    Y2 = fz2_enth[:, ncontrol2:, :]  # predict
    X2pn = (X2p - x2min) / (x2max - x2min)
    X2in = (X2i - x2mini) / (x2maxi - x2mini)
    X2n = np.concatenate((X2pn, X2in), axis=1)
    Y2n = (Y2 - Y2.min()) / (Y2.max() - Y2.min())
    s2 = np.asarray([x2min, x2max, x2mini, x2maxi, 40, 80, Y2.min(), Y2.max()])

    # ========== fault zone m ==========
    ncontrolm = params['ncontrolm']
    Xm = fzm_enth[:, :ncontrolm, :]
    Ym = fzm_enth[:, ncontrolm:, :]
    Xmn = (Xm - xmmin) / (xmmax - xmmin)
    Ymn = (Ym - Ym.min()) / (Ym.max() - Ym.min())
    sm = np.asarray([xmmin, xmmax, Ym.min(), Ym.max()])

    # ========== fault zone 3 ==========
    np3 = fz_pwell[3]
    ncontrol3 = params['ncontrol3']
    # X3 = fz3_enth[:, :ncontrol3, :]
    X3p = fz3_enth[:, :ncontrol3, :][:, :np3, :]
    X3i = fz3_enth[:, :ncontrol3, :][:, np3:, :]
    X3pn = (X3p - x3min) / (x3max - x3min)
    X3in = (X3i - x3mini) / (x3maxi - x3mini)
    X3n = np.concatenate((X3pn, X3in), axis=1)
    Y3 = fz3_enth[:, ncontrol3:, :]
    Y3n = (Y3 - Y3.min()) / (Y3.max() - Y3.min())
    s3 = np.asarray([x3min, x3max, x3mini, x3maxi, 40, 80, Y3.min(), Y3.max()])

    # ========== 36A-15 ==========
    X_ = fz__enth
    X_n = (X_ - x_mini) / (x_maxi - x_mini)
    s_ = np.asarray([x_mini, x_maxi])

    # ========== combine scalers ==========
    s = [tf.constant(s1, dtype='float32'), tf.constant(s2, dtype='float32'), tf.constant(sm, dtype='float32'),
         tf.constant(s3, dtype='float32'), tf.constant(s_, dtype='float32')]
    #  ========== generate bounds ==========
    # Prod_name  =         ["21-28","78-20","88-19","77A-19","21-19","77-19"];
    # Fault zones:             3       m       2       2        1       2
    Prod_bounds = np.array([[s3[1], sm[1], s2[1], s2[1], s1[1], s2[1]],
                            [s3[0], sm[0], s2[0], s2[0], s1[0], s2[0]]])
    Enth_bounds = np.array([[s3[-1], sm[-1], s2[-1], s2[-1], s1[-1], s2[-1]],
                            [s3[-2], sm[-2], s2[-2], s2[-2], s1[-2], s2[-2]]])
    # Inj_name  =          ["44-21","23-17","36-17","37-17","85-19","38-21","36A-15","2429I"];
    # Fault zones:             3       2       2       2       2       3       -        2
    Injt_bounds = np.array([[s3[3], s2[3], s2[3], s2[3], s2[3], s3[3], s_[1], s2[3]],
                            [s3[2], s2[2], s2[2], s2[2], s2[2], s3[2], s_[0], s2[2]]])
    # ========== generate history ==========
    h1 = np.concatenate((X1n[:twindow, :, :1], Y1n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h2 = np.concatenate((X2n[:twindow, :, :1], Y2n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    hm = np.concatenate((Xmn[:twindow, :, :1], Ymn[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h3 = np.concatenate((X3n[:twindow, :, :1], Y3n[:twindow, :, :1]), axis=1).transpose(2, 0, 1)
    h_ = X_n[:twindow, :, :1].transpose(2, 0, 1)
    History = {"Fault 1": h1, "Fault 2": h2, "Fault m": hm, "Fault 3": h3, "36A-15": h_}
    Dataset = {"Fault 1": [X1n, Y1n], "Fault 2": [X2n, Y2n], "Fault m": [Xmn, Ymn], "Fault 3": [X3n, Y3n],
               "36A-15": [X_n]}
    ProxyBounds = {"Prod_bounds": Prod_bounds, "Injt_bounds": Injt_bounds, "Enth_bounds": Enth_bounds}
    return History, Dataset, s, nfeatures, ProxyBounds


def process_data_fz1(ktrain, K, PATH, time0, nstep, predictwindow, totaln, twindow, twindow2,
                     frequency=1, labelall=None, folder=r'', ykeyword="predicts_Prod_Enth"):
    # Define scalers
    if labelall is None:
        labelall = []
    ymax = float('-inf')
    ymin = float('inf')
    xmax = float('-inf')
    xmin = float('inf')
    for i in range(1, K):
        y0 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, 4:5])
        ymin = min(np.min(y0), ymin)
        ymax = max(np.max(y0), ymax)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, 4:5])
        xmin = min(np.min(x0), xmin)
        xmax = max(np.max(x0), xmax)

    # Process data
    # t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        y = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, 4:5])
        y = (y - ymin) / (ymax - ymin)
        x = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                   time0:nstep, 4:5])
        x = (x - xmin) / (xmax - xmin)
        # x = np.concatenate((t, x), axis=1)
        if len(labelall) is not 0:
            label = labelall[:, i - 1:i]
            y = y * label
            x = x * label
            x = np.concatenate((label, x), axis=1)
        if i == ktrain[0]:
            history, control, yout, historyt1, controlt1, youtt1, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
        else:
            historynew, controlnew, youtnew, historytnew, controltnew, youttnew, traintesthistorynew, traintestcontrolnew, obsnew, initialwindownew, xinnew, yinnew = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
            historyt1 = np.concatenate((historyt1, historytnew), axis=0)
            controlt1 = np.concatenate((controlt1, controltnew), axis=0)
            youtt1 = np.concatenate((youtt1, youttnew), axis=0)
    nfeature = youtt1.shape[2]
    ncontrol = controlt1.shape[2]
    # Write diary into log.txt
    Diary = open(join(folder, r"FZ1_Log{}.txt".format(ktrain.shape[0])), "w+")
    Diary.write('Size of training datasets: {}'.format(ktrain.shape[0]))
    Diary.write('\nKtrain: {}'.format(ktrain))
    Diary.write('\nK: {}'.format(K))
    Diary.write('\nPATH: {}'.format(PATH))
    Diary.write(
        '\ntime0: {}, \nnstep: {}, \npredictwindow: {}, \ntotaln: {}, \nfrequency: {}, \ntwindow: {}, \ntwindow2: {}'.format(
            time0, nstep, predictwindow, totaln, frequency, twindow, twindow2))
    Diary.write('\nxmin: {}, xmax: {}, \nymin: {}, ymax: {}'.format(xmin, xmax, ymin, ymax))
    Diary.write('\nShape of historyt1: {}'.format(historyt1.shape))
    Diary.write('\nShape of controlt1: {}'.format(controlt1.shape))
    Diary.write('\nShape of youtt1: {}'.format(youtt1.shape))
    Diary.close()
    return historyt1, controlt1, youtt1, nfeature, ncontrol, [xmin, xmax, ymin, ymax]


def process_data_fz2(ktrain, K, PATH, time0, nstep, predictwindow, totaln, twindow, twindow2,
                     frequency=1, labelall=None, folder=r'', ykeyword="predicts_Prod_Enth"):
    # Define scalers
    if labelall is None:
        labelall = []
    ymax = float('-inf')
    ymin = float('inf')
    xmax1 = float('-inf')
    xmin1 = float('inf')
    xmax2 = float('-inf')
    xmin2 = float('inf')
    xmax3 = float('-inf')
    xmin3 = float('inf')
    jp = [2, 3, 5]
    ji = [1, 2, 3, 4, 7]
    for i in range(1, K):
        # production well
        y0 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, jp])
        ymin = min(np.min(y0), ymin)
        ymax = max(np.max(y0), ymax)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, jp])
        xmin1 = min(np.min(x0), xmin1)
        xmax1 = max(np.max(x0), xmax1)
        # injection well
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                    time0:nstep, ji])
        xmin2 = min(np.min(x0), xmin2)
        xmax2 = max(np.max(x0), xmax2)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                    time0:nstep, ji])
        xmin3 = min(np.min(x0), xmin3)
        xmax3 = max(np.max(x0), xmax3)
    # Process data
    t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        y = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, jp])
        y = (y - ymin) / (ymax - ymin)
        x1 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, jp])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        x2 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                    time0:nstep, ji])
        x2 = (x2 - xmin2) / (xmax2 - xmin2)
        x3 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                    time0:nstep, ji])
        x3 = (x3 - xmin3) / (xmax3 - xmin3)
        x = np.concatenate((x1, x2, x3), axis=1)
        # x = np.concatenate((t, x1, x2), axis=1)
        if len(labelall) is not 0:
            label = labelall[:, i - 1]
            y = y * label[:, None]
            x = x * label[:, None]
            label = np.expand_dims(label, axis=1)
            x = np.concatenate((label, x), axis=1)
        if i == ktrain[0]:
            history, control, yout, historyt1, controlt1, youtt1, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
        else:
            historynew, controlnew, youtnew, historytnew, controltnew, youttnew, traintesthistorynew, traintestcontrolnew, obsnew, initialwindownew, xinnew, yinnew = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
            historyt1 = np.concatenate((historyt1, historytnew), axis=0)
            controlt1 = np.concatenate((controlt1, controltnew), axis=0)
            youtt1 = np.concatenate((youtt1, youttnew), axis=0)
    nfeature = youtt1.shape[2]
    ncontrol = controlt1.shape[2]
    # Write diary into log.txt
    Diary = open(join(folder, r"FZ2_Log{}.txt".format(ktrain.shape[0])), "w+")
    Diary.write('Size of training datasets: {}'.format(ktrain.shape[0]))
    Diary.write('\nKtrain: {}'.format(ktrain))
    Diary.write('\nK: {}'.format(K))
    Diary.write('\nPATH: {}'.format(PATH))
    Diary.write(
        '\ntime0: {}, \nnstep: {}, \npredictwindow: {}, \ntotaln: {}, \nfrequency: {}, \ntwindow: {}, \ntwindow2: {}'.format(
            time0, nstep, predictwindow, totaln, frequency, twindow, twindow2))
    Diary.write(
        '\nxmin1: {}, xmax1: {}, \nxmin2: {}, xmax2: {}, \nymin: {}, ymax: {}'.format(xmin1, xmax1, xmin2, xmax2, ymin,
                                                                                      ymax))
    Diary.write('\nShape of historyt1: {}'.format(historyt1.shape))
    Diary.write('\nShape of controlt1: {}'.format(controlt1.shape))
    Diary.write('\nShape of youtt1: {}'.format(youtt1.shape))
    Diary.close()
    return historyt1, controlt1, youtt1, nfeature, ncontrol, [xmin1, xmax1, xmin2, xmax2, xmin3, xmax3, ymin, ymax]


def process_data_fzm(ktrain, K, PATH, time0, nstep, predictwindow, totaln, twindow, twindow2,
                     frequency=1, labelall=None, folder=r'', ykeyword="predicts_Prod_Enth"):
    if labelall is None:
        labelall = []
    # Define scalers
    ymax = float('-inf')
    ymin = float('inf')
    xmax = float('-inf')
    xmin = float('inf')
    for i in range(1, K):
        y0 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, 1:2])
        ymin = min(np.min(y0), ymin)
        ymax = max(np.max(y0), ymax)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, 1:2])
        xmin = min(np.min(x0), xmin)
        xmax = max(np.max(x0), xmax)
    # Process data
    # t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        y = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, 1:2])
        y = (y - ymin) / (ymax - ymin)
        x = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                   time0:nstep, 1:2])
        x = (x - xmin) / (xmax - xmin)
        if len(labelall) is not 0:
            label = labelall[:, i - 1:i]
            y = y * label
            x = x * label
            x = np.concatenate((label, x), axis=1)  # x = np.concatenate((t, x), axis=1)
        if i == ktrain[0]:
            history, control, yout, historyt1, controlt1, youtt1, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
        else:
            historynew, controlnew, youtnew, historytnew, controltnew, youttnew, traintesthistorynew, traintestcontrolnew, obsnew, initialwindownew, xinnew, yinnew = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
            historyt1 = np.concatenate((historyt1, historytnew), axis=0)
            controlt1 = np.concatenate((controlt1, controltnew), axis=0)
            youtt1 = np.concatenate((youtt1, youttnew), axis=0)
    nfeature = youtt1.shape[2]
    ncontrol = controlt1.shape[2]
    # Write diary into log.txt
    Diary = open(join(folder, r"FZm_Log{}.txt".format(ktrain.shape[0])), "w+")
    Diary.write('Size of training datasets: {}'.format(ktrain.shape[0]))
    Diary.write('\nKtrain: {}'.format(ktrain))
    Diary.write('\nK: {}'.format(K))
    Diary.write('\nPATH: {}'.format(PATH))
    Diary.write(
        '\ntime0: {}, \nnstep: {}, \npredictwindow: {}, \ntotaln: {}, \nfrequency: {}, \ntwindow: {}, \ntwindow2: {}'.format(
            time0, nstep, predictwindow, totaln, frequency, twindow, twindow2))
    Diary.write('\nxmin: {}, xmax: {}, \nymin: {}, ymax: {}'.format(xmin, xmax, ymin, ymax))
    Diary.write('\nShape of historyt1: {}'.format(historyt1.shape))
    Diary.write('\nShape of controlt1: {}'.format(controlt1.shape))
    Diary.write('\nShape of youtt1: {}'.format(youtt1.shape))
    Diary.close()
    return historyt1, controlt1, youtt1, nfeature, ncontrol, [xmin, xmax, ymin, ymax]


def process_data_fz3(ktrain, K, PATH, time0, nstep, predictwindow, totaln, twindow, twindow2,
                     frequency=1, labelall=None, folder=r'', ykeyword="predicts_Prod_Enth"):
    # Define scalers
    ymax = float('-inf')
    ymin = float('inf')
    xmax1 = float('-inf')
    xmin1 = float('inf')
    xmax2 = float('-inf')
    xmin2 = float('inf')
    xmax3 = float('-inf')
    xmin3 = float('inf')
    j = [0, 5]
    for i in range(1, K):
        y0 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, 0:1])
        ymin = min(np.min(y0), ymin)
        ymax = max(np.max(y0), ymax)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, 0:1])
        xmin1 = min(np.min(x0), xmin1)
        xmax1 = max(np.max(x0), xmax1)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                    time0:nstep, j])
        xmin2 = min(np.min(x0), xmin2)
        xmax2 = max(np.max(x0), xmax2)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                    time0:nstep, j])
        xmin3 = min(np.min(x0), xmin3)
        xmax3 = max(np.max(x0), xmax3)
    # Process data
    # t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        y = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))[ykeyword])[time0:nstep, 0:1])
        y = (y - ymin) / (ymax - ymin)
        x1 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, 0:1])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        x2 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                    time0:nstep, j])
        x2 = (x2 - xmin2) / (xmax2 - xmin2)
        x3 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                    time0:nstep, j])
        x3 = (x3 - xmin3) / (xmax3 - xmin3)
        x = np.concatenate((x1, x2, x3), axis=1)
        if len(labelall) is not 0:
            label = labelall[:, i - 1]
            y = y * label[:, None]
            x = x * label[:, None]
            label = np.expand_dims(label, axis=1)
            x = np.concatenate((label, x), axis=1)  # x = np.concatenate((t, x), axis=1)
        if i == ktrain[0]:
            history, control, yout, historyt1, controlt1, youtt1, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
        else:
            historynew, controlnew, youtnew, historytnew, controltnew, youttnew, traintesthistorynew, traintestcontrolnew, obsnew, initialwindownew, xinnew, yinnew = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
            historyt1 = np.concatenate((historyt1, historytnew), axis=0)
            controlt1 = np.concatenate((controlt1, controltnew), axis=0)
            youtt1 = np.concatenate((youtt1, youttnew), axis=0)
    nfeature = youtt1.shape[2]
    ncontrol = controlt1.shape[2]
    # Write diary into log.txt
    Diary = open(join(folder, r"FZ3_Log{}.txt".format(ktrain.shape[0])), "w+")
    Diary.write('Size of training datasets: {}'.format(ktrain.shape[0]))
    Diary.write('\nKtrain: {}'.format(ktrain))
    Diary.write('\nK: {}'.format(K))
    Diary.write('\nPATH: {}'.format(PATH))
    Diary.write(
        '\ntime0: {}, \nnstep: {}, \npredictwindow: {}, \ntotaln: {}, \nfrequency: {}, \ntwindow: {}, \ntwindow2: {}'.format(
            time0, nstep, predictwindow, totaln, frequency, twindow, twindow2))
    Diary.write(
        '\nxmin1: {}, xmax1: {}, \nxmin2: {}, xmax2: {}, \nymin: {}, ymax: {}'.format(xmin1, xmax1, xmin2, xmax2, ymin,
                                                                                      ymax))
    Diary.write('\nShape of historyt1: {}'.format(historyt1.shape))
    Diary.write('\nShape of controlt1: {}'.format(controlt1.shape))
    Diary.write('\nShape of youtt1: {}'.format(youtt1.shape))
    Diary.close()
    return historyt1, controlt1, youtt1, nfeature, ncontrol, [xmin1, xmax1, xmin2, xmax2, xmin3, xmax3, ymin, ymax]


def process_data(fz_str, ktrain, K, PATH, time0, nstep, predictwindow, totaln, twindow, twindow2,
                 frequency=1, labelall=None, folder=r'', ykeyword="Enthalpy"):
    # Define well index
    jp_ind = {'fz1': [4],
              'fz2': [2, 3, 5],
              'fzm': [1],
              'fz3': [0]}
    ji_ind = {'fz1': [],
              'fz2': [1, 2, 3, 4, 7],
              'fzm': [],
              'fz3': [0, 5]}
    jp = jp_ind[fz_str]
    ji = ji_ind[fz_str]
    # Define scalers
    if labelall is None:
        labelall = []
    scaler_xy = {'prod_rate': [], 'injt_rate': [], 'injt_temp': [],
                 'prod_enth': [], 'prod_bhp': [], 'injt_bhp': []}
    ymax1 = float('-inf')  # prod_enth
    ymin1 = float('inf')
    ymax2 = float('-inf')  # prod_bhp
    ymin2 = float('inf')
    ymax3 = float('-inf')  # injt_bhp
    ymin3 = float('inf')
    xmax1 = float('-inf')  # prod_rate
    xmin1 = float('inf')
    xmax2 = float('-inf')  # injt_rate
    xmin2 = float('inf')
    xmax3 = float('-inf')  # injt_temp
    xmin3 = float('inf')
    for i in range(1, K):
        # predict
        if ykeyword is "Enthalpy":
            y0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))
                                 ["predicts_Prod_Enth"])[time0:nstep, jp])
            ymin1 = min(np.min(y0), ymin1)
            ymax1 = max(np.max(y0), ymax1)
        else:
            y0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))
                                 ["predicts_Prod_BHP"])[time0:nstep, jp])
            ymin2 = min(np.min(y0), ymin2)
            ymax2 = max(np.max(y0), ymax2)
            y0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))
                                 ["predicts_Inj_BHP"])[time0:nstep, ji])
            ymin3 = min(np.min(y0), ymin3)
            ymax3 = max(np.max(y0), ymax3)
        # control
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, jp])
        xmin1 = min(np.min(x0), xmin1)
        xmax1 = max(np.max(x0), xmax1)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                    time0:nstep, ji])
        xmin2 = min(np.min(x0), xmin2)
        xmax2 = max(np.max(x0), xmax2)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                    time0:nstep, ji])
        xmin3 = min(np.min(x0), xmin3)
        xmax3 = max(np.max(x0), xmax3)
    # Process data
    t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        if ykeyword is "Enthalpy":
            y = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))
                                ["predicts_Prod_Enth"])[time0:nstep, jp])
            y = (y - ymin1) / (ymax1 - ymin1)
        else:
            y2 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))
                                 ["predicts_Prod_BHP"])[time0:nstep, jp])
            y2 = (y2 - ymin2) / (ymax2 - ymin2)
            y3 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))
                                 ["predicts_Inj_BHP"])[time0:nstep, ji])
            y3 = (y3 - ymin3) / (ymax3 - ymin3)
            y = np.concatenate((y2, y3), axis=1)
        x1 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, jp])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        x2 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                    time0:nstep, ji])
        x2 = (x2 - xmin2) / (xmax2 - xmin2)
        x3 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                    time0:nstep, ji])
        x3 = (x3 - xmin3) / (xmax3 - xmin3)
        x = np.concatenate((x1, x2, x3), axis=1)
        if len(labelall) is not 0:
            label = labelall[:, i - 1]
            y = y * label[:, None]
            x = x * label[:, None]
            label = np.expand_dims(label, axis=1)
            x = np.concatenate((label, x), axis=1)
        if i == ktrain[0]:
            history, control, yout, historyt1, controlt1, youtt1, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
        else:
            historynew, controlnew, youtnew, historytnew, controltnew, youttnew, traintesthistorynew, traintestcontrolnew, obsnew, initialwindownew, xinnew, yinnew = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
            historyt1 = np.concatenate((historyt1, historytnew), axis=0)
            controlt1 = np.concatenate((controlt1, controltnew), axis=0)
            youtt1 = np.concatenate((youtt1, youttnew), axis=0)
    nfeature = youtt1.shape[2]
    ncontrol = controlt1.shape[2]
    scaler_xy['prod_enth'] = [ymin1, ymax1]
    scaler_xy['prod_bhp'] = [ymin2, ymax2]
    scaler_xy['injt_bhp'] = [ymin3, ymax3]
    scaler_xy['prod_rate'] = [xmin1, xmax1]
    scaler_xy['injt_rate'] = [xmin2, xmax2]
    scaler_xy['injt_temp'] = [xmin3, xmax3]
    # Write diary into log.txt
    Diary = open(join(folder, r"FZ2_Log{}.txt".format(ktrain.shape[0])), "w+")
    Diary.write('Size of training datasets: {}'.format(ktrain.shape[0]))
    Diary.write('\nKtrain: {}'.format(ktrain))
    Diary.write('\nK: {}'.format(K))
    Diary.write('\nPATH: {}'.format(PATH))
    Diary.write(
        '\ntime0: {}, \nnstep: {}, \npredictwindow: {}, \ntotaln: {}, \nfrequency: {}, \ntwindow: {}, \ntwindow2: {}'.format(
            time0, nstep, predictwindow, totaln, frequency, twindow, twindow2))
    Diary.write(
        '\nxmin1: {}, xmax1: {}, \nxmin2: {}, xmax2: {}, \nymin: {}, ymax: {}'.format(xmin1, xmax1, xmin2, xmax2, ymin1,
                                                                                      ymax1))
    Diary.write('\nShape of historyt1: {}'.format(historyt1.shape))
    Diary.write('\nShape of controlt1: {}'.format(controlt1.shape))
    Diary.write('\nShape of youtt1: {}'.format(youtt1.shape))
    Diary.close()
    return historyt1, controlt1, youtt1, nfeature, ncontrol, scaler_xy


def data_processer2(fz_str, ktrain, K, PATH, time0, nstep, totaln, twindow, labelall=None, ykeyword="Enthalpy"):
    History = []
    Control = []
    Feature = []
    # Define well index
    jp_ind = {'FZ1': [4],
              'FZ2': [2, 3, 5],
              'FZm': [1],
              'FZ3': [0]}
    ji_ind = {'FZ1': [],
              'FZ2': [1, 2, 3, 4, 7],
              'FZm': [],
              'FZ3': [0, 5]}
    jp = jp_ind[fz_str]
    ji = ji_ind[fz_str]

    # Define scalers
    if labelall is None:
        labelall = []
    scaler_xy = {'prod_rate': [], 'injt_rate': [], 'injt_temp': [], 'prod_enth': [], 'prod_bhp': [], 'injt_bhp': []}
    ymax1 = float('-inf')  # prod_enth
    ymin1 = float('inf')
    ymax2 = float('-inf')  # prod_bhp
    ymin2 = float('inf')
    ymax3 = float('-inf')  # injt_bhp
    ymin3 = float('inf')
    xmax1 = float('-inf')  # prod_rate
    xmin1 = float('inf')
    xmax2 = float('-inf')  # injt_rate
    xmin2 = float('inf')
    xmax3 = float('-inf')  # injt_temp
    xmin3 = float('inf')
    for i in range(1, K):
        # predict
        if ykeyword is "Enthalpy":
            # print('Enthalpy in data_processer2')
            y0 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_Enth"])[
                time0:nstep, jp])
            ymin1 = min(np.min(y0), ymin1)
            ymax1 = max(np.max(y0), ymax1)
        else:
            # print('BHP in data_processer2')
            y0 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_BHP"])[
                time0:nstep, jp])
            ymin2 = min(np.min(y0), ymin2)
            ymax2 = max(np.max(y0), ymax2)
            if len(ji) is not 0:
                y0 = np.abs(
                    np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Inj_BHP"])[
                    time0:nstep, ji])
                ymin3 = min(np.min(y0), ymin3)
                ymax3 = max(np.max(y0), ymax3)
        # control
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, jp])
        xmin1 = min(np.min(x0), xmin1)
        xmax1 = max(np.max(x0), xmax1)
        if len(ji) is not 0:
            x0 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                time0:nstep, ji])
            xmin2 = min(np.min(x0), xmin2)
            xmax2 = max(np.max(x0), xmax2)
            x0 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                time0:nstep, ji])
            xmin3 = min(np.min(x0), xmin3)
            xmax3 = max(np.max(x0), xmax3)

    # Process data
    t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        if ykeyword is "Enthalpy":
            y = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_Enth"])[
                time0:nstep, jp])
            y = (y - ymin1) / (ymax1 - ymin1)
        else:
            y2 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_BHP"])[
                time0:nstep, jp])
            y2 = (y2 - ymin2) / (ymax2 - ymin2)
            # y3 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Inj_BHP"])[time0:nstep, ji])
            # y3 = (y3 - ymin3) / (ymax3 - ymin3)
            # y = np.concatenate((y2, y3), axis=1)
            y = y2
        x1 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, jp])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        if len(ji) is not 0:
            x2 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                time0:nstep, ji])
            x2 = (x2 - xmin2) / (xmax2 - xmin2)
            x3 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                time0:nstep, ji])
            x3 = (x3 - xmin3) / (xmax3 - xmin3)
            x = np.concatenate((x1, x2, x3), axis=1)
        else:
            x = x1
        if len(labelall) is not 0:
            label = labelall[:, i - 1]
            y = y * label[:, None]
            x = x * label[:, None]
            label = np.expand_dims(label, axis=1)
            x = np.concatenate((label, x), axis=1)
        # print(x.shape, y.shape)
        ncontrol = x.shape[1]
        nfeature = y.shape[1]
        control = np.reshape(x, (1, -1, ncontrol))[:, twindow:, :]
        feature = np.reshape(y, (1, -1, nfeature))[:, twindow:, :]
        history = np.concatenate((control, feature), axis=2)[:, :twindow, :]
        Feature.append(feature)
        History.append(history)
        Control.append(control)
    Features = np.squeeze(np.array(Feature), axis=1)
    Historys = np.squeeze(np.array(History), axis=1)
    Controls = np.squeeze(np.array(Control), axis=1)
    nfeature = Features.shape[2]
    ncontrol = Controls.shape[2]
    scaler_xy['prod_enth'] = [ymin1, ymax1]
    scaler_xy['prod_bhp'] = [ymin2, ymax2]
    scaler_xy['injt_bhp'] = [ymin3, ymax3]
    scaler_xy['prod_rate'] = [xmin1, xmax1]
    scaler_xy['injt_rate'] = [xmin2, xmax2]
    scaler_xy['injt_temp'] = [xmin3, xmax3]
    return Historys, Controls, Features, nfeature, ncontrol, scaler_xy


def data_processer3(fz_str, ktrain, PATH, time0, nstep, totaln, twindow, scaler_xy,
                    twindow2, predictwindow, frequency=1, labelall=None, ykeyword="Enthalpy"):
    History = []
    Control = []
    Feature = []
    # Define well index
    jp_ind = {'FZ1': [4],
              'FZ2': [2, 3, 5],
              'FZm': [1],
              'FZ3': [0]}
    ji_ind = {'FZ1': [],
              'FZ2': [1, 2, 3, 4, 7],
              'FZm': [],
              'FZ3': [0, 5]}
    jp = jp_ind[fz_str]
    ji = ji_ind[fz_str]

    # Define scalers
    if labelall is None:
        labelall = []

    ymin1 = scaler_xy['prod_enth'][0]  # predicts_Prod_Enth
    ymax1 = scaler_xy['prod_enth'][1]
    ymin2 = scaler_xy['prod_bhp'][0]  # predicts_Prod_BHP
    ymax2 = scaler_xy['prod_bhp'][1]
    ymin3 = scaler_xy['injt_bhp'][0]  # predicts_Inj_BHP
    ymax3 = scaler_xy['injt_bhp'][1]
    # control
    xmin1 = scaler_xy['prod_rate'][0]  # controls_Prod_Rate
    xmax1 = scaler_xy['prod_rate'][1]
    xmin2 = scaler_xy['injt_rate'][0]  # controls_Inj_Rate
    xmax2 = scaler_xy['injt_rate'][1]
    xmin3 = scaler_xy['injt_temp'][0]  # controls_Inj_Temp
    xmax3 = scaler_xy['injt_temp'][1]

    # Process data
    t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        if ykeyword is "Enthalpy":
            y = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_Enth"])[
                time0:nstep, jp])
            y = (y - ymin1) / (ymax1 - ymin1)
        else:
            y2 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_BHP"])[
                time0:nstep, jp])
            y2 = (y2 - ymin2) / (ymax2 - ymin2)
            # y3 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Inj_BHP"])[time0:nstep, ji])
            # y3 = (y3 - ymin3) / (ymax3 - ymin3)
            # y = np.concatenate((y2, y3), axis=1)
            y = y2
        x1 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep, jp])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        if len(ji) is not 0:
            x2 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                time0:nstep, ji])
            x2 = (x2 - xmin2) / (xmax2 - xmin2)
            x3 = np.abs(
                np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                time0:nstep, ji])
            x3 = (x3 - xmin3) / (xmax3 - xmin3)
            x = np.concatenate((x1, x2, x3), axis=1)
        else:
            x = x1
        if len(labelall) is not 0:
            label = labelall[:, i - 1]
            y = y * label[:, None]
            x = x * label[:, None]
            label = np.expand_dims(label, axis=1)
            x = np.concatenate((label, x), axis=1)

        if i == ktrain[0]:
            history, control, yout, historyt1, controlt1, youtt1, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
        else:
            historynew, controlnew, youtnew, historytnew, controltnew, youttnew, traintesthistorynew, traintestcontrolnew, obsnew, initialwindownew, xinnew, yinnew = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
            historyt1 = np.concatenate((historyt1, historytnew), axis=0)
            controlt1 = np.concatenate((controlt1, controltnew), axis=0)
            youtt1 = np.concatenate((youtt1, youttnew), axis=0)
    nfeature = youtt1.shape[2]
    ncontrol = controlt1.shape[2]
    return historyt1, controlt1, youtt1, nfeature, ncontrol


def predict_data_fz1(model_long, nfeature, ncontrol, labelall, ktest, PATH, twindow, time0, nstep, ykeyword, scaler_xy):
    xmin = scaler_xy[0]
    xmax = scaler_xy[1]
    ymin = scaler_xy[2]
    ymax = scaler_xy[3]
    Feature = []
    History = []
    Control = []
    Predict = []
    for i in ktest:
        y = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Predict.mat"))[ykeyword])[time0:nstep, 4:5])
        y = (y - ymin) / (ymax - ymin)
        x = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Prod_Rate"])[time0:nstep,
            4:5])
        x = (x - xmin) / (xmax - xmin)
        label = labelall[:, i - 1:i]
        y = y * label
        x = x * label
        x = np.concatenate((label, x), axis=1)
        control = np.reshape(x, (1, -1, ncontrol))[:, twindow:, :]
        feature = np.reshape(y, (1, -1, nfeature))[:, twindow:, :]
        history = np.concatenate((control, feature), axis=2)[:, :twindow, :]
        is_training = np.ones((1, 1))
        state = np.zeros((1, nfeature))
        predict = model_long.predict([history, control, state, is_training, control[:, :, :nfeature]])[0]
        # plotpredictionlong(feature, predict, frequency, 1, 1)
        Feature.append(feature)
        History.append(history)
        Control.append(control)
        Predict.append(predict)
    Features = np.squeeze(np.array(Feature), axis=1)
    Historys = np.squeeze(np.array(History), axis=1)
    Controls = np.squeeze(np.array(Control), axis=1)
    Predicts = np.squeeze(np.array(Predict), axis=1)
    return Predicts, Features, Controls, Historys


def predict_data_fz2(model_long, nfeature, ncontrol, labelall, ktest, PATH, twindow, time0, nstep, ykeyword, scaler_xy):
    xmin1 = scaler_xy[0]
    xmax1 = scaler_xy[1]
    xmin2 = scaler_xy[2]
    xmax2 = scaler_xy[3]
    xmin3 = scaler_xy[4]
    xmax3 = scaler_xy[5]
    ymin = scaler_xy[-2]
    ymax = scaler_xy[-1]
    Feature = []
    History = []
    Control = []
    Predict = []
    jp = [2, 3, 5]
    ji = [1, 2, 3, 4, 7]
    for i in ktest:
        y = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Predict.mat"))[ykeyword])[time0:nstep, jp])
        y = (y - ymin) / (ymax - ymin)
        x1 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Prod_Rate"])[time0:nstep,
            jp])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        x2 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Inj_Rate"])[time0:nstep,
            ji])
        x2 = (x2 - xmin2) / (xmax2 - xmin2)
        x3 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Inj_Temp"])[time0:nstep,
            ji])
        x3 = (x3 - xmin3) / (xmax3 - xmin3)
        x = np.concatenate((x1, x2, x3), axis=1)
        label = labelall[:, i - 1:i]
        y = y * label
        x = x * label
        x = np.concatenate((label, x), axis=1)
        control = np.reshape(x, (1, -1, ncontrol))[:, twindow:, :]
        feature = np.reshape(y, (1, -1, nfeature))[:, twindow:, :]
        history = np.concatenate((control, feature), axis=2)[:, :twindow, :]
        is_training = np.ones((1, 1))
        state = np.zeros((1, nfeature))
        predict = model_long.predict([history, control, state, is_training, control[:, :, :nfeature]])[0]
        Feature.append(feature)
        History.append(history)
        Control.append(control)
        Predict.append(predict)
    Features = np.squeeze(np.array(Feature), axis=1)
    Historys = np.squeeze(np.array(History), axis=1)
    Controls = np.squeeze(np.array(Control), axis=1)
    Predicts = np.squeeze(np.array(Predict), axis=1)
    return Predicts, Features, Controls, Historys


def predict_data_fzm(model_long, nfeature, ncontrol, labelall, ktest, PATH, twindow, time0, nstep, ykeyword, scaler_xy):
    xmin = scaler_xy[0]
    xmax = scaler_xy[1]
    ymin = scaler_xy[2]
    ymax = scaler_xy[3]
    Feature = []
    History = []
    Control = []
    Predict = []
    for i in ktest:
        y = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Predict.mat"))[ykeyword])[time0:nstep, 1:2])
        y = (y - ymin) / (ymax - ymin)
        x = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Prod_Rate"])[time0:nstep,
            1:2])
        x = (x - xmin) / (xmax - xmin)
        label = labelall[:, i - 1:i]
        y = y * label
        x = x * label
        x = np.concatenate((label, x), axis=1)
        control = np.reshape(x, (1, -1, ncontrol))[:, twindow:, :]
        feature = np.reshape(y, (1, -1, nfeature))[:, twindow:, :]
        history = np.concatenate((control, feature), axis=2)[:, :twindow, :]
        is_training = np.ones((1, 1))
        state = np.zeros((1, nfeature))
        predict = model_long.predict([history, control, state, is_training, control[:, :, :nfeature]])[0]
        # plotpredictionlong(feature, predict, frequency, 1, 1)
        Feature.append(feature)
        History.append(history)
        Control.append(control)
        Predict.append(predict)
    Features = np.squeeze(np.array(Feature), axis=1)
    Historys = np.squeeze(np.array(History), axis=1)
    Controls = np.squeeze(np.array(Control), axis=1)
    Predicts = np.squeeze(np.array(Predict), axis=1)
    return Predicts, Features, Controls, Historys


def predict_data_fz3(model_long, nfeature, ncontrol, labelall, ktest, PATH, twindow, time0, nstep, ykeyword, scaler_xy):
    xmin1 = scaler_xy[0]
    xmax1 = scaler_xy[1]
    xmin2 = scaler_xy[2]
    xmax2 = scaler_xy[3]
    xmin3 = scaler_xy[4]
    xmax3 = scaler_xy[5]
    ymin = scaler_xy[-2]
    ymax = scaler_xy[-1]
    Feature = []
    History = []
    Control = []
    Predict = []
    jp = [0]
    ji = [0, 5]
    for i in ktest:
        y = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Predict.mat"))[ykeyword])[time0:nstep, jp])
        y = (y - ymin) / (ymax - ymin)
        x1 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Prod_Rate"])[time0:nstep,
            jp])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        x2 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Inj_Rate"])[time0:nstep,
            ji])
        x2 = (x2 - xmin2) / (xmax2 - xmin2)
        x3 = np.abs(
            np.array(sio.loadmat(join(PATH, "Dataset{}".format(i), "Control.mat"))["controls_Inj_Temp"])[time0:nstep,
            ji])
        x3 = (x3 - xmin3) / (xmax3 - xmin3)
        x = np.concatenate((x1, x2, x3), axis=1)
        label = labelall[:, i - 1:i]
        y = y * label
        x = x * label
        x = np.concatenate((label, x), axis=1)
        # print(x.shape, y.shape)
        control = np.reshape(x, (1, -1, ncontrol))[:, twindow:, :]
        feature = np.reshape(y, (1, -1, nfeature))[:, twindow:, :]
        history = np.concatenate((control, feature), axis=2)[:, :twindow, :]
        is_training = np.ones((1, 1))
        state = np.zeros((1, nfeature))
        predict = model_long.predict([history, control, state, is_training, control[:, :, :nfeature]])[0]
        # plotpredictionlong(feature, predict, frequency, 1, 1)
        Feature.append(feature)
        History.append(history)
        Control.append(control)
        Predict.append(predict)
    Features = np.squeeze(np.array(Feature), axis=1)
    Historys = np.squeeze(np.array(History), axis=1)
    Controls = np.squeeze(np.array(Control), axis=1)
    Predicts = np.squeeze(np.array(Predict), axis=1)
    return Predicts, Features, Controls, Historys


def predict_full(RMSE, fault, N, fz_str, fz_str1, prop, ktest, K, PATH, time0, nstep, totaln, twindow, labelall,
                 ykeyword='Enthalpy', prefix='enth_'):
    # RMSE = {'N25': [], 'N50': [], 'N100': [], 'N200': [], 'N400': []}
    if ykeyword.upper().find('ENTH') is not -1:  # enthalpy
        # print('Enthalpy in predict full')
        history, control, feature, nfeature, ncontrol, scaler_xy = data_processer2(
            fz_str, ktest, K, PATH, time0, nstep, totaln, twindow, labelall)
    elif ykeyword.upper().find('BHP') is not -1:  # bhp
        # print('BHP in predict full')
        history, control, feature, nfeature, ncontrol, scaler_xy = data_processer2(
            fz_str, ktest, K, PATH, time0, nstep, totaln, twindow, labelall, ykeyword=ykeyword)
    else:
        raise ValueError('No such keyword!')
    is_training = False * np.ones(history.shape[0])
    state = np.zeros((history.shape[0], nfeature))
    label = np.repeat(control[:, :, :1], nfeature, axis=2)
    Predict = []
    for ni in range(len(N)):
        rmse1 = []
        for i in range(5):
            folder = 'FZ{}_{}'.format(fz_str1, N[ni])
            model = prefix + 'N{}_no{}'.format(N[ni], i + 1)
            path0 = join(prop, fault, folder)
            print('Property: {}, \nKeyword: {}, \npath: {}'.format(prop, ykeyword, join(path0, 'saved_model', model)))
            model_long = tf.keras.models.load_model(join(path0, 'saved_model', model))
            # predict on validation dataset
            predict = model_long.predict([history, control, state, is_training, label])[0]
            predict[np.isnan(predict)] = 0
            predict[np.isinf(predict)] = 0
            print(predict.shape, feature.shape)
            rmse = mean_squared_error(predict[:, :, 0].T, feature[:, :, 0].T, multioutput='raw_values', squared=False)
            rmse1.append(rmse)
            Predict.append(predict)
        RMSE['N{}'.format(N[ni])] = np.array(rmse1)  # shape: (5, 50)
    return RMSE, Predict, history, control, feature, nfeature, ncontrol, scaler_xy


def predict_full_test(RMSE, RMSE_test, fault, N, fz_str, fz_str1, prop, ktest, K, PATH,
                      time0, nstep, totaln, twindow, labelall, ykeyword, prefix='enth_'):
    if ykeyword.upper().find('ENTH') is not -1:  # enthalpy
        history, control, feature, nfeature, ncontrol, scaler_xy = data_processer2(
            fz_str, ktest, K, PATH, time0, nstep, totaln, twindow, labelall)
    elif ykeyword.upper().find('BHP') is not -1:  # bhp
        history, control, feature, nfeature, ncontrol, scaler_xy = data_processer2(
            fz_str, ktest, K, PATH, time0, nstep, totaln, twindow, labelall, ykeyword=ykeyword)
    else:
        raise ValueError('No such keyword!')
    is_training = False * np.ones(history.shape[0])
    state = np.zeros((history.shape[0], nfeature))
    label = np.repeat(control[:, :, :1], nfeature, axis=2)
    Predict = []
    for ni in range(len(N)):
        rmse1 = []
        i = np.argmin(RMSE['N{}'.format(N[ni])].mean(axis=1))
        folder = 'FZ{}_{}'.format(fz_str1, N[ni])
        model = prefix + 'N{}_no{}'.format(N[ni], i + 1)
        path0 = join(prop, fault, folder)
        print('Property: {}, \nKeyword: {}, \npath: {}'.format(prop, ykeyword, join(path0, 'saved_model', model)))
        model_long = tf.keras.models.load_model(join(path0, 'saved_model', model))
        # predict on validation dataset
        predict = model_long.predict([history, control, state, is_training, label])[0]
        predict[np.isnan(predict)] = 0
        predict[np.isinf(predict)] = 0
        print(predict.shape, feature.shape)
        rmse = mean_squared_error(predict[:, :, 0].T, feature[:, :, 0].T, multioutput='raw_values', squared=False)
        rmse1.append(rmse)
        Predict.append(predict)
        RMSE_test['N{}'.format(N[ni])] = np.array(rmse1)  # shape: (5, 50)
    return RMSE_test, Predict, history, control, feature


# process data without temperature or label
def process_data_fz2_a(ktrain, K, PATH, time0, nstep, predictwindow, totaln, frequency, twindow, twindow2):
    # Define scalers
    ymax = float('-inf')
    ymin = float('inf')
    xmax1 = float('-inf')
    xmin1 = float('inf')
    xmax2 = float('-inf')
    xmin2 = float('inf')
    xmax3 = float('-inf')
    xmin3 = float('inf')
    for i in range(1, K):
        j = [2, 3, 5]
        y0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_Enth"])[
                    time0:nstep - 1, j])
        ymin = min(np.min(y0), ymin)
        ymax = max(np.max(y0), ymax)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])[
                    time0:nstep - 1, j])
        xmin1 = min(np.min(x0), xmin1)
        xmax1 = max(np.max(x0), xmax1)
        j = [1, 2, 3, 4, 7]
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])[
                    time0:nstep - 1, j])
        xmin2 = min(np.min(x0), xmin2)
        xmax2 = max(np.max(x0), xmax2)
        x0 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Temp"])[
                    time0:nstep - 1, j])
        xmin3 = min(np.min(x0), xmin3)
        xmax3 = max(np.max(x0), xmax3)

    # Process data
    t = np.expand_dims(np.asarray(range(totaln - 1)), axis=1) / totaln
    for i in ktrain:
        y = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Predict.mat"))["predicts_Prod_Enth"])
                   [time0:nstep - 1, [2, 3, 5]])
        y = (y - ymin) / (ymax - ymin)
        x1 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Prod_Rate"])
                    [time0:nstep - 1, [2, 3, 5]])
        x1 = (x1 - xmin1) / (xmax1 - xmin1)
        x2 = np.abs(np.array(sio.loadmat(join(PATH, "Dataset{}".format(str(i)), "Control.mat"))["controls_Inj_Rate"])
                    [time0:nstep - 1, [1, 2, 3, 4, 7]])
        x2 = (x2 - xmin2) / (xmax2 - xmin2)
        x = np.concatenate((t, x1, x2), axis=1)
        if i == ktrain[0]:
            history, control, yout, historyt1, controlt1, youtt1, traintesthistory, traintestcontrol, obs, initialwindow, xin, yin = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
        else:
            historynew, controlnew, youtnew, historytnew, controltnew, youttnew, traintesthistorynew, traintestcontrolnew, obsnew, initialwindownew, xinnew, yinnew = GenerateSets(
                x, y, frequency, twindow, twindow2, predictwindow)
            historyt1 = np.concatenate((historyt1, historytnew), axis=0)
            controlt1 = np.concatenate((controlt1, controltnew), axis=0)
            youtt1 = np.concatenate((youtt1, youttnew), axis=0)

    # Write diary into log.txt
    Diary = open(r"FZ2_Log{}.txt".format(ktrain.shape[0]), "w+")
    Diary.write('Size of training datasets: {}'.format(ktrain.shape[0]))
    Diary.write('\nKtrain: {}'.format(ktrain))
    Diary.write('\nK: {}'.format(K))
    Diary.write('\nPATH: {}'.format(PATH))
    Diary.write(
        '\ntime0: {}, \nnstep: {}, \npredictwindow: {}, \ntotaln: {}, \nfrequency: {}, \ntwindow: {}, \ntwindow2: {}'.format(
            time0, nstep, predictwindow, totaln, frequency, twindow, twindow2))
    Diary.write(
        '\nxmin1: {}, xmax1: {}, \nxmin2: {}, xmax2: {}, \nymin: {}, ymax: {}'.format(xmin1, xmax1, xmin2, xmax2, ymin,
                                                                                      ymax))
    Diary.write('\nShape of historyt1: {}'.format(historyt1.shape))
    Diary.write('\nShape of controlt1: {}'.format(controlt1.shape))
    Diary.write('\nShape of youtt1: {}'.format(youtt1.shape))
    Diary.close()

    return historyt1, controlt1, youtt1, [xmin1, xmax1, xmin2, xmax2, ymin, ymax]
