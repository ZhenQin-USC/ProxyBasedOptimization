from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd



# plotting function
def plot1(x, true, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        # ax1 = axesf[i]
        if i < true.shape[1]:
            # ax1.set_ylim([0,1.05])
            ax1.scatter(x, true[:, i], color='black', s=1)
        else:
            ax1.axis('off')


def plot2(true, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        # ax1 = axesf[i]
        if i < true.shape[1]:
            # ax1.set_ylim([0,1.05])
            ax1.plot(true[:, i], color='black')
        else:
            ax1.axis('off')

def plot2a(true, nrows, ncols, ylim):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        # ax1 = axesf[i]
        if i < true.shape[1]:
            ax1.set_ylim(ylim)
            ax1.plot(true[:, i], color='black')
        else:
            ax1.axis('off')


def plot3(true, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < true.shape[1]:
            ax1.set_ylim([0, 1.05])
            ax1.plot(true[:, i], color='black')
        else:
            ax1.axis('off')


def ploth(Clean_X, wellname, nrows, ncols, Ktrain, Ktest, alpha=0.7):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=True, sharey=True)
    ax = np.array(axes)
    kwargs = dict(histtype='stepfilled', alpha=alpha, bins=40)
    for i, ax1 in enumerate(ax.flat):
        if i < len(wellname):
            ax1.hist(Clean_X[i, Ktrain], **kwargs)
            ax1.hist(Clean_X[i, Ktest], **kwargs)
            ax1.set_xlim([0, 1])
            if i >= len(wellname)- ncols:
                ax1.set_xlabel('Normalized rate')
            if i%ncols == 0:
                ax1.set_ylabel('Count')
            ax1.set_title(wellname[i])
            ax1.legend(['Training', 'Testing'])
        else:
            ax1.axis('off')


def plotpredictionb(true, train0, predict0, nstep, frequency, nrows, ncols):
    # print(np.round(np.sqrt(np.mean((true-predict0)**2)),3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    plt.ylim(0, 1)
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < predict0.shape[1]:
            ax1.axvline(x=nstep - 0.5, color='grey', linestyle='--')
            # ax1.plot(true[:,i],color='black', linestyle = '-',marker = None,markersize = 5,label='OBS '+title[i])
            ax1.scatter(np.arange(0, true.shape[0], frequency), true[:, i], color='black', marker='o', s=1, label='True')
            ax1.scatter(np.arange(0, nstep, frequency), train0[:int(nstep / frequency), i], color='orange', marker='o', s=1, label='Short-term')
            ax1.scatter(np.arange(nstep, nstep + predict0.shape[0] * frequency, frequency), predict0[:, i], color='green', marker='o', s=1, label='Long-term')
            # err = np.round(np.sqrt(np.mean((true[:,i]-predict0[:,i])**2)),3)
            ax1.set_ylim([0, 1])
            # ax1.set_title('RMSE = ' + str(err))
            # ax1.set_title(title[i])
            ax1.legend(loc=0, fontsize=8)
        else:
            ax1.axis('off')


def plotprediction3b0(true, predict0, nstep, frequency, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    # plt.ylim(0.8,1)
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < predict0.shape[1]:
            ax1.axvline(x=nstep - 0.5, color='grey', linestyle='--')
            # ax1.plot(true[:,i],color='black', linestyle = '-',marker = None,markersize = 5,label='OBS '+title[i])
            ax1.scatter(np.arange(0, true.shape[0], frequency), true[:, i], color='black', marker='o', s=1, label='True')
            ax1.scatter(np.arange(0, nstep, frequency), predict0[:int(nstep / frequency), i], color='green', marker='o', s=1, label='Training')
            ax1.scatter(np.arange(nstep, predict0.shape[0] * frequency, frequency), predict0[int(nstep / frequency):, i], color='green', marker='o', s=1, label='Prediction')
            # ax1.set_title(title[i])
            # ax1.set_title('RMSE = ' + str(err))
            ax1.legend(loc=0, fontsize=8)
            ax1.set_ylim([0, 1])
        else:
            ax1.axis('off')


def plotprediction3b(true, predict0, nstep, frequency, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    # plt.ylim(0.8,1)
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < predict0.shape[1]:
            # ax1.plot(true[:,i],color='black', linestyle = '-',marker = None,markersize = 5,label='OBS '+title[i])
            ax1.scatter(np.arange(0, true.shape[0], frequency), true[:, i], color='black', marker='o', s=1, label='True')
            ax1.scatter(np.arange(0, predict0.shape[0] * frequency, frequency), predict0[:, i], color='green', marker='o', s=1, label='Prediction')
            # ax1.set_title(title[i])
            # ax1.set_title('RMSE = ' + str(err))
            ax1.legend(loc=0, fontsize=8)
            ax1.set_ylim([0, 1])
        else:
            ax1.axis('off')


def plotpredictionlong(true, predict0, frequency, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    # plt.ylim(0.8,1)
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < predict0.shape[1]:
            # ax1.plot(true[:,i],color='black', linestyle = '-',marker = None,markersize = 5,label='OBS '+title[i])
            ax1.scatter(np.arange(0, true.shape[0], frequency), true[:, i], color='black', marker='o', s=1,
                        label='True')
            ax1.scatter(np.arange(0, predict0.shape[1] * frequency, frequency),
                        predict0[0, :, i], color='green', marker='o', s=1, label='Prediction')
            # ax1.set_title(title[i])
            # ax1.set_title('RMSE = ' + str(err))
            ax1.legend(loc=0, fontsize=8)
            ax1.set_ylim([0, 1])
        else:
            ax1.axis('off')


def plotpredictionlong2(true, predict0, frequency, trainwindow, nrows, ncols, ylim=[0, 1]):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    # plt.ylim(0.8,1)
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < predict0.shape[-1]:
            # ax1.plot(true[:,i],color='black', linestyle = '-',marker = None,markersize = 5,label='OBS '+title[i])
            ax1.scatter(np.arange(0, true.shape[0], frequency), true[:, i], color='black', marker='o', s=1,
                        label='True')
            ax1.scatter(np.arange(0, predict0.shape[1] * frequency, frequency),
                        predict0[0, :, i], color='green', marker='o', s=1, label='Prediction')
            ax1.axvline(x=trainwindow, color='grey', linestyle='--')
            # ax1.set_title(title[i])
            # ax1.set_title('RMSE = ' + str(err))
            ax1.legend(loc=0, fontsize=8)
            ax1.set_ylim(ylim)
        else:
            ax1.axis('off')


def plotpredictionlong2a(true, predict0, frequency, trainwindow, nrows, ncols, ylim=None):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    # plt.ylim(0.8,1)
    ax = np.array(axes)
    for i, ax1 in enumerate(ax.flat):
        if i < predict0.shape[-1]:
            # ax1.plot(true[:,i],color='black', linestyle = '-',marker = None,markersize = 5,label='OBS '+title[i])
            ax1.scatter(np.arange(0, true.shape[0], frequency), true[:, i], color='black', marker='o', s=1,
                        label='True')
            ax1.scatter(np.arange(0, predict0.shape[1] * frequency, frequency),
                        predict0[0, :, i], color='green', marker='o', s=1, label='Prediction')
            ax1.axvline(x=trainwindow, color='grey', linestyle='--')
            # ax1.set_title(title[i])
            # ax1.set_title('RMSE = ' + str(err))
            ax1.legend(loc=0, fontsize=8)
            if ylim != None:
                ax1.set_ylim(ylim)
        else:
            ax1.axis('off')

            
def plotmonitor(DATE, historyEnth, ENTH_sim, mask, Prod_name=None, NIter=-1, ylim=None, yticks=None):
    if Prod_name is None:
        Prod_name = ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"]

    wellind = ['enth3', 'enthm', 'enth2a', 'enth2b', 'enth1', 'enth2c']

    ind, _ = np.where(mask == 1)
    Colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    markersize = 4
    plt.figure(figsize=(10, 4), dpi=500)
    for i in range(len(wellind)):
        h = plt.plot(np.asarray(DATE)[ind], ENTH_sim[:, i, NIter],
                     label='{} Simulation'.format(Prod_name[i]),
                     linestyle='', marker='o', markersize=markersize, color=Colors[i],
                     markerfacecolor='black', markeredgewidth=1.5)
        h = plt.plot(DATE, historyEnth[wellind[i]][NIter][0],
                     label='{} Proxy'.format(Prod_name[i]),
                     linestyle='--', marker='o', markersize=markersize, color=Colors[i],
                     markerfacecolor = 'white', markeredgewidth=1.5)
    plt.xlabel('Date')

    plt.legend(bbox_to_anchor=(1.01, 0.95), ncol=1)  # plt.legend(bbox_to_anchor=(1., 1), loc='best', ncol=1)
    if ylim != None:
        plt.ylim(ylim)
        if yticks != None:
            plt.yticks(np.linspace(ylim[0], ylim[1], len(yticks)), yticks)
            if yticks[-1] == 1:
                plt.ylabel('Normalized enthalpy')
            elif yticks[-1] > 1:
                plt.ylabel('Specific enthalpy')
    plt.tight_layout()


def plotmonitor2(DATE, historyEnth, ENTH_sim, mask, Prod_name=None, ylim=None, yticks=None, wellind=None):
    if Prod_name is None:
        Prod_name = ["21-28", "78-20", "88-19", "77A-19", "21-19", "77-19"]
    if wellind is None:
        wellind = ['enth3', 'enthm', 'enth2a', 'enth2b', 'enth1', 'enth2c']

    ind, _ = np.where(mask == 1)
    Colors = ["royalblue", "orangered", "orange", "rebeccapurple", "limegreen", "deepskyblue", '#e377c2', '#7f7f7f', '#bcbd22']
#     Colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markersize = 4
    plt.figure(figsize=(9, 4), dpi=500)
    for i in range(len(wellind)):
        h = plt.plot(np.asarray(DATE)[ind], ENTH_sim[:, i],
                     label='{} Simulation'.format(Prod_name[i]),
                     linestyle='', marker='o', markersize=markersize, color='black',
                     markerfacecolor=Colors[i], markeredgewidth=.5)
        h = plt.plot(DATE, historyEnth[wellind[i]],
                     label='{} Proxy'.format(Prod_name[i]),
                     linestyle='--', marker='o', markersize=markersize, color=Colors[i],
                     markerfacecolor = 'white', markeredgewidth=.8)
    plt.xlabel('Date')

    plt.legend(bbox_to_anchor=(1.01, 0.95), ncol=1)  # plt.legend(bbox_to_anchor=(1., 1), loc='best', ncol=1)
    if ylim != None:
        plt.ylim(ylim)
        if yticks != None:
            plt.yticks(np.linspace(ylim[0], ylim[1], len(yticks)), yticks)
            if yticks[-1] == 1:
                plt.ylabel('Normalized Specific enthalpy')
            elif yticks[-1] > 1:
                plt.ylabel('Specific enthalpy')
    plt.tight_layout()


def plot_pca(feature, xiter, samples=None):
    xmean = np.expand_dims(np.mean(feature, axis=0), axis=0)
    if samples != None:
        xsample_c = samples - xmean
    feature_c= feature - xmean
    xiter_c  = xiter - xmean
    n_components = 3
    pca = PCA(n_components)
    pca.fit(feature_c)
    v_pca = pd.DataFrame(pca.transform(feature_c), columns=['PCA%i' % i for i in range(n_components)])
    v_pca0 = pd.DataFrame(pca.transform(xiter_c), columns=['PCA%i' % i for i in range(n_components)])
    if samples != None:
        v_pca1= pd.DataFrame(pca.transform(xsample_c), columns=['PCA%i' % i for i in range(n_components)])
    plt.figure(figsize=(10, 3.2), tight_layout=True, dpi=500)
    plt.subplot(1, 2, 1)
    plt.scatter(v_pca['PCA0'][:], v_pca['PCA1'][:], s=30, alpha=0.6) # training set
    if samples != None:
        plt.scatter(v_pca1['PCA0'][:], v_pca1['PCA1'][:], s=30, alpha=0.6, c = 'g') #  additional sample
    for i in range(len(v_pca0)): # iteration
        alpha = max(0.1,(i+1)/len(v_pca0))
        if i == len(v_pca0)-1:
            plt.scatter(v_pca0['PCA0'][i], v_pca0['PCA1'][i], s=80, c = 'k', alpha=alpha, marker = 'd')
        else:
            plt.scatter(v_pca0['PCA0'][i], v_pca0['PCA1'][i], s=30, c = 'r', alpha=alpha)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title('Projections')
    #plt.yticks([-1,-0.5,0,0.5,1])
    #plt.xticks([-1,-0.5,0,0.5,1])

    plt.subplot(1, 2, 2)
    plt.scatter(v_pca['PCA1'][:], v_pca['PCA2'][:], s=30, alpha=0.6, label='Training dataset') # c=cs,
    if samples != None:
        plt.scatter(v_pca1['PCA1'][:], v_pca1['PCA2'][:], s=30, alpha=0.6, label='Additional sample', c = 'g') #  c=cs,
    for i in range(len(v_pca0)):
        alpha = max(0.1,(i+1)/len(v_pca0))
        if i == len(v_pca0)-1:
            plt.scatter(v_pca0['PCA1'][i], v_pca0['PCA2'][i], s=80, c = 'k', alpha=alpha, marker = 'd', label='Optimal control')
        elif i == len(v_pca0)-2:
            plt.scatter(v_pca0['PCA1'][i], v_pca0['PCA2'][i], s=30, c = 'r', alpha=alpha, label='Optimization iteration')
        else:
            plt.scatter(v_pca0['PCA1'][i], v_pca0['PCA2'][i], s=30, c = 'r', alpha=alpha)
    plt.legend(bbox_to_anchor=(0.5, 0., 1.4, 0.5), loc='right', borderaxespad=0.)
    plt.xlabel('$z_2$')
    plt.ylabel('$z_3$')
    plt.title('Projections')
    #plt.yticks([-1,-0.5,0,0.5,1])
    #plt.xticks([-1,-0.5,0,0.5,1])


def plot_pca1(feature, xiter, split, samples=None):
    xmean = np.expand_dims(np.mean(feature, axis=0), axis=0)
    feature_c = feature - xmean
    xiter_c = xiter - xmean
    n_components = 3
    pca = PCA(n_components)
    pca.fit(feature_c)
    #print(pca.transform(feature_c).shape)
    v_pca = pd.DataFrame(pca.transform(feature_c)[:split,:], columns=['PCA%i' % i for i in range(n_components)])
    v_pca_ = pd.DataFrame(pca.transform(feature_c)[split:,:], columns=['PCA%i' % i for i in range(n_components)])
    v_pca0 = pd.DataFrame(pca.transform(xiter_c), columns=['PCA%i' % i for i in range(n_components)])
    #print(v_pca)
    # plot
    plt.figure(figsize=(10, 3.2), tight_layout=True, dpi=500)
    plt.subplot(1, 2, 1)
    plt.scatter(v_pca_['PCA0'][:], v_pca_['PCA1'][:], s=30, alpha=0.5, c = 'gray') # training set
    plt.scatter(v_pca['PCA0'][:], v_pca['PCA1'][:], s=30, alpha=0.6) # training set
    if samples != None:
        xsample_c = samples - xmean
        v_pca1= pd.DataFrame(pca.transform(xsample_c), columns=['PCA%i' % i for i in range(n_components)])
        plt.scatter(v_pca1['PCA0'][:], v_pca1['PCA1'][:], s=30, alpha=0.6, c = 'g') #  additional sample
    for i in range(len(v_pca0)): # iteration
        alpha = max(0.1,(i+1)/len(v_pca0))
        if i == len(v_pca0)-1:
            plt.scatter(v_pca0['PCA0'][i], v_pca0['PCA1'][i], s=80, c = 'k', alpha=alpha, marker = 'd')
        else:
            plt.scatter(v_pca0['PCA0'][i], v_pca0['PCA1'][i], s=30, c = 'r', alpha=alpha)
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title('Projections')
    #plt.yticks([-1,-0.5,0,0.5,1])
    #plt.xticks([-1,-0.5,0,0.5,1])

    plt.subplot(1, 2, 2)
    plt.scatter(v_pca_['PCA1'][:], v_pca_['PCA2'][:], s=30, alpha=0.5, c = 'gray', label='Background')  # testing set
    plt.scatter(v_pca['PCA1'][:], v_pca['PCA2'][:], s=30, alpha=0.6, label='Training dataset') # c=cs,
    if samples != None:
        plt.scatter(v_pca1['PCA1'][:], v_pca1['PCA2'][:], s=30, alpha=0.6, label='Additional sample', c = 'g') #  c=cs,
    for i in range(len(v_pca0)):
        alpha = max(0.1,(i+1)/len(v_pca0))
        if i == len(v_pca0)-1:
            plt.scatter(v_pca0['PCA1'][i], v_pca0['PCA2'][i], s=80, c = 'k', alpha=alpha, marker = 'd', label='Optimal control')
        elif i == len(v_pca0)-2:
            plt.scatter(v_pca0['PCA1'][i], v_pca0['PCA2'][i], s=30, c = 'r', alpha=alpha, label='Optimization iteration')
        else:
            plt.scatter(v_pca0['PCA1'][i], v_pca0['PCA2'][i], s=30, c = 'r', alpha=alpha)
    plt.legend(bbox_to_anchor=(0.5, 0., 1.4, 0.5), loc='right', borderaxespad=0.)
    plt.xlabel('$z_2$')
    plt.ylabel('$z_3$')
    plt.title('Projections')
    #plt.yticks([-1,-0.5,0,0.5,1])
    #plt.xticks([-1,-0.5,0,0.5,1])