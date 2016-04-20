### datasciencetools.py
# this is a module developed by Matthew Oberhardt to analyze
# and visualize data.

import pandas as pd
import json
import pickle
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
from numpy import nan
import seaborn as sns
import datetime
from scipy.stats import ttest_ind
from scipy.stats import ranksums
from contextlib import contextmanager
import scipy as sp
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

# sklearn imports:
from sklearn import linear_model
import sklearn
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.tree
import sklearn.ensemble
import numpy as np
from sklearn.utils.validation import check_consistent_length, _num_samples
import sklearn.preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.grid_search import GridSearchCV
import sys
from contextlib import contextmanager


# Plot styles:
sns.set(style="darkgrid", color_codes=True, font_scale=1.5)
#sns.set(style="white", color_codes=True, font_scale=1.5)

#############################
## Visualization functions ##
#############################


def heatmap_with_separated_colorbar(mat, rownames, colnames, xlabel=[], ylabel=[], figsize=(10,10)):
    print 'not working yet'
#    '''
#    Builds an imshow of any matrix, with label names
#    mat = np.ndarray (2d matrix)
#    rownames = array of row names
#    colnames = array of column names
#
#    instructions on getting the colorbar to the right size from:
#    http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
#    '''
#
#    aspect = 20#figsize[0]/figsize[1]
#    pad_fraction = 20
#
##    ax = plt.gca()
##    im = ax.imshow(np.arange(200).reshape((20, 10)))
##    divider = make_axes_locatable(ax)
##    width = axes_size.AxesY(ax, aspect=1/aspect)
##    pad = axes_size.Fraction(pad_fraction, width)
##    cax = divider.append_axes("right", size=width, pad=pad)
##    plt.colorbar(im, cax=cax)
#
#    # build plot
#    sns.set(style="white", color_codes=True, font_scale=1.5)
#    plt.figure(figsize=figsize)
#    ax = plt.gcacol()
#    im = plt.imshow(mat, interpolation='nearest')
#
#    # ticks and labels:
#    plt.yticks(range(len(rownames)),rownames)
#    plt.xticks(range(len(colnames)),colnames, rotation='vertical')
#    if len(xlabel) > 0:
#        plt.xlabel(xlabel)
#    if len(ylabel) > 0:
#        plt.ylabel(ylabel)
#
#    # get sizing info for heatmap, then plot colorbar:
#    divider = make_axes_locatable(ax)
#    width = axes_size.AxesY(ax, aspect=1/aspect)
#    pad = axes_size.Fraction(pad_fraction, width)
#    cax = divider.append_axes("right", size=width, pad=pad)
#    plt.colorbar(im, cax=cax)
#
#    # hack..
#    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)


def heatmap(mat, rownames=[], colnames=[], xlabel=[], ylabel=[], figsize=(10,10)):
    '''
    Builds an imshow of any matrix, with label names
    mat = np.ndarray (2d matrix)
    rownames = array of row names
    colnames = array of column names

    instructions on getting the colorbar to the right size from:
    http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    '''

    # build plot
    sns.set(style="white", color_codes=True, font_scale=1.5)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = plt.imshow(mat, interpolation='nearest')

    # ticks and labels
    if len(rownames) > 0:
        plt.yticks(range(len(rownames)),rownames)
    if len(colnames) > 0:
        plt.xticks(range(len(colnames)),colnames, rotation='vertical')
    if len(xlabel) > 0:
        plt.xlabel(xlabel)
    if len(ylabel) > 0:
        plt.ylabel(ylabel)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # hack..
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)


def reorder_df_cols_by_hierarchical_clusters(df, nsamples=10000):
    '''
    This will reorder the columns of a dataframe based on
    hierarchical clustering of the columns.

    '''

    # choose # rows to use. note, too many will be very slow.
    nrows = df.shape[0]
    if nsamples > nrows:
        nsamples = nrows

    # perform hierarchical clustering
    Z=linkage(df.sample(n=nsamples).values.T, 'single', 'correlation')
    W = dendrogram(Z, no_plot=True)
    inds = map(int, W['ivl'])

    df_clustered = df.iloc[:, inds]
    return df_clustered


def plot_corr(df,size=10, toCluster=True, nsamples=10000):
    '''

    Function plots a graphical correlation matrix for each pair of
    columns in the dataframe.

    Developed by Dyfrig

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        nsamples = # samples used for clustering of columns

    '''

    # first, reorder columns based on hierarchical clustering:
    if toCluster == True:
        df = reorder_df_cols_by_hierarchical_clusters(df, nsamples=nsamples)

    # now, plot the correlation matrix:
    sns.set(style="white", color_codes=True, font_scale=1.5)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns,  rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);

    # return style (hack):
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)


def pairgrid_with_hues(df, hue, huevals, huelabels, figsize=(20,20), DispCorrs=True, alpha=0.2, setloglog=False, rename_features=False, legendLoc=1):
    '''
    This plots something like a pairgrid from seaborn, but will properly
    handle a 'hue' variable. The hue variable is a column in df that has
    2 values in it, which we want to visualize separately. So the plots
    will each have 2 colors, which correspond to the two values of hue
    (which is a column in df). So, for example, if there is a df of cancer
    characteristics, and column 'hascancer' specifies whether a patient
    (row) has cancer, you will plot the distribution of cancer and non-
    cancer patients for each column(=feature) by setting hue='hascancer'.

    The last row of the plot will correspond to the hue variable, and will
    output distplots of the two values of hue, lined up with the plots
    above.

    To exclude some columns, run this function on dfplot here:
    dfplot = df.loc[:,list(set(df.columns) - set(['cols','to','exclude'])) ]

    inputs:

    df = a pandas dataframe
    hue = 'hascancer' (column name of the label variable to be plotted)
    huevals = [0, 1] (the unique values in hue)
    huelabels = ['normal','cancer'] (the names of the huevals)
    figsize # the size of the figure, default (20,20)
    DispCorrs # to display correlation coeffs? default is True

    ex:
    df = pd.DataFrame({'a':[1,2,3,10],'b':[4,5,6,100],'d':[10000,1000,100,1],'c':[1, 1, 0, 0]})
    pairgrid_with_hues(df, 'c', [1, 0], ["one","zero"], alpha=1, setloglog=True, DispCorrs=False)

    '''

    # set the ratio of blank space on edges of plots:
    edgeratio = 0.1

    # if plotting loglog, remove zero entries from non-hue cols:
    if setloglog==True:

        nonhuecols = list(set(df.columns) - set([hue]))
        for col in nonhuecols:
            df = df.loc[df[col]>0,:]

            ##why?

    fig = plt.figure(figsize=figsize)

    # put hue col at end:
    df = move_col_to_end_of_df(df, hue)

    # split into dfs for each hue val:
    df1 = df.loc[df[hue]==huevals[0],:]
    df2 = df.loc[df[hue]==huevals[1],:]

    # col = # of columns in df (not the same as subplot column):
    cols = df1.columns
    Ncols = len(cols)

    # step through rows (iR) and columns (iC) of subplots:
    for iR in xrange(1,Ncols):
        for iC in xrange(0,iR):
            # set the current subplot axis:
            ax=plt.subplot(Ncols,Ncols,iR*Ncols+iC+1)

            # determine the current data for this subplot:
            xdata1 = df1[cols[iC]]
            xdata2 = df2[cols[iC]]
            xdata_all = df[cols[iC]]

            ydata1 = df1[cols[iR]]
            ydata2 = df2[cols[iR]]
            ydata_all = df[cols[iR]]

            # if on last row, do distplot of the hue column:
            if iR == Ncols-1:
                g = sns.distplot(xdata1, label=huelabels[0], color='blue')
                sns.distplot(xdata2, label=huelabels[1], color='red')
                plt.legend(loc=legendLoc)
                g.set(yticklabels=[])

            # otherwise, do a scatter plot:
            else:

                plt.scatter(xdata1,ydata1,alpha=alpha,color="blue")
                plt.scatter(xdata2,ydata2,alpha=alpha,color="red")

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                # determine the correlations:
                if DispCorrs:
                    cc1 = sp.stats.pearsonr(xdata_all,ydata_all)[0]
                    cc2 = sp.stats.spearmanr(xdata_all,ydata_all).correlation
                    plt.title("Corr: %.2f | %.2f" % (cc1,cc2))

                # set y_lim:
                ymin = ydata_all.min()
                ymax = ydata_all.max()
                y_range = ymax - ymin

                # log axes, if desired:
                if setloglog==True:
                    ax.set_yscale('log')
                    ax.set_ylim([ymin,ymax])

                else:
                    yedge = y_range*edgeratio
                    ax.set_ylim([ymin-yedge, ymax+yedge])

            # set x_lim (including for the hue variable row):
            xmin = xdata_all.min()
            xmax = xdata_all.max()
            x_range = xmax - xmin

            # log axes, if desired:
            if setloglog==True:
                ax.set_xscale('log')
                ax.set_xlim([xmin, xmax])

            else:
                xedge = x_range*edgeratio
                ax.set_xlim([xmin-xedge, xmax+xedge])

#            print 'col:%s, row:%s, xmin: %s, xmax: %s, xedge: %s, ' % (cols[iC], cols[iR], xmin, xmax, xedge)

            # labels:
            if (iC==0) & (iR < Ncols-1):
                if rename_features:

                    plt.ylabel(feature_names([cols[iR]])[0])
                else:
                    plt.ylabel(cols[iR])

            if iR==Ncols-1:
                if rename_features:
                    plt.xlabel(feature_names([cols[iC]])[0])
                else:
                    plt.xlabel(cols[iC])

#       fig.tight_layout()
#    return fig, ax, xmin, xmax, xedge

def distplot_subplots_with_hue(df, hue, huevals, huelabels, Nplotrows, Nplotcols, setxlog=False, figsize=(20,10), legendloc=1, rug=False, rename_features=False):

    '''
    Create series of subplots, where each is the distribution plot
    of a different column in df (a pandas dataframe). The
    plots will each have 2 colors, which correspond to the two values
    of hue (which is a column in df). So, for example, if there is
    a df of cancer characteristics, and column 'hascancer' specifies
    whether a patient (row) has cancer, you will plot the distribution
    of cancer and non-cancer patients for each column(=feature)
    by setting hue='hascancer'.

    To exclude some columns, run this function on dfplot here:
    dfplot = df.loc[:,list(set(df.columns) - set(['cols','to','exclude'])) ]

    inputs:
    df = a pandas dataframe
    hue = 'hascancer' (column name of the label variable to be plotted)
    huevals = [0, 1] (the unique values in hue)
    huelabels = ['normal','cancer'] (the names of the huevals)
    Nplotrows = 2 (# of rows of subplots to plot)
    Nplotcols = 5 (# of columns of subplots to plot)

    '''

    fig = plt.figure(figsize=figsize)

    df1 = df.loc[df[hue]==huevals[0], :]
    df2 = df.loc[df[hue]==huevals[1], :]

    n = 1
    for col in df:
        if not(col==hue):
            ax = fig.add_subplot(Nplotrows,Nplotcols,n)
            plt.sca(ax)
            sns.distplot(df1[col], label=huelabels[0], rug=rug)
            sns.distplot(df2[col], label=huelabels[1], rug=rug)

            # log axes, if desired:
            if setxlog==True:
                ax.set_xscale('log')

            plt.legend(loc=legendloc)
            n = n + 1

            if rename_features:
                plt.xlabel(feature_names([col])[0])


    fig.tight_layout()

#    return fig


def distplot_subplots(df, Nplotrows, Nplotcols, setxlog=False, figsize=(20,10), legendloc=1, rug=False, rename_features=False, dropzeros=False):

    '''
    Create series of subplots, where each is the distribution plot
    of a different column in df (a pandas dataframe).

    To exclude some columns, run this function on dfplot here:
    dfplot = df.loc[:,list(set(df.columns) - set(['cols','to','exclude'])) ]

    inputs:
    df = a pandas dataframe
    hue = 'hascancer' (column name of the label variable to be plotted)
    huevals = [0, 1] (the unique values in hue)
    huelabels = ['normal','cancer'] (the names of the huevals)
    Nplotrows = 2 (# of rows of subplots to plot)
    Nplotcols = 5 (# of columns of subplots to plot)

    '''

    fig = plt.figure(figsize=figsize)

    n = 1
    for col in df:
        ax = fig.add_subplot(Nplotrows,Nplotcols,n)
        plt.sca(ax)

        if dropzeros:
            sns.distplot(df.loc[~(df[col]==0), col], rug=rug)
        else:
            sns.distplot(df[col], rug=rug)

        # log axes, if desired:
        if setxlog==True:
            ax.set_xscale('log')

#        plt.legend(loc=legendloc)
        n = n + 1

        if rename_features:
            plt.xlabel(feature_names([col])[0])

    fig.tight_layout()


def plot_feature_importances(X_names, importances, Nfeatures=100):
    '''
    Need to fix this code..
    Builds barplot of feature importances for random forest.
    model should be a randomForest model, already trained.
    X_names are the names of the features (np array)
    importances = model.feature_importances_ for random forest
    importances = model.get_params() for logistic regression
    Nfeatures = # features to plot (default is all of them)
    '''

    # get nice feature names for plot:
#    if useFeatureNames:
#        fnames = feature_names(X_names)
#    else:
#        fnames = X_names
    fnames = X_names

    # convert fnames type if needed
    if fnames.__class__ ==list:
        fnames = np.array(fnames)

    # optionally, take only the top features:
    indices = np.argsort(importances)#[::-1]
#    print len(indices)
    if Nfeatures < len(indices):
        indices = indices[-Nfeatures:]

#    print fnames
    fnames_plot = fnames[indices]
    imps_plot = importances[indices]

    plt.figure(figsize=(3, 7))
    plt.title('Feature importances')
    plt.barh(range(len(imps_plot)), imps_plot, align='center', )
    plt.ylim([-1, len(fnames_plot)])
    plt.yticks(range(len(fnames_plot)), fnames_plot)
#    plt.barh(range(len(fnames)), importances[indices], align='center', )
#    plt.ylim([-1, len(fnames)])
#    plt.yticks(range(len(fnames)), fnames[indices])
    plt.xticks(rotation=90)
#    for f in range(features):
#        print ("%2d) %-*s %f" % (f+1, 30, features[f], importances[indices[f]]))

    return plt


def display_num_nulls_per_column(df):
    numnulls = df.isnull().sum()
    pd.set_option('display.max_rows', len(numnulls))
    numnulls.sort_values(inplace=True, ascending=True)
    print 'Number of nulls per column:\n'
    print numnulls


def squaregridhistplot(features_df):
    '''
    Plots a grid of plots, each row & col corresponding to a column in the dataframe, with contour maps for each pair & hists on the diagonal
    From: http://stanford.edu/~mwaskom/software/seaborn-dev/tutorial/distributions.html
    '''
    g = sns.PairGrid(features_df)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
    return g


def render_confusion_matrix(y_true, y_pred, pos_class=True, neg_class=False):
    '''
    Code adapted from Python Machine Learning book
    name_pos_class is name of the positive class (string)
    name_neg_class is name of the negative class (string)
    Usually put y_test as y_true input
    '''

    #ax.set_axis_bgcolor('white')

    # set style:
    sns.set(style="white", color_codes=True, font_scale=1.5)
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # make plot:
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    # fix labels:
    ax.set(xticklabels=['', neg_class, pos_class, ''])
    ax.set(yticklabels=['', neg_class, pos_class, ''])

    plt.tight_layout()
    plt.show()

    # return to default (this is a hack..)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)


def plot_roc_curve(y_true, y_predictedprobs, startNewPlot=True, withLabel=True):
    '''
    Plots an roc curve.

    For random forest:
    y_predictedprobs = model.predict_proba(X_test)[:,1]
    '''

    # set style:
    sns.set(style="white", color_codes=True, font_scale=1.5)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_predictedprobs)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    if startNewPlot:
        plt.figure()


    # Plot of a ROC curve for a specific class
    if withLabel:
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    else:
        plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    err = 0.01
    plt.xlim([-err, 1])
    plt.ylim([0.0, 1+err])
    plt.axes().set_aspect('equal')
#    plt.show()

    # return to default (this is a hack..)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)

    return fpr, tpr, thresholds


def plot_roc_curves_with_mean(y_trues, y_pred_probas):
    '''
    y_trues and y_pred_probas are lists of length nIters,
    with a y_true and a y_predicted_probability vector in
    each list element (as come out of a machine learning model)
    '''

    nIters = len(y_trues)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    plt.figure()

    for iter in range(nIters):
       probas = y_pred_probas[iter]
       y_true = y_trues[iter]
       fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, probas)
       mean_tpr += np.interp(mean_fpr, fpr, tpr)
       mean_tpr[0] = 0.0
       roc_auc = sklearn.metrics.auc(fpr, tpr)
       plot_roc_curve(y_true, probas, startNewPlot=False, withLabel=False)

    # determine mean line:
    mean_tpr /= nIters
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], 'k-')
    plt.legend(loc="lower right")
    plt.show()

    # return the style:
    sns.set(style="darkgrid", color_codes=True, font_scale=1.5)



#############################
## Miscellaneous functions ##
#############################


# Transform String to Integer Variables
def transform_string_cols_to_int_variables(df, cols_to_integerize):

#    to_integerize = ['Gender','Sex','Marital Status','Party Affiliation','Race']
    check = pd.DataFrame()
    mappings = {}
    for col in cols_to_integerize:
        keys = df[col].unique()
        values =  range(len(keys))
        mapping = {k: v for k, v in zip(keys, values)}
        df[col] = df[col].map(lambda s: mapping.get(s) if s in mapping else s)
        mappings[col] = mapping

    return df, mappings


def unique_vals(df, nvalscutoff=50):
    '''
    Print the unique values of columns in df, if they have less
    than nvalscutoff unique vals.

    inputs:
    df = a dataframe
    nvalscutoff = the max # of unique vals that is accepted for
    printing the column in unq

    output:
    unq (a dataframe)
    '''

    print 'total columns in df:', len(df)
    cols = [col for col in df.columns if len(df[col].unique())<=nvalscutoff]

    uniquevals = [np.sort(df[col].unique()) for col in cols]
    numnans = [len(df) - df[col].count() for col in cols]
    unq = pd.DataFrame({'col':cols, 'uniquevals':uniquevals, 'numnans':numnans})

    return unq


def convert_boolean_col_to_int(df, col):
    '''
    This function deals with a (potential bug?) problem in pandas
    When doing groupby, sometimes boolean columns are dropped, as
    they seem to not be treated as numbers (sometimes not - i'm not
    sure why). This function will take a boolean column in df and
    will convert it to integer, and will leave any other values
    alone (i.e., nans will stay as nans).

    '''
    df = df.copy()
    S = df[col].copy()
    S[S==False] = S[S==False].astype(int)
    S[S==True] = S[S==True].astype(int)
    df[col] = S
    return df


def groupby_col_and_avg_other_cols(df, col, keepColinDf=False):
    '''
    dataframe must be all numeric (aside from 'col' column)
    this groups by values in 'col', and then averages the values
    for all other columns corresponding to each unique value in 'col'.
    will keep 'col' in the output df if keepColinDf is true
    '''
    grouped = df.groupby(col)
    df = grouped.apply(lambda x: x.mean())

    # note, the index of the output df is the unique vals from col
    # this option makes them a new column as well (with same name)
    if keepColinDf:
        df[col] = df.index
    return df


def move_col_to_end_of_df(df, colname):
    '''
    moves column colname to be the last column of dataframe df
    '''
    col = df[colname]
    df = df.drop(colname, axis=1)
    df[colname] = col
    return df


def convert_regression_coefs_to_pdSeries(coef_, X_names):
    inlist = coef_.tolist()[0]
    index = X_names.tolist()
    S = pd.Series(inlist, index=index)
    return S


def test():
    return 1


def column_ttests(df, ttestcol, ttestcolCutoff=0.5):
    '''
    Performs ttest of all columns in df (except for ttestcol), against
    values of ttesetcol, where the values are split into positive and
    negative classes based on ttestcol being above or below
    ttestcolCutoff value.
    e.g., ttestcol = 'hasParkinsons',
    df = features_df
    NOTE: will throw an error if any of the columns are not numerical.
    '''

    # split data into low & high categories based on ttestcol vals:
    catlow = df[df[ttestcol] < ttestcolCutoff]
    cathigh = df[df[ttestcol] >= ttestcolCutoff]

    # create list of columns, not including the ttestcol:
    testcols = df.columns.tolist()
    testcols.pop(testcols.index(ttestcol))

    tstats = []
    pvals = []
    for feature in testcols:
#        print 'feature ======= %s' % feature
        t, p = ttest_ind(catlow[feature].dropna(), cathigh[feature].dropna())
        tstats.append(t)
        pvals.append(p)
        #scipy.stats.ranksums()[source]

    ttestresults = pd.DataFrame({'pvals':pvals,'tstats':tstats},index=testcols)
    ttestresults = ttestresults.sort_values('pvals')
    return ttestresults # testcols, pvals, tstats


def resample_to_match_distribution(df, distcol, splitcol, splitVal_resample, splitVal_guide, nbins, nResamples):
    '''
    This will take a dataframe, split it into two parts based on values
    in splitcol (which must only have 2 values in it), and will then
    resample rows from the resulting df_resample dataframe. The resampling
    will be done such that the distribution in column distcol matches
    the distribution of values in distcol in the df_guide dataframe.
    This is intended to help deconfound variables: e.g., if age is a
    confound for hasParkinsons, run it like this:

    df = features_df
    distcol = 'age'
    splitcol = 'hasParkinsons'
    splitVal_resample = False
    splitVal_guide = True
    nbins = 10
    nResamples = 100

    df = the dataframe to work on
    distcol = the column in df that should have matching distributions
    splitcol = the column in df that will be split into resample and guide
    splitVal_resample = value in splitcol defining the df to be resampled
    splitVal_guide = value in splitcol for df whose dist should be matched
    nResamples = # of rows from the resample df to be output

    resamples done without replacement. nans are not included in distribution.

    outputs:
    df_resampled = the resampled version of df_resample
    df_guide = the df with splitVal_guide vals in splitcol
    df_resample = the df to resample
    '''

    ### split dataframe into df_resample and df_guide:
    df_resample = df[df[splitcol] == splitVal_resample]
    df_guide = df[df[splitcol] == splitVal_guide]

    ### take a histogram of df_guide, to get density distribution:
    guidevals = df_guide[distcol].values
    guidevals = guidevals[~np.isnan(guidevals)]
    hist, binedges = np.histogram(guidevals, bins=nbins)

    ### create weights vector:

    # (1-row per row in df_resample, with a weight on that row
    # determined by the density distribution of df_guide)
    resamplevals = df_resample[distcol].values
    wts = np.zeros(resamplevals.shape)

    # find the vals within each histogram bin:
    for n, histval in enumerate(hist):
        leftedge = binedges[n]
        rightedge = binedges[n+1]
        goodinds = np.where((resamplevals > leftedge) & \
                               (resamplevals <= rightedge))
        wts[goodinds] = histval

    ### normalize weights:
    wts = wts/sum(wts)

    ### sample the indices of resamplevals with the given weights:
    #samples = np.array(['a','b','c','d'])
    allinds = np.arange(len(resamplevals))
    indsamples = np.random.choice(allinds, size=nResamples, replace=False, p=wts)

    ### output df_resample with only the sampled rows:
    # (this makes the index point to allinds, so they must be identical!)

    #df_resample = df_resample.reindex(allinds)
    #assert (numpy.array_equal(df_resample.index.values, allinds)), 'there is a problem with\the indices. sampling won''t come out right.'

    df_resampled = df_resample.iloc[indsamples,:]

    return df_resampled, df_guide, df_resample


@contextmanager
def suppress_stdout():
    '''
    From http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    Suppresses print statements for a function call.

    Use like:

    print "You can see this"
    with suppress_stdout():
        print "You cannot see this"
    print "And you can see this again"

    '''

    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def check_for_string_in_dfcol(df, col):
    '''
    Check if a column of a dataframe contains any strings
    '''
    hasString = False
    for val in df[col].values:
        if type(val)==str:
            hasString = True

    return hasString


#     ledger.loc[:,'hour_period'] = np.vectorize(lambda h, p: 'h' + str(h) + '_' + str(p))(ledger['hour'], ledger['period'])


def add_binary_cutoff_col(df, col, bincol, cutoff=nan):
    '''
    Adds a new column to df which is a binary label on col, with
    the binary values determined by cutoff. If a cutoff is not
    given, it will be taken as the mean of the original col.

    df: a dataframe
    col: column name in the dataframe to be binarized.
    bincol: name of the new binarized column to be added.
    cutoff: the cutoff in column col for determining what will be
    1 and what 0 in bincol.

    e.g., if cutoff = 2:
    col bincol
    1   0
    2   0
    3   1
    4   1
    '''

    if pd.isnull(cutoff):
        cutoff = np.mean(df[col])

    df[bincol] = nan
    df.loc[df[col]<=cutoff, bincol] = 0
    df.loc[df[col]>cutoff, bincol] = 1

    return df


##################
# Other/unneeded #
##################

def df_to_na_mask(df):
    '''
    Create 'mask' matrix that denotes positions of all missing
    values in a dataframe.
    '''

    # break df into values and colnames:
    colnames = df.columns
    na_mat = df.values
    na_mask = pd.isnull(df)

#    # initiate the mask matrix:
#    na_mask = np.ones(na_mat.shape)*np.nan
#
#    # assign values to mask:
#    Nrows = na_mat.shape[0]
#    Ncols = na_mat.shape[1]
#    for row in range(Nrows):
#        for col in range(Ncols):
#            print na_mat[row, col]
#            if pd.isnull(na_mat[row, col]):
#                na_mask[row, col] = 1
#            else:
#                na_mask[row, col] = 0
#
#    assert sum(np.isnan(na_mask))==0, 'Not all values assigned'
#
    return na_mask, colnames



