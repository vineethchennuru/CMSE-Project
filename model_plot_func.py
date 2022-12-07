import pickle
import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_curve, recall_score,auc,confusion_matrix

from sklearn.metrics import (f1_score, roc_auc_score,accuracy_score,confusion_matrix,
                             precision_recall_curve, auc, roc_curve, recall_score,classification_report)


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import xgboost as xgb


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, GridSearchCV


# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from scipy.stats import randint

from sklearn.metrics import (f1_score, roc_auc_score,accuracy_score,confusion_matrix,
                             precision_recall_curve, auc, roc_curve, recall_score,classification_report)
from sklearn.metrics import accuracy_score


import pickle



def get_xg_boost_data(train,dataset_name):
    X = train.drop(['Response'], axis=1)
    y = train['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)    
    X_train = X_train.drop(['Region_Code', 'Vintage', 'Driving_License'], axis=1)
    X_test = X_test.drop(['Region_Code', 'Vintage', 'Driving_License'], axis=1)    

    cate_ = ['Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes',
            'Policy_Sales_Channel','Previously_Insured','Gender']

    for col in cate_:
        X_train[col] = X_train[col].astype('float').astype('int')
        X_test[col] = X_test[col].astype('float').astype('int')    

    if dataset_name == 'Train':
        return X_train,y_train
    elif dataset_name == 'Test':
        return X_test,y_test
    


def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))
    return fig



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []

    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
#         print(v)
        plotMat.append(v)

#     print('plotMat: {0}'.format(plotMat))
#     print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    ans = heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

    return ans 
    
def plot_roc(model_path,x,y):
    model = pickle.load(open(model_path, 'rb'))

    y_score = model.predict_proba(x)[:,1]
    fpr, tpr, _ = roc_curve(y, y_score)

    # fpr, tpr, thresholds = roc_curve(y, y_score)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig

def plot_confusion_matrix(model_path,x,y):
    model = pickle.load(open(model_path, 'rb'))
    y_pred = model.predict(x)
    plt = px.imshow(confusion_matrix(y,y_pred),text_auto=True)
    plt.update_layout(
        xaxis_title="Predicted value",
        yaxis_title="Real Value"
    )

    return plt




pickle_file_map = {
    "Nearest Neighbors":'IPYNB files/models/'+"Nearest Neighbors"+'_model.sav',
    "SVC":'IPYNB files/models/'+"SVC"+'_model.sav',
    "Decision Tree":'IPYNB files/models/'+"Decision Tree"+'_model.sav',
    "Random Forest":'IPYNB files/models/'+"Random Forest"+'_model.sav',
    "Neural Net":'IPYNB files/models/'+"Neural Net"+'_model.sav',
    "AdaBoost":'IPYNB files/models/'+"AdaBoost"+'_model.sav',
    "Naive Bayes":'IPYNB files/models/'+"Naive Bayes"+'_model.sav',
    "QDA":'IPYNB files/models/'+"QDA"+'_model.sav'
}


def get_ClassificationReport(model_path,x,y):
    model = pickle.load(open(model_path, 'rb'))
    y_pred = model.predict(x)
    return classification_report(y, y_pred,zero_division=0)

def get_predict(model_path,x,row_number):
    model = pickle.load(open(model_path, 'rb'))
    y_pred = model.predict(x.iloc[[row_number]])
    return y_pred