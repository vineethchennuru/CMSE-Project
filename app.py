from pickle import TRUE
import random
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import hiplot as hip

import streamlit as st

from plots import *

st.set_page_config(page_title='CMSE Project',layout="wide")

st.markdown("<h1 style=' text-align: center; color: magenta;'>CMSE Project Webapp</h1>", unsafe_allow_html=True)

st.markdown(
    '<p><center><img alt="Car Insurance" src="https://www.financialexpress.com/wp-content/uploads/2022/10/Why-you-should-buy-your-new-car-insurance-directly-from-the-insurer_Reference-image.png"  width="500" height="250"> </center></p>', unsafe_allow_html=True
    )

st.markdown('<p><hr></p>',unsafe_allow_html=True)

st.markdown(
    '<p style="color:magenta; font-size:22px"><u>Abstract</u>: Predict Health Insurance Owners who will be interested in Vehicle Insurance (Classification Problem)',unsafe_allow_html=True
)
st.markdown(
    '<p><u>Part1</u>: I will be exploring the dataset(understanding all the given columns), performing different EDA techniques, and deploying a dashboard with various visualization to slice, dice, and generate valuable insights from the data."',unsafe_allow_html=True
)

st.markdown(
    '<p ><u>Part2</u>: Taking insights generated from Part1 and building a model to predict whether the policyholders (customers) from the past year will also be interested in Vehicle Insurance provided by the company."',unsafe_allow_html=True
)

st.markdown('<p><hr></p>',unsafe_allow_html=True)

st.markdown('Code Repository : [GitHub Code Repo](https://github.com/vineethchennuru/CMSE-Project)')
st.markdown('Dataset link : [Health Insurance Cross Sell Prediction Dataset🏠🏥](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)')

st.markdown('<p><hr></p>',unsafe_allow_html=True)


# Reading data
original_data = pd.read_csv('Health_Insurance_Cross_Sell_Prediction.csv')

original_data.drop('id',axis=1,inplace=True)

original_data = original_data.astype({'Response':'string'})
original_data = original_data.astype({'Region_Code':'int'})
original_data = original_data.astype({'Annual_Premium':'int'})
original_data = original_data.astype({'Policy_Sales_Channel':'int'})

# For computational purposes taking 20% of data for analysis
# original_data = original_data.sample(n=int(len(original_data)/20))
original_data = original_data[0:int(len(original_data)/20)]
data = original_data.copy()

data['count_'] = 1

data.Driving_License = data.Driving_License.map({1:'YES',0:'NO'})

criteria = [data.Age.between(20, 39), data.Age.between(40, 50), data.Age.between(51, 85)]
values = ['20-40','41-60','61-85']
data.Age = np.select(criteria, values, 0)

criteria = [data.Region_Code.between(0, 18), data.Region_Code.between(19, 36), data.Region_Code.between(37, 52)]
values = ['0-18','19-36','37-52']
data.Region_Code = np.select(criteria, values, 0)

# CSS to inject contained in a string
hide_table_row_index = """
            <style>

            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.header('Data Description')
st.table(pd.read_csv('data_description.csv'))

st.header('Sample Data')
st.table(original_data.iloc[1:].head(10))


st.markdown('<p><hr></p>',unsafe_allow_html=True)

cols = ['Gender','Age','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']
st.header('Sankey chart')
options = st.multiselect(
    'Select columns you want to visualize in Sankey chart',
    cols,
    cols)
fig = genSankey(data,cat_cols=options,value_cols='count_',title='Sankey')
# Plot!
st.plotly_chart(fig, use_container_width=True)

tab1, tab2, tab3, tab4 = st.tabs(["1D Analysis", "2D Analysis", "Hi Plot", "Plots and steps for next steps",])

with tab1:
    st.header("One dimentional analysis of different columns w.r.t to Response")

    # Plot-1 in Tab-1

    st.subheader("Distplot with Response as hue ")

    column_name =  st.selectbox(
        'For which column would you like to see the distplot',
        ('Age','Region_Code','Annual_Premium','Policy_Sales_Channel','Vintage'))
    fig = getDistributionplot(original_data,column_name)
    st.plotly_chart(fig,use_container_width = True)

    # Plot-2 in Tab-1

    st.subheader('Sunburst chart for given path')

    path = ['Gender','Age','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']
    options = st.multiselect(
        'Select columns you want to visualize in Sunburst chart',
        path,
        ['Region_Code','Age','Response'])
    if len(options) >4:
        st.markdown(
            '<p style="color:red; font-size:22px">For better understanding purpose please select less than or equal to 4 columns in the filter</p>',unsafe_allow_html=True)
    else:
        fig = getSunburstPlot(data,options)
        st.plotly_chart(fig,use_container_width = True)

with tab2:
    st.header("Two dimentional analysis of different columns w.r.t to Response")
    col1, col2 = st.columns(2,gap='large')

    with col1:
        options = ('Gender','Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                    'Vehicle_Damage','Annual_Premium', 'Policy_Sales_Channel', 'Vintage')
        options_cols_x = options
        options_cols_y = options

        st.subheader("Relation plot with hue as Response")

        options_x = st.selectbox('x-axis',options_cols_x,index=1)
        options_y = st.selectbox('y-axis',options_cols_y,index=6)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.write('You selected:', options)

        fig = getRelPlot(original_data,options_x,options_y)
        st.pyplot(fig,use_container_width = True)

    with col2:

        cols=['Gender','Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                'Vehicle_Damage','Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
        
        st.subheader("Pairplot with hue as Response")
        
        options = st.multiselect(
            'Select columns you want to visualize in Pairplot',
            cols,
            ['Age','Annual_Premium','Vintage'])
        
        fig = getPairPlot(original_data,options)
        st.pyplot(fig,use_container_width = True)

    options = ('Gender','Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
                'Vehicle_Damage','Annual_Premium', 'Policy_Sales_Channel', 'Vintage')

    st.markdown('<p><hr></p>',unsafe_allow_html=True)

    st.subheader("Marginal Plot with hue: Response")

    marginal_x = options
    marginal_y = options

    marginal_options_x = st.selectbox('x axis',marginal_x,index=1)
    marginal_options_y = st.selectbox('y axis',marginal_y,index=7)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.write('You selected:', options)

    fig = getMarginalPlot(original_data,marginal_options_x,marginal_options_y)
    st.plotly_chart(fig,use_container_width = True)


with tab3:
    xp = hip.Experiment.from_dataframe(original_data)
    # Instead of calling directly `.display()`
    # just convert it to a streamlit component with `.to_streamlit()` before
    st.subheader('High-dimensional interactive plot for the dataset')
    ret_val = xp.to_streamlit().display()

with tab4:
    st.header("Plots and insights that will help us for modelling")
    st.subheader('Correaltion matrix for the dataset')
    fig = getCorrelationPlot(original_data)
    st.plotly_chart(fig)


    st.subheader('Few intresting stats about the data')
    original_data = original_data.astype({'Response':'int'})
    df_descibe = original_data.describe().reset_index().rename({'index':''},axis=1)
    st.table(df_descibe)

st.markdown('<p><hr></p>',unsafe_allow_html=True)

st.subheader('Next we will be trying to do preprocessing on data and modelling the data with different models')

st.text('EDA steps on the data are:')
st.text('--> Applied standard scalar on "Age" and "Vintage" columms')
st.text('--> Dummy variables from "Vehicle age" column')
st.text('--> "Gender", "Vehicle Damage" are mapped to intergers')
st.text('--> Since the no of data points per class is highly imbalanced, we applied SMOTE(Synthetic Minority Oversampling Technique)')

st.text('These are the brief list of steps that are followed after above EDA')
st.text('--> All the models are being created in a batch processing manner')
st.text('--> And the models are saved as binary files, which can be used later for predictions')
st.text('--> We are saving the models as the amount of data the model is being trained on is quite high(381109 rows and 8 columns)')
st.text('--> We used 9 classification models, they are :')
st.table(pd.DataFrame(('XG Boost',"Nearest Neighbors","SVC","Decision Tree","Random Forest","Neural Net" ,"AdaBoost","Naive Bayes","QDA"
),columns=['Model Name']
).T)
st.text('--> Cross validation and hyperparameter tuning were used to get the best model(without overfitting) and best hyperparamters for the given\n data')


st.markdown('<p><hr></p>',unsafe_allow_html=True)

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (f1_score, roc_auc_score,accuracy_score,confusion_matrix,
                             precision_recall_curve, auc, roc_curve, recall_score,classification_report)

from model_plot_func import (plot_classification_report,
                            plot_roc,plot_confusion_matrix,
                            get_ClassificationReport,
                            get_xg_boost_data,
                            get_predict)

import plotly.express as px

import pickle

# Reading data
original_data = pd.read_csv('Health_Insurance_Cross_Sell_Prediction.csv')
original_data = original_data[0:int(len(original_data)/20)]
columns = original_data.columns

train,test = train_test_split(original_data,test_size=0.25)

### Data Preprocessing

num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel']

train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
train=pd.get_dummies(train,drop_first=True)
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

from imblearn.over_sampling import SMOTE
# Resampling the minority class. The strategy can be changed as required.
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(train.drop('Response', axis=1), train['Response'])
train = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)

ss = StandardScaler()
train[num_feat] = ss.fit_transform(train[num_feat])

mm = MinMaxScaler()
train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])

train=train.drop('id',axis=1)

for column in cat_feat:
    train[column] = train[column].astype('str')


test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
test[num_feat] = ss.fit_transform(test[num_feat])

from imblearn.over_sampling import SMOTE
# Resampling the minority class. The strategy can be changed as required.
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(test.drop('Response', axis=1), test['Response'])
test = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)

mm = MinMaxScaler()
test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])
for column in cat_feat:
    test[column] = test[column].astype('str')


train_copy = train
train_target=train['Response']
train=train.drop(['Response'], axis = 1)
x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)

### Using saved models

model_names = (
    'XG Boost',
    "Nearest Neighbors",
    "SVC",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA"
)

st.header("Different Modelling outputs")

column_name =  st.selectbox(
    'Train / Test Data',
    ('Train','Test'))
model_name =  st.selectbox(
    'Select which model output you want to view',
    model_names)

if column_name == 'Train':
    path = 'IPYNB files/models/'+model_name+'_model.sav'
    x_,y_ = x_train,y_train
    if model_name == 'XG Boost':
        x_,y_ = get_xg_boost_data(train_copy,'Train')
elif column_name == 'Test':
    path = 'IPYNB files/models/'+model_name+'_model.sav'
    x_,y_ = x_test,y_test
    if model_name == 'XG Boost':
        x_,y_ = get_xg_boost_data(train_copy,'Test')


col1, col2 = st.columns(2,gap='large')

with col1:
    text = get_ClassificationReport(path,x_,y_)
    st.subheader("Classification Report")
    st.text('->'+text)

with col2:
    st.subheader("Classification Plot")
    fig = plot_classification_report(text)
    st.pyplot(fig,use_container_width = True)

col3, col4 = st.columns(2,gap='small')

with col3:
    st.subheader("Roc Plot")
    fig = plot_roc(path,x_,y_)
    st.plotly_chart(fig)

with col4:
    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(path,x_,y_)
    st.plotly_chart(fig)


st.subheader("Predict on "+column_name+" data")
st.text(column_name+' data has - '+str(len(x_))+' rows')
row_number = st.slider(label = 'Row number of test data to be predicted',
                        min_value=1,max_value=len(x_),step=1)
output_predict = get_predict(path,x_,row_number)
st.text('For row number:'+str(row_number)+' which has the folloing data :')
st.table(x_.iloc[[row_number]])
st.text('The Predicted Reponse is : '+str(output_predict[0]))

