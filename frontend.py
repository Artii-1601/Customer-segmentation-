import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import pairwise_distances_argmin_min
with open('Agglomodel.pkl','rb') as file :
    acmodel=pickle.load(file)
with open('scaled_obj.pkl','rb') as file:
    obj=pickle.load(file)
d={0:'Average income with low spending',
1:'Low income with average spending',
2:'High income with low spending',
3:'High income with high spending'}
df_string=pd.read_csv('copydataset.csv')
df_scaled=pd.read_csv('orginal_df.csv')
st.set_page_config(page_title='Customer Lifestyle segmentation')
st.header("💼 Customer Lifestyle Segmentation")
st.subheader("Income and Spending Pattern Analysis")
with st.container(border=True):
    c1,c2=st.columns(2)
    gen=c1.radio('Gender',options=df_string['Gender'].unique())
    age=c2.slider("Age",min_value=15)
    AnIn=c1.number_input('Annual Income')
    spendingscore=c2.slider('Spending score',min_value=0,max_value=100)
    gender=list(df_string['Gender'].unique())
    gender.sort()
    in_vals=[[gender.index(gen),age,AnIn,spendingscore]]
    in_vals=obj.transform(in_vals)
    df_new=pd.DataFrame(in_vals,columns=['Gender','Age','Annual Income','Spending Score'])
    df_scaled.drop('Unnamed: 0',axis=1)
    col1,col2,col3=st.columns([1.7,1.8,1]) #for button 
    cl1,cl2,cl3=st.columns([0.5,1.8,0.2]) #for output
    if col2.button('**Analyze Now**'):
        df_scaled=df_scaled.drop('Unnamed: 0',axis=1)
        new,_=pairwise_distances_argmin_min(df_new,df_scaled)
        label=acmodel.labels_[new]
        cl2.subheader(d[label[0]])
