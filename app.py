from asyncio.windows_events import NULL
from pickle import TRUE
from pkgutil import get_data
from numpy import NaN
import streamlit as st
#EDA packs 
import pandas as pd
import codecs
#Components packs*
#we called this ver of comp so if we have a new ver our comp will note break
import streamlit.components.v1 as components
from pandas_profiling import ProfileReport 
from streamlit_pandas_profiling import st_profile_report
# Components Pkgs
import streamlit.components.v1 as components
from atom import ATOMClassifier
from atom.data_cleaning import Encoder


def main():
    """"a simple eda App with streamlit"""
    #create a page select menu
    st.set_page_config(layout="wide")
    menu=["Explore your data","Machine learning"]
    choice=st.sidebar.selectbox("Menu",menu)
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    preprecess=False
    if data_file is not None:
        df = pd.read_csv(data_file)
        #st.dataframe(df)
        scale = st.sidebar.checkbox("Scale", False, "scale")
        encode = st.sidebar.checkbox("Encode", False, "encode")
        impute = st.sidebar.checkbox("Impute", False, "impute")
        outliers = st.sidebar.checkbox("Outliers", False, "outliers")
        balancer = st.sidebar.checkbox("Balance", False, "balancer")
  
        placeholder = st.empty()  # Empty to overwrite write statements
        placeholder.write("Data Prepprocessing...")

        # Initialize atom
        
        X=df.loc[:, df.columns != -1]
        y=df[:-1]
        print(y)
        atom = ATOMClassifier(X,y, verbose=2, random_state=1)


        if impute:
            preprecess=True
            placeholder.write("Imputing the missing values...")
            atom.impute(strat_num="median",  strat_cat="drop", max_nan_cols=0.8)
        if encode:
            preprecess=True
            placeholder.write("Encoding the categorical features...")

            atom.encode(strategy="LeaveOneOut", max_onehot=10)
        if outliers:
            preprecess=True
            placeholder.write("Pruning values...")
            atom.prune(strategy="z-score", max_sigma=2, include_target=False)


        if scale:
            preprecess=True
            placeholder.write("Scaling the data...")
            atom.scale(strategy="minmax")
        if balancer :
            preprecess=True
            placeholder.write("Balance data values...")
            atom.balance(strategy="NearMiss", sampling_strategy=0.7, n_neighbors=10)
        placeholder.write("Preprocessing is over...")
        preprecess=False
        df=atom.dataset
        
        st.dataframe(df)
              
        if choice=="Explore your data" :
            if st.button('Start'):
                st.subheader("Automated EDA")
                profile = ProfileReport(df)
                st_profile_report(profile)
        elif choice=="Machine learning":
            if preprecess==False:
                st.subheader("Automated Machine Learning")
            
            



if __name__=="__main__":
    main()   