from asyncio.windows_events import NULL
from pickle import TRUE
from pkgutil import get_data
from numpy import NaN
import streamlit as st
#EDA packs 
import matplotlib
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
import matplotlib.pyplot as plt



import seaborn as sns

matplotlib.use("Agg")
fig, ax = plt.subplots()
matplotlib.rcParams.update({"font.size": 8})
st.set_option("deprecation.showPyplotGlobalUse", False)
def categorical_column(df, max_unique_values=15):
    categorical_column_list = []
    for column in df.columns:
        if df[column].nunique() < max_unique_values:
            categorical_column_list.append(column)
    return categorical_column_list





    # st.balloons()
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

                st.subheader("Automated EDA")
                 

                # Show Columns
                if st.checkbox("Columns Names"):
                    st.write(df.columns)

                # Show Shape
                if st.checkbox("Shape of Dataset"):
                    st.write(df.shape)
                    data_dim = st.radio("Show Dimension by ", ("Rows", "Columns"))
                    if data_dim == "Columns":
                        st.text("Numbers of Columns")
                        st.write(df.shape[1])
                    elif data_dim == "Rows":
                        st.text("Numbers of Rows")
                        st.write(df.shape[0])
                    else:
                        st.write(df.shape)

                # Select Columns
                if st.checkbox("Select Column to show"):
                    all_columns = df.columns.tolist()
                    selected_columns = st.multiselect("Select Columns", all_columns)
                    new_df = df[selected_columns]
                    st.dataframe(new_df)

                # Show Value Count
                if st.checkbox("Show Value Counts"):
                    all_columns = df.columns.tolist()
                    selected_columns = st.selectbox("Select Column", all_columns)
                    st.write(df[selected_columns].value_counts())

                # Show Datatypes
                if st.checkbox("Show Data types"):
                    st.text("Data Types")
                    df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
                    st.write(df_types.astype(str))

                # Show Summary
                if st.checkbox("Show Summary"):
                    st.text("Summary")
                    st.write(df.describe().T)

                # Plot and visualization
                st.subheader("Data Visualization")
                all_columns_names = df.columns.tolist()

                # Correlation Seaborn Plot
                if st.checkbox("Show Correlation Plot"):
                    st.success("Generating Correlation Plot ...")
                    if st.checkbox("Annot the Plot"):
                        st.write(sns.heatmap(df.corr(), annot=True))
                    else:
                        st.write(sns.heatmap(df.corr()))
                    st.pyplot()

                # Count Plot
                if st.checkbox("Show Value Count Plots"):
                    x = st.selectbox("Select Categorical Column", all_columns_names)
                    st.success("Generating Plot ...")
                    if x:
                        if st.checkbox("Select Second Categorical column"):
                            hue_all_column_name = df[df.columns.difference([x])].columns
                            hue = st.selectbox("Select Column for Count Plot", hue_all_column_name)
                            st.write(sns.countplot(x=x, hue=hue, data=df, palette="Set2"))
                        else:
                            st.write(sns.countplot(x=x, data=df, palette="Set2"))
                        st.pyplot()

                # Pie Chart
                if st.checkbox("Show Pie Plot"):
                    all_columns = categorical_column(df)
                    selected_columns = st.selectbox("Select Column", all_columns)
                    if selected_columns:
                        st.success("Generating Pie Chart ...")
                        st.write(df[selected_columns].value_counts().plot.pie(autopct="%1.1f%%"))
                        st.pyplot()

                # Customizable Plot
                st.subheader("Customizable Plot")

                type_of_plot = st.selectbox(
                    "Select type of Plot", ["area", "bar", "line", "hist", "box", "kde"]
                )
                selected_columns_names = st.multiselect("Select Columns to plot", all_columns_names)

                if st.button("Generate Plot"):
                    st.success(
                        "Generating Customizable Plot of {} for {}".format(
                            type_of_plot, selected_columns_names
                        )
                    )

                    custom_data = df[selected_columns_names]
                    if type_of_plot == "area":
                        st.area_chart(custom_data)

                    elif type_of_plot == "bar":
                        st.bar_chart(custom_data)

                    elif type_of_plot == "line":
                        st.line_chart(custom_data)

                    elif type_of_plot:
                        custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
                        st.write(custom_plot)
                        st.pyplot()
                            #profile = ProfileReport(df)
                            #st_profile_report(profile)
        elif choice=="Machine learning":
            if preprecess==False:
                st.subheader("Automated Machine Learning")
            
            



if __name__=="__main__":
    main()   