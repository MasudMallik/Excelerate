import streamlit as st
import pandas as pd
st.title("Prediction")
st.write("upload your dataet")
file_path=st.file_uploader(
    label="Upload .xlsx or .csv file",
    type=[".xlsx",".csv"]
)
if file_path:
    with st.status("file opening"):
        if file_path.name.endswith(".xlsx"):
            df=pd.read_excel(file_path)
        else:
            df=pd.read_csv(file_path)
        st.success("file succesfully opende")

        st.dataframe(df)