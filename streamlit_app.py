import streamlit as st
import pandas as pd

st.set_page_config(page_title="Steel Fault App", layout="wide")

st.title("🔩 Steel Plate Fault Detection App")
st.write("A machine learning web app that detects faults in steel plates.")

st.write("Step 1: loading dataset...")

try:
    df = pd.read_csv("Steel_Plates_Faults.csv")
    st.success("Dataset loaded successfully!")
except Exception as e:
    st.error("❌ ERROR: Dataset not found or cannot be loaded")
    st.exception(e)
    st.stop()

st.write("Step 2: dataset preview")
st.dataframe(df.head())
