import streamlit as st

# Page config
st.set_page_config(
    page_title="Excelerate - Auto ML Platform",
    page_icon="🚀",
    layout="wide"
)

# Hero Section
st.title("🚀 Excelerate")
st.subheader("Automate Data Preprocessing & Machine Learning in Seconds")

st.markdown("""
Welcome to **Excelerate**, your all-in-one platform to:
- ⚡ Automatically clean and preprocess data
- 📊 Perform EDA (Exploratory Data Analysis)
- 🤖 Train multiple ML models instantly
- 🏆 Get the best model with performance metrics
- 📥 Download trained models

No coding required. Just upload your dataset and go!
""")

# CTA Button
st.divider()
if st.button("🚀 Get Started"):
    st.switch_page("pages/upload.py")  # create this page later

# Features Section
st.divider()
st.header("✨ Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📂 Data Upload")
    st.write("Upload CSV/Excel datasets easily.")

with col2:
    st.subheader("🧹 Auto Preprocessing")
    st.write("Handles missing values, encoding, scaling automatically.")

with col3:
    st.subheader("🤖 ML Models")
    st.write("Regression & Classification models with comparison.")

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("📊 EDA")
    st.write("Visualize distributions, correlations, and insights.")

with col5:
    st.subheader("🏆 Best Model Selection")
    st.write("Automatically selects the best-performing model.")

with col6:
    st.subheader("📥 Download")
    st.write("Download trained model & processed dataset.")

# How it Works
st.divider()
st.header("⚙️ How It Works")

st.markdown("""
1. 📤 Upload your dataset  
2. 🔍 Select target column  
3. ⚡ Auto preprocessing runs  
4. 🤖 Multiple ML models trained  
5. 🏆 Best model selected  
6. 📥 Download results  
""")

# Footer
st.divider()
st.markdown("💡 Built with Streamlit | By Masud Mallik")