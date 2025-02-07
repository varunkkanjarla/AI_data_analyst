import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set up page layout
st.set_page_config(page_title="AI CSV Data Analyst", layout="wide")

# Initialize Gemini API (Replace with your API Key)
GEMINI_API_KEY = "api-key"
genai.configure(api_key=GEMINI_API_KEY)

# Create two columns (70:30 split)
left_col, right_col = st.columns([7, 3])

with left_col:
    st.title("📊 AI-Powered CSV Data Analyst")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read File
        file_ext = uploaded_file.name.split(".")[-1]
        if file_ext == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_ext == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # Display the DataFrame
        st.subheader("📂 Uploaded Data")
        st.dataframe(df)

        # Data Insights
        st.subheader("📈 Data Insights")
        
        # Dataset Summary
        st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.write(f"**Missing Values:**\n{df.isnull().sum()}")

        # Basic Statistics
        st.subheader("📊 Statistical Summary")
        st.write(df.describe())

        # Visualizations
        st.subheader("📉 Data Visualizations")

        # Select Column for Histogram
        numeric_columns = df.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            col = st.selectbox("Select a column for histogram:", numeric_columns)
            fig = px.histogram(df, x=col, title=f"Histogram of {col}")
            st.plotly_chart(fig)

        # Correlation Heatmap
        if len(numeric_columns) > 1:
            st.subheader("🔍 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

with right_col:
    st.subheader("💬 Chat with Your Data")

    user_query = st.text_input("Ask a question about the data...")

    if user_query and uploaded_file is not None:
        # Prepare prompt for AI
        prompt = f"""
        You are a data analyst. The user has uploaded a dataset. 
        Answer the query based on the dataset provided.

        Dataset Overview:
        {df.describe().to_string()}

        User Question:
        {user_query}
        """

        # Get AI response
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        # Display AI Response
        if response.text:
            st.write("🤖 AI Response:")
            st.write(response.text)
