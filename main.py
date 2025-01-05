import streamlit as st
import openai
from langchain.llms import OpenAI
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import pickle

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model
llm = OpenAI(temperature=0.7)

# Cache setup
cache_file = "response_cache.pkl"
if not os.path.exists(cache_file):
    with open(cache_file, "wb") as f:
        pickle.dump({}, f)

def load_cache():
    with open(cache_file, "rb") as f:
        return pickle.load(f)

def save_cache(cache):
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

# Streamlit UI
st.title("Cache-Augmented AI Agent")

# Query input
user_query = st.text_input("Enter your query:")

if st.button("Generate Response"):
    cache = load_cache()
    if user_query in cache:
        st.write("Cached Response:")
        st.write(cache[user_query])
    else:
        response = llm(user_query)
        cache[user_query] = response
        save_cache(cache)
        st.write("New Response:")
        st.write(response)

# Data analysis and visualization
st.subheader("Upload CSV for Data Analysis")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Visualization
    st.subheader("Data Visualization")
    column = st.selectbox("Select a column to visualize:", df.columns)
    fig = px.histogram(df, x=column, title=f"Distribution of {column}")
    st.plotly_chart(fig)
