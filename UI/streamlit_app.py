import streamlit as st
import requests
from PIL import Image
from io import StringIO
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import streamlit.components.v1 as components
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="🎬 Movie Review Sentiment Analysis 🎬",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="auto"
)

# Azure Blob Storage settings
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = "project2"
BLOB_NAME = "IMDB Dataset.csv"

# Define the FastAPI URL endpoints
API_URL = "http://127.0.0.1:8000"
TOKEN_ENDPOINT = f"{API_URL}/token"
PREDICT_ENDPOINT = f"{API_URL}/predict/"

# Initialize session state for authorization and access token
if "authorized" not in st.session_state:
    st.session_state["authorized"] = False
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None

# Sidebar for page navigation
page = st.sidebar.selectbox("Choose a page", ["Login", "Data Drift Detection", "Sentiment Analysis"])

# Page 1: Login
if page == "Login":
    st.title("Login")

    # Login input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Request an access token from FastAPI using username and password
        response = requests.post(TOKEN_ENDPOINT, data={"username": username, "password": password})
        if response.status_code == 200:
            access_token = response.json().get("access_token")
            st.success("Logged in successfully!")
            st.session_state["access_token"] = access_token
            st.session_state["authorized"] = True
        else:
            st.error("Login failed. Please check your credentials.")

# Page 2: Data Drift Detection (only accessible after login)
elif page == "Data Drift Detection" and st.session_state["authorized"]:
    st.title("Data Drift Detection")

    def download_blob_to_df(container_name, blob_name):
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob().readall()
        data_str = blob_data.decode('utf-8')
        df = pd.read_csv(StringIO(data_str))
        return df

    # Load the dataset from Azure Blob Storage
    df = download_blob_to_df(BLOB_CONTAINER_NAME, BLOB_NAME)

    if df is not None:
        if 'review' not in df.columns:
            st.error("'review' column not found in the dataset.")
        else:
            df.drop(columns=['sentiment'], inplace=True)

            vectorizer = TfidfVectorizer(max_features=100)
            review_vectorized = vectorizer.fit_transform(df['review']).toarray()
            review_df = pd.DataFrame(review_vectorized, columns=vectorizer.get_feature_names_out())

            sample_ref = review_df.iloc[100:200]
            sample_test = review_df.iloc[600:700]

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=sample_ref, current_data=sample_test)
            report_file = "drift_report.html"
            report.save_html(report_file)

            with open(report_file, "r") as f:
                report_html = f.read()

            st.write("### Data Drift Report")
            components.html(report_html, height=1000)
    else:
        st.error("Failed to load the dataset from Azure.")

# Page 3: Sentiment Analysis (only accessible after login)
elif page == "Sentiment Analysis" and st.session_state["authorized"]:
    st.title("🎬 Movie Review Sentiment Analysis 🎬")

    # Display a movie-themed image
    image = Image.open("movies.jpg")
    st.image(image, caption='Let’s find out what people think about this movie!', use_column_width=True)

    # Input selection for either text or CSV
    input_type = st.radio("Choose input type:", ('Text', 'CSV'))

    # Text input or CSV file upload
    if input_type == 'Text':
        review_input = st.text_area("Enter a movie review:")
    elif input_type == 'CSV':
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Create a button to make a prediction
    if st.button("🎬 Analyze Sentiment 🎬"):
        headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}

        # Prediction for text input
        if input_type == "Text" and review_input:
            payload = {"text": review_input}
            response = requests.post(PREDICT_ENDPOINT, data=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()
                st.markdown(f"### 🎭 Sentiment Analysis Result 🎭")
                st.table(pd.DataFrame([{
                    "Text": result.get("review", "No text"),
                    "Label": result.get("sentiment", "Unknown label"),
                    "Score": result.get("Model Confidence Score", "No score")
                }]))
            else:
                st.error(f"Prediction failed. Status Code: {response.status_code}")

        # Prediction for CSV input
        elif input_type == "CSV" and uploaded_file:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(PREDICT_ENDPOINT, files=files, headers=headers)

            if response.status_code == 200:
                result = response.json()
                sentiment_label = result.get("overall_sentiment", "Unknown")
                combined_reviews = result.get("combined_reviews", "No review text")
                confidence = result.get("Model Confidence Score", 0)

                st.markdown(f"### 🎭 Sentiment Analysis Result for CSV 🎭")
                st.markdown(f"**Overall Sentiment**: {sentiment_label}")
                st.markdown(f"**Confidence**: {confidence:.2f}")
                st.markdown(f"**Combined Reviews (first 500 chars)**: {combined_reviews[:500]}")
            else:
                st.error(f"Prediction failed. Status Code: {response.status_code}")
        else:
            st.error("Please provide input for prediction.")

else:
    if not st.session_state["authorized"]:
        st.warning("Please log in to access the other pages.")