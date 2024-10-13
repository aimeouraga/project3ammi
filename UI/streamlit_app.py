import streamlit as st
import requests
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(
    page_title="🎬 Movie Review Sentiment Analysis 🎬",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="auto"
)

# Define the FastAPI URL endpoint
API_URL = "http://127.0.0.1:8000"
TOKEN_ENDPOINT = f"{API_URL}/token"
PREDICT_ENDPOINT = f"{API_URL}/predict/"

# Initialize session state for authorization and access token
if "authorized" not in st.session_state:
    st.session_state["authorized"] = False
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None

# Sidebar: Login and Authorization
st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login and Authorize"):
    # Request an access token from FastAPI using username and password
    response = requests.post(TOKEN_ENDPOINT, data={"username": username, "password": password})
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        st.sidebar.success("Logged in and authorized successfully")
        st.session_state["access_token"] = access_token
        st.session_state["authorized"] = True
    else:
        st.sidebar.error("Login failed. Please check your credentials.")

# Check if authorization is successful
if st.session_state["authorized"]:
    # Add a header image and styling
    st.markdown(
        """
        <style>
        .header {
            background-color: #1f1f1f;
            padding: 15px;
            text-align: center;
            color: #f0c929;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stApp {
            background: linear-gradient(to right, #141E30, #243B55);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="header"><h1>🎬 Movie Review Sentiment Analysis 🎥</h1></div>', unsafe_allow_html=True)

    # Display a movie-themed image
    image = Image.open("movies.jpg")
    st.image(image, caption='Let’s find out what people think about this movie!', use_column_width=True)

    # User input
    review_input = st.text_area("Enter a movie review:", "")

    # Create a button to make a prediction
    if st.button("🎬 Analyze Sentiment 🎬"):
        if not review_input:
            st.error("Please enter a review to analyze.")
        else:
            # Prepare the data payload
            payload = {
                "review": review_input  # Make sure the key matches the FastAPI expected key
            }

            headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}

            try:
                # Send a POST request to the FastAPI endpoint with the access token
                response = requests.post(PREDICT_ENDPOINT, json=payload, headers=headers)

                # Check if the request was successful
                if response.status_code == 200:
                    result = response.json()
                    # Display the results
                    sentiment_label = result.get("sentiment", "Unknown")
                    probability = result.get("Model Confidence Score", 0)

                    st.markdown(f"### 🎭 Sentiment Analysis Result 🎭")
                    if sentiment_label == "Positive":
                        st.markdown(
                            f"<div style='background-color: #28a745; padding: 10px; border-radius: 5px; text-align: center;'>"
                            f"<h3 style='color: white;'>{sentiment_label} (Confidence: {probability:.2f})</h3></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div style='background-color: #dc3545; padding: 10px; border-radius: 5px; text-align: center;'>"
                            f"<h3 style='color: white;'>{sentiment_label} (Confidence: {probability:.2f})</h3></div>",
                            unsafe_allow_html=True
                        )

                else:
                    st.error(f"Prediction failed. Status Code: {response.status_code}, Error: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the prediction service: {e}")

    # Footer
    st.markdown(
        """
        <hr style="border: 1px solid #f0c929;">
        <div style="text-align: center; color: #f0c929;">
            Made with ❤️ by AMMI Group II | Powered by Streamlit
        </div>
        """, unsafe_allow_html=True
    )
else:
    st.info("Please login and authorize from the sidebar to access the sentiment analysis.")
