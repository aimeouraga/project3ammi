# Use the official Python image from Docker Hub
FROM python:3.11.5-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and .env file
COPY requirements.txt ./
# COPY .env ./


# Install the dependencies
RUN pip install --default-timeout=1000 --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy only the necessary directories and files
COPY app/ ./app/
COPY auth/ ./auth/
COPY data/ ./data/
COPY model/ ./model/
COPY movies.jpg ./
COPY UI/ ./UI/

# Expose the ports for FastAPI:8000 and Streamlit:8501
EXPOSE 8000
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to start Streamlit first, then FastAPI
# CMD ["sh", "-c", "streamlit run streamlit.py --server.port 8501 --server.address 0.0.0.0 & sleep 5 && uvicorn app.app:app --host 0.0.0.0 --port 8000"]
CMD ["sh", "-c", "streamlit run UI/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 & sleep 5 && uvicorn app.app:app --host 0.0.0.0 --port 8000"]

