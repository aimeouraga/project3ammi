import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import io

# Load environment variables
load_dotenv()

# Azure Blob Storage connection
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
MODEL_URL=os.getenv("MODEL_URL")
TOKENIZER_URL=os.getenv("TOKENIZER_URL")
EMBEDDING_URL=os.getenv("EMBEDDING_URL")

# Initialize Azure Blob Storage client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Function to read blob content into memory
def read_blob_to_memory(blob_url):
    blob_client = blob_service_client.get_blob_client(container="project2", blob=blob_url.split('/')[-1])
    return blob_client.download_blob().readall()

# Define the model architecture (ensure it matches the training architecture)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, n_layers=2, embedding_matrix=None):
        super(SentimentLSTM, self).__init__()

        # Load pre-trained embeddings
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Adding an extra dense layer
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last timestep
        x = F.dropout(F.relu(self.fc1(lstm_out)), p=0.3)  # Apply dropout with probability 0.3
        x = self.fc2(x)
        return self.sigmoid(x)

# Function to load the model, tokenizer, and embedding matrix
def load_model():
    # Download and read the tokenizer from Azure Blob Storage
    tokenizer_blob = read_blob_to_memory(TOKENIZER_URL)
    tokenizer = json.loads(tokenizer_blob.decode("utf-8"))

    # Download and read the embedding matrix from Azure Blob Storage
    embedding_blob = read_blob_to_memory(EMBEDDING_URL)
    embedding_matrix = np.load(io.BytesIO(embedding_blob), allow_pickle=True)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    # Initialize the model with the embedding matrix
    vocab_size = len(tokenizer) + 1
    model = SentimentLSTM(vocab_size, embedding_dim=embedding_matrix.shape[1], embedding_matrix=embedding_matrix)
    
    # Download and read the model weights from Azure Blob Storage
    model_blob = read_blob_to_memory(MODEL_URL)
    model.load_state_dict(torch.load(io.BytesIO(model_blob), weights_only=True, map_location=torch.device('cpu')))
    
    # Set device and return model
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    return model, device

# Function to make predictions
def predict_sentiment(model, device, input_tensor):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()  # Get the probability score from the model
        
        # Determine the sentiment label and confidence score
        if probability >= 0.5:
            sentiment_label = "Negative"
            confidence_score = probability
        else:
            sentiment_label = "Positive"
            confidence_score = 1 - probability

    return sentiment_label, confidence_score
