import numpy as np
import torch
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from azure.storage.blob import BlobServiceClient
from nltk.corpus import stopwords
# import os
from dotenv import load_dotenv
import os
load_dotenv()

# Azure storage connection string and tokenizer URL
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
TOKENIZER_URL = os.getenv("TOKENIZER_URL")
MAX_LENGTH = 200

# Initialize Azure Blob Storage client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Function to read blob content into memory
def read_blob_to_memory(container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall()

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Download the tokenizer from Azure Blob Storage
tokenizer_blob_name = TOKENIZER_URL.split('/')[-1]  # Extract the blob name from the URL
tokenizer_json_bytes = read_blob_to_memory("project2", tokenizer_blob_name)

# Decode the bytes and load the tokenizer
tokenizer_json_str = tokenizer_json_bytes.decode('utf-8')  # Decode bytes to string
tokenizer = json.loads(tokenizer_json_str)  # Parse JSON

# Text preprocessing functions
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase the text
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english')])  # Remove stopwords
    return text

def text_to_sequence(text):
    return [tokenizer.get(word, 0) for word in word_tokenize(text)]

def pad_sequence(sequence, maxlen=int(MAX_LENGTH)):
    return np.array(np.pad(sequence, (0, max(0, maxlen - len(sequence))), 'constant')[:maxlen])

def preprocess_text(text):
    text = clean_text(text)
    sequence = text_to_sequence(text)
    padded_seq = pad_sequence(sequence)
    padded_seq = torch.tensor(padded_seq, dtype=torch.long).unsqueeze(0)
    return padded_seq