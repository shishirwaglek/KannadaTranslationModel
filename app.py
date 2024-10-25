from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import pickle
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__, template_folder='./frontend/')
CORS(app)  # Enable CORS to allow cross-origin requests

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML file

# Define parameters
SOS_token = 0
EOS_token = 1
UNK_token = 2
MAX_LENGTH = 100
hidden_size = 250

# Language Class Definition
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS, EOS, and UNK tokens

# Encoder Definition
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.layers, 1, self.hidden_size, device=device),
                torch.zeros(self.layers, 1, self.hidden_size, device=device))

# Attention Decoder Definition
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH, layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.layers = layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(self.layers, 1, self.hidden_size, device=device),
                torch.zeros(self.layers, 1, self.hidden_size, device=device))

# Load environment variables from .env file
load_dotenv()
AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Define model paths in Azure Blob Storage
model_container = "mlmodels"  # Replace with your Azure Blob container name
encoder_path = "encoder.pth"
decoder_path = "decoder.pth"
input_lang_path = "input_lang.pkl"
output_lang_path = "output_lang.pkl"

# Local storage path
LOCAL_MODEL_DIR = "/home/site/wwwroot/models/"

# Ensure the directory exists
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Function to download a blob if it doesn't exist locally
def download_blob_if_not_exists(blob_name, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {blob_name} from Azure Blob Storage...")
        blob_client = blob_service_client.get_blob_client(container=model_container, blob=blob_name)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
        print(f"{blob_name} downloaded and saved to {local_path}.")
    else:
        print(f"{blob_name} already exists at {local_path}.")

# Load model components from Azure Blob Storage
def load_model():
    # Download model files if they don't exist
    download_blob_if_not_exists(encoder_path, os.path.join(LOCAL_MODEL_DIR, encoder_path))
    download_blob_if_not_exists(decoder_path, os.path.join(LOCAL_MODEL_DIR, decoder_path))
    download_blob_if_not_exists(input_lang_path, os.path.join(LOCAL_MODEL_DIR, input_lang_path))
    download_blob_if_not_exists(output_lang_path, os.path.join(LOCAL_MODEL_DIR, output_lang_path))

    # Initialize language objects before loading them
    input_lang = Lang("English")
    output_lang = Lang("Kannada")

    # Load language dictionaries from pickle files
    with open(os.path.join(LOCAL_MODEL_DIR, input_lang_path), 'rb') as f:
        input_lang = pickle.load(f)

    with open(os.path.join(LOCAL_MODEL_DIR, output_lang_path), 'rb') as f:
        output_lang = pickle.load(f)

    # Load the encoder and decoder using the loaded language objects
    encoder = EncoderRNN(len(input_lang.word2index), hidden_size)
    encoder.load_state_dict(torch.load(os.path.join(LOCAL_MODEL_DIR, encoder_path), map_location=device))
    encoder.eval()  # Set to evaluation mode

    decoder = AttnDecoderRNN(hidden_size, len(output_lang.word2index))
    decoder.load_state_dict(torch.load(os.path.join(LOCAL_MODEL_DIR, decoder_path), map_location=device))
    decoder.eval()  # Set to evaluation mode

    return encoder, decoder, input_lang, output_lang

# Load models and languages
encoder, decoder, input_lang, output_lang = load_model()

# Move the models to the correct device
encoder.to(device)
decoder.to(device)

# Evaluate function for translation
def evaluate(encoder, decoder, sentence):
    input_tensor = tensorFromSentence(input_lang, sentence)
    encoder_hidden = encoder.initHidden()

    # Move input tensor to the device
    input_tensor = input_tensor.to(device)
    
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    for ei in range(input_tensor.size(0)):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH, device=device)  # Move to the same device

    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        if topi.item() == EOS_token:
            break
        else:
            decoded_words.append(output_lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()  # Next input is the current output
        decoder_input = decoder_input.to(device)  # Ensure it's on the right device

    return decoded_words

# Function to convert sentence to tensor
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index.get(word, UNK_token) for word in sentence.split(' ')]  # Use UNK for unknown words
    indexes.append(EOS_token)  # Add EOS token at the end
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Define translation route
@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        sentence = data['sentence']  # Ensure it matches the frontend key
        translation = evaluate(encoder, decoder, sentence)
        return jsonify({"translation": ' '.join(translation)})  # Return under 'translation'

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
