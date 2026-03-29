import torch
import librosa
import numpy as np
import joblib
from transformers import pipeline, AutoTokenizer, AutoModel
import whisper
import spacy

# Import your model class
from model import build_model, BERT_DIM, NUMERIC_DIM

# 1. Load your trained artifacts globally so they don't reload on every request
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(device=DEVICE)
model.load_state_dict(torch.load("processed/best_model.pt", map_location=DEVICE))
model.eval()

scaler = joblib.load("processed/scaler.pkl")
imputer = joblib.load("processed/imputer.pkl")

# Load heavy feature extractors once
whisper_model = whisper.load_model("base")
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)

def process_single_audio(file_path: str):
    """Runs all 4 phases on a single audio file and returns the prediction."""
    
    # Phase 1: Transcribe
    result = whisper_model.transcribe(file_path)
    transcript = result["text"]
    
    # Phase 2 & 3: Feature Extraction
    # Note: You will need to call your specific functions from dataset_loader/feature_extractor here
    # Example placeholder:
    # linguistic_feats = extract_lexical_features(transcript)
    # acoustic_feats = extract_mel_features(file_path)
    # numeric_array = combine_into_43_dim_array(linguistic_feats, acoustic_feats)
    
    # For this example, let's assume you've structured the 43 numeric features into a 1D numpy array
    numeric_features = np.zeros(43) # Replace with actual feature extraction calls
    
    # Phase 4: BERT Vectorization
    inputs = tokenizer(transcript, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        bert_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    # Scale and Impute Numeric Features
    numeric_features = numeric_features.reshape(1, -1)
    numeric_features = imputer.transform(numeric_features)
    numeric_features = scaler.transform(numeric_features)
    
    # Convert to Tensors
    bert_tensor = torch.tensor(bert_embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        logits = model(bert_tensor, numeric_tensor)
        probability = torch.sigmoid(logits).item()
        
    return probability