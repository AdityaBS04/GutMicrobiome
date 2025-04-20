
from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import joblib
import numpy as np

# Encoder-Decoder Model Definition
class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Image Models
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in list(self.resnet.parameters())[:-6]:
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)

class DiseaseClassifier(nn.Module):
    def __init__(self):  # Removed num_classes parameter
        super(DiseaseClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for param in list(self.resnet.parameters())[:-6]:
            param.requires_grad = False
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)  # Hardcoded to 3 classes
        )
    
    def forward(self, x):
        return self.resnet(x)

class TextProcessor:
    def __init__(self, input_dim=5, hidden_dim=128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.models = []
        self.clf = None
        self.scaler = None
        self.disease_mapping = {
            0: "Ulcerative Colitis",
            1: "Crohn Disease",
            2: "Colorectal Neoplasms",
            3: "Cystic Fibrosis",
            4: "Liver Cirrhosis",
            5: "Parkinson's Disease"
        }
        self.load_components()

    def load_components(self):
        try:
            # Load models
            for i in range(3):  # We have 3 encoder-decoder models
                model = EncoderDecoder(self.input_dim, self.hidden_dim)
                model.load_state_dict(torch.load(f'encoder_decoder_{i}.pth'))
                model.eval()
                self.models.append(model)

            # Load classifier and scaler
            self.clf = joblib.load('classifier.joblib')
            self.scaler = joblib.load('scaler.joblib')
        except Exception as e:
            print(f"Error loading components: {e}")
            raise

    def predict_disease_probabilities(self, user_input):
        try:
            # Convert input to correct format
            user_input = np.array(user_input, dtype=np.float32)
            if len(user_input) < self.input_dim:
                user_input = np.pad(user_input, (0, self.input_dim - len(user_input)), mode='constant')
            user_input = user_input.reshape(1, -1)

            # Scale input
            user_input_scaled = self.scaler.transform(user_input)
            user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

            # Get encoded features from all models
            encoded_features = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    encoded, _ = model(user_input_tensor)
                encoded_features.append(encoded.numpy())

            # Stack features and get predictions
            encoded_features = np.hstack(encoded_features)
            probabilities = self.clf.predict_proba(encoded_features)[0]

            # Create results dictionary with all diseases
            results = {disease: 0.0 for disease in self.disease_mapping.values()}
            for idx, prob in enumerate(probabilities):
                if idx in self.disease_mapping:
                    results[self.disease_mapping[idx]] = float(prob)

            # Get predicted class
            predicted_class = self.disease_mapping[np.argmax(probabilities)]
            confidence = float(np.max(probabilities))

            return {
                'result': 'Analysis Complete',
                'disease_type': predicted_class,
                'confidence': f"{confidence:.2%}",
                'probabilities': {disease: f"{prob:.2%}" for disease, prob in results.items()}
            }
        except Exception as e:
            return {'error': 'Error processing input', 'details': str(e)}

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
try:
    binary_model = BinaryClassifier().to(device)
    disease_model = DiseaseClassifier().to(device)  # Removed num_classes argument
    binary_model.load_state_dict(torch.load('identification_model.pth', map_location=device))
    disease_model.load_state_dict(torch.load('classification_model.pth', map_location=device))
    binary_model.eval()
    disease_model.eval()
except Exception as e:
    print(f"Error loading image models: {e}")

# Initialize text processor with appropriate input dimension
text_processor = TextProcessor(input_dim=5)  # Adjust input_dim based on your data

def process_image(image_data):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            binary_output = binary_model(input_tensor).item()
            is_disease = binary_output < 0.5

        if not is_disease:
            return {'result': 'Normal', 'confidence': f"{binary_output:.2%}"}

        with torch.no_grad():
            disease_output = disease_model(input_tensor)
            probabilities = torch.softmax(disease_output, dim=1).squeeze(0).tolist()
            diseases = ['Esophagitis', 'Polyps', 'Ulcerative Colitis']
            predicted_idx = np.argmax(probabilities)

        return {
            'result': 'Disease Detected',
            'disease_type': diseases[predicted_idx],
            'confidence': f"{max(probabilities):.2%}",
            'probabilities': {disease: f"{prob:.2%}" for disease, prob in zip(diseases, probabilities)}
        }
    except Exception as e:
        return {'error': 'Error processing image', 'details': str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    return jsonify(process_image(file.read()))

@app.route('/predict/text', methods=['POST'])
def predict_text():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({'error': 'No input values provided'}), 400

        text_input = data["text"].strip()
        
        # Convert input text (comma-separated values) into a list of floats
        try:
            values = [float(x.strip()) for x in text_input.split(",")]
        except ValueError:
            return jsonify({'error': 'Invalid input format. Please enter numerical values separated by commas.'}), 400

        # Make prediction
        result = text_processor.predict_disease_probabilities(values)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': 'Error processing request', 'details': str(e)}), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True)
