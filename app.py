from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": ["http://localhost:3000", "https://advaithmalka.github.io/cop-classifier/"]}})

@app.route("/")
def index():
    return "<p>Welcome the Cop Classification API</p>"


DEVICE = 'cpu'
# Load the pre-trained model
model = torch.load("models/Cop_ClassifierV3_20_epochs.pt").to(device=DEVICE)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error='No file provided'), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0).to(device=DEVICE)

    with torch.inference_mode():
        output = model(img)
        prediction = torch.sigmoid(output).item() < 0.5

    return jsonify(prediction=prediction)

