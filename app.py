import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageFilter
import io
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# --- CNN Model ---
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- Model Loader ---
def load_model(model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),  # Noise reduction
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# # --- Dataset Paths ---
# GRAPE_DATASET_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\collab\Dataset"
# --- Dataset Paths ---
GRAPE_DATASET_PATH = "Dataset/Grape"
MANGO_DATASET_PATH = "Dataset/Mango"
TULSI_DATASET_PATH = "Dataset/Tulsi"

# --- Model Paths ---
GRAPE_MODEL_PATH = "grape_final_model.pth"
MANGO_MODEL_PATH = "mango_leaf_disease_cnn6.pth"
TULSI_MODEL_PATH = "tul_leaf_disease_cnn.pth"


# # --- Model Paths ---
# GRAPE_MODEL_PATH = "grape_final_model.pth"
# MANGO_MODEL_PATH = "mango_leaf_disease_cnn6.pth"
# TULSI_MODEL_PATH = "tul_leaf_disease_cnn.pth"

# --- Class Labels ---
try:
    GRAPE_CLASSES = datasets.ImageFolder(GRAPE_DATASET_PATH, transform=transforms.ToTensor()).classes
except Exception as e:
    print(f"Error loading grape classes: {e}")
    GRAPE_CLASSES = []

MANGO_CLASSES = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
                 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# TULSI_CLASSES = ['bacterial', 'fungal', 'healthy', 'pests']
TULSI_CLASSES = [
    'bacterial',
    'fungal',
    'healthy',
    'pests',
    'class5',
    'class6',
    'class7',
    'class8'
]


# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/grape")
def grape_page():
    return render_template("grape.html")

@app.route("/mango")
def mango_page():
    return render_template("mango.html")

@app.route("/tulsi")
def tulsi_page():
    return render_template("tulsi.html")

@app.route("/predict/<plant_type>", methods=["POST"])
def predict(plant_type):
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)

        if plant_type == "grape":
            if not GRAPE_CLASSES:
                return jsonify({"error": "Grape class labels not loaded"}), 500
            model, device = load_model(GRAPE_MODEL_PATH, len(GRAPE_CLASSES))
            class_labels = GRAPE_CLASSES

        elif plant_type == "mango":
            model, device = load_model(MANGO_MODEL_PATH, len(MANGO_CLASSES))
            class_labels = MANGO_CLASSES

        elif plant_type == "tulsi":
            model, device = load_model(TULSI_MODEL_PATH, len(TULSI_CLASSES))
            class_labels = TULSI_CLASSES

        else:
            return jsonify({"error": "Invalid plant type"}), 400

        image = image.to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prediction = class_labels[predicted.item()]

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
