# 🌿 Plant Disease Detection System Using Deep Learning

A **deep learning–based web application** for detecting plant leaf diseases using **PyTorch CNN models** and a **Flask** backend. The system allows users to upload leaf images and instantly receive disease predictions, supporting **smart and precision agriculture**.

Currently supported plants:

* 🍇 **Grape**
* 🥭 **Mango**
* 🌿 **Tulsi (Holy Basil)**

---

## ✨ Key Features

* Web-based interface for image upload and prediction
* Separate CNN models for different plant species
* Real-time inference using PyTorch
* Flask REST API for predictions
* Clean and responsive UI using HTML & CSS
* Image preprocessing with noise reduction (Gaussian Blur)

---

## 🧠 Model Architecture

Each plant uses a **custom Convolutional Neural Network (CNN)** with:

* 3 convolutional layers (32 → 64 → 128 filters)
* ReLU activation and MaxPooling
* Fully connected layers for classification
* Input image size: **128 × 128 RGB**

The models are trained separately for each plant type and loaded dynamically during prediction.

---

## 🛠️ Tech Stack

* **Python 3**
* **PyTorch** – deep learning framework
* **Flask** – backend web framework
* **HTML / CSS / JavaScript** – frontend
* **Pillow (PIL)** – image handling & preprocessing
* **Torchvision** – image transforms

---

## 📁 Project Structure

```
Plant-Disease-Detection-System-Using-Deep-Learning/
│── app.py                  # Flask app & prediction logic
│── README.md
│── requirements.txt
│── .gitignore
│
├── templates/               # Frontend HTML pages
│   ├── index.html           # Home page
│   ├── grape.html           # Grape disease page
│   ├── mango.html           # Mango disease page
│   └── tulsi.html           # Tulsi disease page
│
├── notebooks/               # Model training notebooks
│   ├── Grape.ipynb
│   ├── Mango.ipynb
│   └── Tulsi.ipynb
│
├── models/                  # Trained models (recommended)
│   ├── grape_model.pth
│   ├── mango_model.pth
│   └── tulsi_model.pth
│
├── Dataset/                 # Training datasets
│   ├── Grape/
│   ├── Mango/
│   └── Tulsi/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/NoyalSam/Plant-Disease-Detection-System-Using-Deep-Learning.git
cd Plant-Disease-Detection-System-Using-Deep-Learning
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🚀 How the System Works

1. User selects a plant type (Grape / Mango / Tulsi)
2. Uploads a leaf image through the web UI
3. Image is preprocessed (blur, resize, normalization)
4. Corresponding CNN model is loaded
5. Model predicts the disease class
6. Result is displayed on the web page

---

## 📊 Supported Classes

### 🍇 Grape

* Black Rot
* Esca (Black Measles)
* Leaf Blight (Isariopsis Leaf Spot)
* Healthy

### 🥭 Mango

* Anthracnose
* Bacterial Canker
* Cutting Weevil
* Die Back
* Gall Midge
* Powdery Mildew
* Sooty Mould
* Healthy

### 🌿 Tulsi

* Bacterial
* Fungal
* Pests
* Healthy

---

## 🔮 Future Enhancements

* Add more plant species
* Improve accuracy with larger datasets
* Deploy on cloud (Render / AWS / Railway)
* Add camera-based real-time detection
* Confidence score for predictions

---

## 🎓 Academic Use

This project is suitable for:

* Final-year / capstone projects
* Machine learning & deep learning demonstrations
* Smart agriculture research

---

## 👨‍💻 Author

**Noyal Sam**

