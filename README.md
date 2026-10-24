# 🌿 Plant Disease Prediction Using Deep Learning and CNN

## 📌 Overview

Plant Disease Prediction Using Deep Learning and Convolutional Neural Networks (CNN) is a machine learning project designed to identify plant diseases from images of plant leaves. The project aims to support early disease detection and help improve crop health through image-based analysis.

Deep learning techniques, particularly CNNs, can learn visual patterns such as leaf spots, discoloration, and other disease symptoms to classify plant images.

## 🎯 Objectives

- Identify plant diseases using leaf images.
- Explore deep learning techniques for image classification.
- Verify and preprocess datasets before model development.
- Support early detection of plant diseases.
- Demonstrate the application of AI in smart agriculture.

## ✨ Features

- **Dataset Verification:** Check the dataset before processing.
- **Image Preprocessing:** Prepare images for deep learning.
- **CNN-Based Classification:** Intended to classify plant images using a convolutional neural network.
- **Agricultural Application:** Demonstrate how AI can assist in plant health monitoring.

> Note: The classification and prediction features require the corresponding model implementation and trained weights.

## 🛠️ Technologies Used

- Python
- Deep Learning
- Convolutional Neural Networks (CNN)
- Image Processing
- NumPy
- OpenCV or Pillow, depending on the implementation
- TensorFlow/Keras or PyTorch, depending on the model framework

## 📂 Project Structure

```text
Plant-Disease-Prediction-Using-DL-and-CNN/
├── Dataset Verification.py
├── Image_preprocessing.py
├── LICENSE
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/k-raghu24/Plant-Disease-Prediction-Using-DL-and-CNN.git
```

### 2. Navigate to the project directory

```bash
cd Plant-Disease-Prediction-Using-DL-and-CNN
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

Install the libraries required by your scripts:

```bash
pip install numpy pillow opencv-python
```

If the implementation uses TensorFlow/Keras, install TensorFlow as well:

```bash
pip install tensorflow
```

If it uses PyTorch, install the appropriate PyTorch version for your system instead.

## 🚀 Usage

### Step 1: Verify the dataset

```bash
python "Dataset Verification.py"
```

### Step 2: Preprocess images

```bash
python Image_preprocessing.py
```

### Step 3: Train and test the CNN model

Once the model training and prediction scripts are available, configure the dataset paths, train the CNN, and evaluate it using separate training and testing datasets.

### Step 4: Predict plant diseases

Provide a plant leaf image to the trained model and use its output to identify the predicted disease class.

> The exact training and prediction commands depend on the model scripts and dataset configuration in the project.

## 🧠 How It Works

1. **Image collection:** Collect images of healthy and diseased plant leaves.
2. **Dataset verification:** Inspect the dataset and its organization.
3. **Image preprocessing:** Resize and normalize images as required by the model.
4. **Model training:** Train a CNN to learn disease-related visual features.
5. **Prediction:** Pass an unseen leaf image to the trained model.
6. **Classification:** Return the predicted disease category.

## 📊 Applications

- Smart agriculture
- Early plant disease screening
- Crop health monitoring
- Agricultural research and education
- AI-assisted farming systems

## 🔮 Future Enhancements

- Add a complete CNN training and evaluation pipeline.
- Build a web application for uploading leaf images.
- Display prediction confidence scores.
- Support more plant species and disease categories.
- Integrate a mobile-friendly interface.
- Explore real-time plant disease detection.

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## 📜 License

This project includes a GNU Affero General Public License v3.0 (AGPL-3.0) license. Refer to the `LICENSE` file for the applicable terms.

## 👨‍💻 Author

**Raghuraman K**

GitHub: [@k-raghu24](https://github.com/k-raghu24)

## ⭐ Support

If you find this project useful, consider giving the repository a star on GitHub.
