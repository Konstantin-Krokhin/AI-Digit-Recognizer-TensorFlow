# 🧠 Handwritten Digit Recognizer (TensorFlow)

This is a simple deep learning project that trains a neural network to recognize handwritten digits using the MNIST dataset. Built with TensorFlow and Keras, it serves as a minimal and educational introduction to image classification.

## 📌 Features

- Uses the **MNIST dataset** of handwritten digits (28x28 grayscale images)
- Builds and trains a **fully connected neural network**
- Saves the trained model to disk
- Includes simple test/evaluation of model performance

---

## 📂 Project Structure:

AI_Digit_Recognizer_TensorFlow/
├── Digit Images Dataset/ # (Optional) Custom digit images (not used in training)
├── Digit Images Dataset.zip # Zipped version of the above folder
├── digit_model.keras # Trained model (saved after training)
├── Handwritten_Digit_Recognizer_Model.py # Main training script
├── mnist.npz # Cached MNIST dataset (auto-downloaded by Keras)
├── venv/ # Local Python virtual environment
├── .git/ # Git repo metadata
└── README.md # Project readme (this file)


---

## 🧰 Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy

Install requirements manually:

```bash
pip install tensorflow numpy


🚀 Usage
Clone the repo:
git clone https://github.com/Konstantin-Krokhin/AI-Digit-Recognizer-TensorFlow.git
cd AI-Digit-Recognizer-TensorFlow
Run the script:
python main.py
This will:
Load and preprocess the MNIST dataset
Build and train the model
Evaluate it on the test set
Save the trained model as digit_model.h5
🧠 Model Architecture
Input: 784 neurons (flattened 28x28 image)
Hidden layers: Two dense layers with ReLU activation
Output: 10 neurons (digits 0–9) with softmax
📈 Performance
Expected test accuracy after a few epochs: ~97–98%
📝 License
This project is licensed under the MIT License.
👤 Author
Konstantin Krokhin
🔗 GitHub Profile
