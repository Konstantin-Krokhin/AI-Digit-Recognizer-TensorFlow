import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# OpenCV to display images
def cv2_imshow(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

#model = keras.models.load_model('mnist_model.h5')



# Load MNIST data
try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
except Exception as e:
    print("MNIST download failed. Using fallback data...")
    with np.load('/mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

if os.path.exists('digit_model.keras'):
    print("Loading existing model...")
    model = keras.models.load_model('digit_model.keras')
else:
    print("Training new model...")
    # define and train your model here
    model = keras.Sequential([
    keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs = 2)

    model.save('digit_model.keras')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
print(f"Test Accuracy: {test_acc:.4f}")




#image_paths = sorted(glob.glob('/kaggle/input/handwritten-digits/*.png'))
dataset_path = "Digit Images Dataset/"

for i in range(1, 8):
    image_path = os.path.join(dataset_path, f"{i}.png")

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
                raise ValueError(f"Image not found at {image_path}")
        
        # 2. Auto-contrast (preserve gradients)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold the image to convert grayscale to binary (black/white)
        # Any pixel value > 128 becomes black (because of THRESH_BINARY_INV)
        # 3. Gentle thresholding (keep gray values)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find bounding box around the digit (non-zero pixels)
        x, y, w, h = cv2.boundingRect(img)
        
        # Crop the digit from the image using the bounding box
        img = img[y:y+h, x:x+w]
        cv2_imshow(img)
        
        # Resize: fit the digit into a 20-pixel box (height or width)
        # Keep the digit's shape (aspect ratio)
        h, w = img.shape
        if h > w:
            # Digit is tall: set height = 20, scale width proportionally
            img = cv2.resize(img, (int(w * 20 / h), 20), cv2.INTER_AREA)
        else:
            # Digit is wide or square: set width = 20, scale height proportionally
            img = cv2.resize(img, (20, int(h * 20 / w)), cv2.INTER_AREA)
        
        # Padding: center the digit in a 28x28 image
        # Calculate how much space to add around it
        pad_top = (28 - img.shape[0]) // 2
        pad_bottom = 28 - img.shape[0] - pad_top
        pad_left = (28 - img.shape[1]) // 2
        pad_right = 28 - img.shape[1] - pad_left
        
        # Add zero-value (black) padding on all sides to make image 28x28
        img = np.pad(img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0)
        
        # Normalize pixel values: from 0–255 to 0.0–1.0 (float range)
        img = img / 255.0

        #Predict and display
        prediction = model.predict(img.reshape(1, 28, 28, 1))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        print(f"\n\n\nPredicted: {predicted_digit} ({confidence:2f}% confidence)\n\n\n")
        
        # Show the processed image
        plt.imshow(img, cmap="gray")
        plt.title("Processed Image for Prediction")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
    
# Make a prediction
#prediction = model.predict(img.reshape(1, 28, 28, 1))
#predicted_digit = np.argmax(prediction)
#print(f"\n\n\nPredicted Digit: {predicted_digit}\n\n\n")