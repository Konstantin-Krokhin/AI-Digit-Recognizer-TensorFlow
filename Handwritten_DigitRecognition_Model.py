import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow import keras

# OpenCV to display images
def cv2_imshow(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#model = keras.models.load_model('mnist_model.h5')

# Load your image (change 'your_image.png' to your actual file)
image_path = "d.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

# Resize to 28x28 (same as MNIST dataset)
img = cv2.resize(img, (28, 28))

# Invert colors if needed (MNIST digits are black on white)
img = 255 - img  # Invert if the digit appears white on black

# Normalize (0-255 → 0-1) and reshape for model input
img = img / 255.0
#img = img.reshape(1, 28, 28, 1)  # Add batch and channel dimension

# Show the processed image
plt.imshow(img, cmap="gray")
plt.title("Processed Image for Prediction")
plt.axis("off")

# Load MNIST data
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

if os.path.exists('digit_model.keras'):
    print("Loading existing model...")
    model = keras.models.load_model('digit_model.keras')
else:
    print("Training new model...")
    # define and train your model here
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=2)
    model.save('digit_model.keras')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Make a prediction
prediction = model.predict(np.expand_dims(img, axis=0))
predicted_digit = np.argmax(prediction)

print(f"Predicted Digit: {predicted_digit}")
