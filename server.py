import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Function to train and save the model
def train_model():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the data to the range [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Reshape the data to be 28x28x1 (for the model)
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # This model architecture has worked well in past
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),  # Flatten the image into a vector
        Dense(800, activation='relu'),  # First dense layer with 800 units
        Dense(10, activation='softmax')  # Output layer with 10 units (one for each digit)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=50, batch_size=64, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")

    # Save the trained model
    model.save('mnist_model.h5')

# Function to load the trained model for prediction
def load_model_for_prediction():
    try:
        # Load the pre-trained model
        model = tf.keras.models.load_model('mnist_model.h5')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model not found. Training new model...")
        train_model()
        return load_model_for_prediction()  # Retry loading after training

# Initialize the model (either load or train and save it)
model = load_model_for_prediction()

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the JSON request
        data = request.get_json()
        
        # Ensure the image data is in the correct format
        image = np.array(data['image'])

        if image is None or len(image) != 784:  # Image should be a flat array of 784 values (28x28)
            return jsonify({'error': 'Invalid image data. Ensure the image is a 28x28 pixel grayscale image.'})

        # Reshape the image to match the model's input shape (28x28x1)
        image = image.reshape(1, 28, 28, 1)

        # Make a prediction
        prediction = model.predict(image)

        # Get the predicted digit (the index with the highest probability)
        predicted_digit = np.argmax(prediction)

        return jsonify({'prediction': int(predicted_digit)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error during prediction. Please try again.'})

# Start the Flask app
if __name__ == '__main__':
    # If the model doesn't exist or hasn't been trained, train it
    if not tf.io.gfile.exists('mnist_model.h5'):
        train_model()
        model = load_model_for_prediction()
    app.run(debug=True)
