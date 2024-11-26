import tensorflow as tf
import numpy as np

# Load your existing Keras model
model = tf.keras.models.load_model('chatbot_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('chatbot_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format!")

# To verify, create a mock input (or use actual input data if available)
# Example: Random array to simulate the correct shape
input_shape = model.input_shape[1:]  # Get the input shape (excluding the batch size)

# Ensure the mock input is of type float32 and has the correct shape
mock_input = np.zeros((1, *input_shape), dtype=np.float32)  # Add a batch dimension

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="chatbot_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the input tensor and run inference
interpreter.set_tensor(input_details[0]['index'], mock_input)
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)
