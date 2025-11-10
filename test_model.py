# Week 2: Test script placeholder

import tensorflow as tf

# Load trained model (to be used in Week 3)
model_path = 'outputs/saved_model.h5'

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully. Ready for evaluation.")
except:
    print("No model found yet. Testing will begin in Week 3.")
