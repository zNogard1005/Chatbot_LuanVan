# Thể hiện mô hình
from tensorflow.keras.models import load_model

model = load_model('templates/chatbot_model.h5')

# To get the name of layers in the model.
layer_names=[layer.name for layer in model.layers]

# for model's summary and details.
model.summary()