import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("arthricare/models/image_model.h5")

img = image.load_img("test_image.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

classes = ["normal", "osteoarthritis", "rheumatoid"]

print("Prediction:", classes[np.argmax(prediction)])