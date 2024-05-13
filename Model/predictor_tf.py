import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms



classes = ['cardboard','glass','metal','paper','plastic','trash']
transformations = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="garbage_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.


# Test model on random input data.




# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.


def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))
    example_image = transformations(image)
    input_data = np.array(example_image).reshape([1,224,224,3])

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = np.array(interpreter.get_tensor(output_details[0]['index'])).reshape((6,))
    print(output_data)
    print("The image resembles", classes[np.argmax(output_data)])

predict_external_image('./plastic.jpg')