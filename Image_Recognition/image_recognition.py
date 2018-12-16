import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50

# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = resnet50.ResNet50()

# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img('bay.jpg', target_size=(224, 224))
'''
Image size is 1365x1365 pixels, which is too large to feed directly into the NN
Size of image needs to match the number of input nodes in the NN
For ResNet50, the image needs to be 224x224 pixels
Let's resize it
Convert the image to a numpy array
'''
x = image.img_to_array(img)
'''
This turns our image into a 3 dimensional array
The first 2 dimensions are the height and width of the image and 3rd is color
Each pixel is made up of RGB values
Our array will be 3 layers deep, with each layers representing how intense the RGB
Add a forth dimension since Keras expects a list of images
'''
x = np.expand_dims(x, axis=0)
# Each value can range from 0 to 255
# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)
'''
This will return a predictions object
The predictions object is a 1,000 element array of floating point numbers
Each element in the array tells us how likely our picture contains
each of 1,000 objects the model is trained to recognize
Look up the names of the predicted classes. Index zero is the results for the first image.
'''
predicted_classes = resnet50.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))

