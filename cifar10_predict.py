import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from PIL import Image
import cv2

# load model
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
model.summary()
# load images
img_names = ['cat.png', 'dog.png']
imgs = [np.array(Image.open(img_name, formats=("PNG",)).getdata()).reshape(32, 32, 3).astype('float32')for img_name in img_names]


imgs = np.array(imgs) / 255

# train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,metrics=['accuracy'])
# predict
predictions = model.predict(imgs)
print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))