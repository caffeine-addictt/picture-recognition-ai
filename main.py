import os
import cv2
import numpy
import matplotlib

import matplotlib.pyplot as plot
from tensorflow import keras


# Setup
training = False

overwritePrevious = False     
modelSaveName = 'save.model'  # OK to leave this alone

imageDirectory = 'images'     # Images have to be scaled to 32x32 (px) [only tested for .jpg]



# Fetch Training and Testing data
(trainingImages, trainingLabels), (testingImages, testingLabels) = keras.datasets.cifar10.load_data()

# Scale images to 0-1 instead of 255
trainingImages = trainingImages / 255.0
testingImages = testingImages / 255.0

# Fetch Names
with open('names.txt', 'r') as f:
  names = f.read().split(',')

for i in range(16):
  plot.subplot(4, 4, i+1)
  plot.xticks([])
  plot.yticks([])

  plot.imshow(trainingImages[i], cmap = matplotlib.colormaps['binary'])
  plot.xlabel(names[trainingLabels[i][0]])

plot.show()


trainingImages = trainingImages[:20000]
trainingLabels = trainingLabels[:20000]
testingImages = testingImages[:4000]
testingLabels = testingLabels[:4000]


model: keras.Model | None

if (
  (overwritePrevious) or
  (not os.path.isdir(os.path.join(os.getcwd(), modelSaveName)))
):
  model = keras.models.Sequential()

  # Add Layers to Model
  model.add(keras.layers.Conv2D(
    32,
    (3, 3),
    activation = 'relu',
    input_shape = (32, 32, 3)
  ))
  model.add(keras.layers.MaxPool2D((2, 2)))
  model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
  model.add(keras.layers.MaxPool2D((2, 2)))
  model.add(keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(64, activation = 'relu'))
  model.add(keras.layers.Dense(10, activation = 'softmax'))

  model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
  )

else:
  model = keras.models.load_model(modelSaveName)



if not model:
  print('Failed to load model')
  exit(1)


if training:
  model.fit(
    trainingImages,
    trainingLabels,
    epochs = 10,                                        # How many times it will see the same image
    validation_data = (testingImages, testingLabels)
  )

  loss, accuracy = model.evaluate(testingImages, testingLabels)
  print(f'Loss: {loss}')
  print(f'Accuracy: {accuracy}')

  model.save(modelSaveName)

else:
  imagePath = os.path.join(os.getcwd(), imageDirectory)
  if not os.path.isdir(imagePath):
    print(f'No image directory found matching {imagePath} Making one...')
    os.mkdir(imagePath)
    print(f'Created at {imagePath}', end='\n\n')

  print('Looking for files...')
  filesFound = []
  for file in os.listdir(imagePath):
    filename = os.fsdecode(file)


  for i, filename in enumerate(filesFound):
    img = cv2.imread(os.path.join(imageDirectory, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    prediction = model.predict(numpy.divide(numpy.array([img]), 255))
    index = numpy.argmax(prediction)

    plot.subplot(4, 4, i+1)
    plot.xticks([])
    plot.yticks([])

    plot.imshow(img, cmap = matplotlib.colormaps['binary'])
    plot.xlabel(f'{names[index]} @ {filename}')

  plot.show()