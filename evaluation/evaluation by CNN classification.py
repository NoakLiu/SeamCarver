import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image


# getting data
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_ape = os.path.join(train_dir, 'ape')
train_human = os.path.join(train_dir, 'human')
validation_ape = os.path.join(validation_dir, 'ape')
validation_human = os.path.join(validation_dir, 'human')

num_ape_tr = len(os.listdir(train_ape))
num_human_tr = len(os.listdir(train_human))
num_ape_val = len(os.listdir(validation_ape))
num_human_val = len(os.listdir(validation_human))

total_train = num_ape_tr + num_human_tr
total_val = num_ape_val + num_human_val

#initialising data
BATCH_SIZE = 32
IMG_SHAPE = 300 # square image
EPOCHS = 20 # initial EPOCHE if user don't inter its EPOCHS
ACC=float(input ("accuracy(0 => base on initial EPOCHS ):")) # desired user EPOCHS, 0 = initial EPOCHS mean: EPOCHS=5.

#generators

#prevent memorization
train_image_generator = ImageDataGenerator(
    rescale=1./255,    
    )

validation_image_generator = ImageDataGenerator(
    rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=validation_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')


#showing one sample of training images
xtbatches , ytbatches = next(train_data_gen)
for i in range(0,3):
    image1 = xtbatches[i]
    plt.imshow(image1)
    plt.show()

#showing one sample of validation images
xvbatches , yvbatches = next(val_data_gen )
for i in range(0,3):
    image1 = xvbatches[i]
    plt.imshow(image1)
    plt.show()

#callback
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.001,
                                         patience=9, mode="min", baseline=ACC)

# model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)), # RGB
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5), # 1/2 of neurons will be turned off randomly
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(2, activation='softmax') #[0, 1] or [1, 0]  also you can use ""sigmoid"" function 

    ])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# if we don't have progress, running code will stop
if ACC == 0 :
  history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )
else: 
  history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
    callbacks=[callback]
    )


# analysis

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = input("Please enter runed epochs :")
epochs_range = range(int(epochs))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("result1.png")
plt.show()



#Running the Model


img_path = 'test/ape/ape101.png' # input your file path

img = load_img(img_path, target_size=(IMG_SHAPE, IMG_SHAPE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)

# announce class of desired image
print(classes[0][0])
if classes[0][0]>0.5:
    print(" your image is a ape ")
else:
    print("your image is a human ")


#Visualizing Intermediate Representations (optional)

successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
train_human_names = os.listdir(train_human)
train_ape_names = os.listdir(train_ape)
human_img_files = [os.path.join(train_human, f) for f in train_human_names]
ape_img_files = [os.path.join(train_ape, f) for f in train_ape_names]
img_path = random.choice(human_img_files )
img = load_img(img_path, target_size=(IMG_SHAPE, IMG_SHAPE))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
x /= 255 # Rescale by 1/255
# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)
# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]
# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    #plt.figure(figsize=(scale * n_features, scale))
    #plt.title(layer_name)
    #plt.grid(False)
    #plt.imshow(display_grid, aspect='auto', cmap='viridis')
#plt.figure()
#plt.show()
    
