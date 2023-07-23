import numpy as np
import keras
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Set the paths to your dataset directories
train_dir = "C:/Users/aksha/OneDrive/Documents/glaucoma detection dataset/training"
val_dir = "C:/Users/aksha/OneDrive/Documents/glaucoma detection dataset/validation"
test_dir = "C:/Users/aksha/OneDrive/Documents/glaucoma detection dataset/test"

# Define the image size and other data augmentation parameters
img_size = (224, 224)
batch_size = 32

# Create data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load the DenseNet121 model without the top layer
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense layer on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the base DenseNet layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=SGD(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint]
)

# Evaluate the model on the test set
model.load_weights('model.h5')
score = model.evaluate(test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
