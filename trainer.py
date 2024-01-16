import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    './flowers/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,  # Set a seed for reproducibility, diferent seed diferent result
    classes=['bougainvillea', 'daisies', 'gardenias', 'gardenroses', 'hibiscus','hydrangeas','lilies','orchids','peonies','tulip']  # Specify your classes
)


validation_generator = train_datagen.flow_from_directory(
    './flowers/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=['bougainvillea', 'daisies', 'gardenias', 'gardenroses', 'hibiscus','hydrangeas','lilies','orchids','peonies','tulip']  # Specify your classes
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') 
])


model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

loss, accuracy = model.evaluate(validation_generator)

model.save("flower_classification_model.h5")

# If you want to save only the model architecture (without weights and optimizer state)
model_json = model.to_json()
with open("flower_classification_model.json", "w") as json_file:
    json_file.write(model_json)

# If you want to save only the weights
model.save_weights("flower_classification_weights.h5")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
