import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据路径
train_dir = "data/train"
test_dir = "data/test"

# 数据增强
datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)

train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")

# 构建 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 训练模型
model.fit(train_generator, validation_data=test_generator, epochs=10)
model.save("model.h5")
