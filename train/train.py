import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'train/data/train', image_size=(128, 128), batch_size=32)
    
    model = create_model()
    model.fit(train_dataset, epochs=10)
    model.save('app/model.h5')
    print("模型训练完成并保存！")

if __name__ == "__main__":
    train_model()
