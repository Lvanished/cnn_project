import tensorflow as tf

def update_model():
    model = tf.keras.models.load_model('app/model.h5')
    new_data = tf.keras.preprocessing.image_dataset_from_directory(
        'train/data/train', image_size=(128, 128), batch_size=32)
    
    model.fit(new_data, epochs=5)
    model.save('app/model.h5')
    print("模型已更新并保存！")

if __name__ == "__main__":
    update_model()
