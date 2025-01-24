import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载旧模型
model = load_model("model.h5")

# 数据路径
train_dir = "data/train"

# 数据增强
datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")

# 增量训练
model.fit(train_generator, epochs=5)
model.save("model.h5")
