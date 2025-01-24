from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# 初始化 FastAPI 应用
app = FastAPI()

# 加载预训练的 CNN 模型
try:
    model = load_model("model.h5")
except Exception as e:
    raise RuntimeError(f"无法加载模型: {str(e)}")

# 图像预处理函数
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # 假设模型接受 224x224 输入
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# 预测接口
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="文件不是图像格式")
    
    try:
        # 打开并预处理图像
        image = Image.open(file.file)
        input_data = preprocess_image(image)
        
        # 模型预测
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return {"filename": file.filename, "predicted_class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
