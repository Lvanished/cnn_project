from fastapi import FastAPI, UploadFile, Form
from pathlib import Path
import shutil
import subprocess
import os

app = FastAPI()

# 配置
TRAIN_DATA_DIR = "train/data/train"
MODEL_PATH = "app/model.h5"
GIT_REPO_DIR = "/path/to/your/project"
BRANCH_NAME = "main"

# 确保类别目录存在
def create_category_folder(category: str):
    category_dir = Path(TRAIN_DATA_DIR) / category
    category_dir.mkdir(parents=True, exist_ok=True)
    return category_dir

# 更新 GitHub
def update_github():
    try:
        os.chdir(GIT_REPO_DIR)
        subprocess.run(["git", "add", "train/data/train"], check=True)
        subprocess.run(["git", "commit", "-m", "Update training data and retrain model"], check=True)
        subprocess.run(["git", "push", "origin", BRANCH_NAME], check=True)
        print("GitHub 已更新！")
    except subprocess.CalledProcessError as e:
        print(f"Git 操作失败: {e}")

# 更新模型
def update_model():
    try:
        subprocess.run(["python", "train/update_model.py"], check=True)
        print("模型已更新！")
    except subprocess.CalledProcessError as e:
        print(f"模型更新失败: {e}")

@app.post("/upload/")
async def upload_image(file: UploadFile, category: str = Form(...)):
    """
    接收图片并保存到训练集目录，触发模型更新和 GitHub 推送。
    """
    try:
        # 确保类别目录存在
        category_dir = create_category_folder(category)

        # 保存文件
        file_path = category_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 更新模型和 GitHub
        update_model()
        update_github()

        return {"message": "文件上传成功，模型和 GitHub 已更新", "path": str(file_path)}
    except Exception as e:
        return {"error": str(e)}
