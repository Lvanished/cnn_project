FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-multipart

COPY . /app

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
