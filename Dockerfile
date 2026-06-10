FROM python:3.12-slim

WORKDIR /app

COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY  app ./app
COPY scripts ./scripts
COPY ui ./ui

RUN mkdir -p \
data/raw/uploads \
data/processed \
vectorstore/UserUploads

EXPOSE 8000

CMD ["python","-m","uvicorn","app.main:app", "--host", "0.0.0.0", "--port", "8000"]