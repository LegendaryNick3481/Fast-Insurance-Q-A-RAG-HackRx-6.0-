FROM python:3.10-slim

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get install -y build-essential gcc \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
