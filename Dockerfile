FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY loan_model.py .
COPY loan_prediction.csv .

EXPOSE 5000

CMD ["sh", "-c", "python loan_model.py && python -u app.py"]
