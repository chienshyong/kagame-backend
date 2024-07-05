FROM python:3.10-slim

WORKDIR /app

# Install gcc, g++, and other necessary build dependencies
RUN apt-get update && \
    apt-get install -y gcc g++ && \
    apt-get clean

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
