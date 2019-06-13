FROM python:alpine

COPY requirements.txt .

RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app