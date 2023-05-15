FROM python:3.9-slim

COPY . /app

WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 7860

CMD python main.py