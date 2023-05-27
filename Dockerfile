# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:23.04-py3
#FROM python:3.10-bullseye
EXPOSE 8888

WORKDIR /workspace

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
RUN pip3 install torch torchaudio torchvision 
RUN pip3 install -r requirements.txt
CMD ["python3", "main.py"]