FROM python:3.9-slim-buster

# Install required systems libraries
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  libgomp1 ffmpeg libsm6 libxext6 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt

ADD export.py .

ENTRYPOINT ["python", "/app/export.py", "--save-path", "/output/model.tgz"]
