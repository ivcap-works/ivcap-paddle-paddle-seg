FROM python:3.9-slim-bullseye AS builder

# Install required systems libraries
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  libgomp1 ffmpeg libsm6 libxext6 \
  git && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*


WORKDIR /app
RUN pip install -U pip
COPY requirements.txt ./
RUN SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt

COPY infer_service.py predictor.py ./
RUN mv infer_service.py service.py

# VERSION INFORMATION
ARG GIT_TAG ???
ARG GIT_COMMIT ???
ARG BUILD_DATE ???

ENV IVCAP_SERVICE_VERSION $GIT_TAG
ENV IVCAP_SERVICE_COMMIT $GIT_COMMIT
ENV IVCAP_SERVICE_BUILD $BUILD_DATE

# Command to run
RUN mkdir -p /data/in /data/out
ENTRYPOINT ["python", "/app/service.py"]