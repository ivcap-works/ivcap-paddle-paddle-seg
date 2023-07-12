SERVICE_CONTAINER_NAME=paddle-paddle-serving
SERVICE_TITLE=PaddlePaddle Servicing

SERVICE_FILE=infer_service.py

PROVIDER_NAME=ivcap.test

# don't foget to login 'az acr login --name cipmain'
AZ_DOCKER_REGISTRY=cipmain.azurecr.io
GKE_DOCKER_REGISTRY=australia-southeast1-docker.pkg.dev/reinvent-science-prod-2ae1/ivap-registry
MINIKUBE_DOCKER_REGISTRY=localhost:5000
DOCKER_REGISTRY=${GKE_DOCKER_REGISTRY}

SERVICE_ID:=ivcap:service:$(shell python3 -c 'import uuid; print(uuid.uuid5(uuid.NAMESPACE_DNS, \
        "${PROVIDER_NAME}" + "${SERVICE_CONTAINER_NAME}"));'):${SERVICE_CONTAINER_NAME}

GIT_COMMIT := $(shell git rev-parse --short HEAD)
GIT_TAG := $(shell git describe --abbrev=0 --tags ${TAG_COMMIT} 2>/dev/null || true)

DOCKER_NAME=$(shell echo ${SERVICE_CONTAINER_NAME} | sed -E 's/-/_/g')
DOCKER_VERSION=${GIT_COMMIT}
DOCKER_TAG=$(shell echo ${PROVIDER_NAME} | sed -E 's/[-:]/_/g')/${DOCKER_NAME}:${DOCKER_VERSION}
DOCKER_DEPLOY=${DOCKER_REGISTRY}/${DOCKER_TAG}

PROJECT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

TEST_MODEL=export_model/model.tgz
TEST_IMG=examples/GOPR0457.JPG
TEST_IMG_DIR=examples/indonesia_flores

run:
	mkdir -p ${PROJECT_DIR}/DATA/run && rm -rf ${PROJECT_DIR}/DATA/run/*
	env PYTHONPATH=../../ivcap-sdk-python/ivcap-service-sdk-python/src \
	python ${SERVICE_FILE} \
	  --device cpu \
	  --model ${PROJECT_DIR}/${TEST_MODEL} \
		--image ${PROJECT_DIR}/${TEST_IMG} \
		--ivcap:in-dir ${PROJECT_DIR} \
		--ivcap:out-dir ${PROJECT_DIR}/DATA/run
	@echo ">>> Output should be in '${PROJECT_DIR}/DATA/run'"

run-collection:
	mkdir -p ${PROJECT_DIR}/DATA/run && rm -rf ${PROJECT_DIR}/DATA/run/*
	env PYTHON_PATH=../../ivcap-sdk-python/sdk_service/src \
	python ${SERVICE_FILE} \
	  --device cpu \
	  --model ${PROJECT_DIR}/${TEST_MODEL} \
		--image ${PROJECT_DIR}/${TEST_IMG_DIR} \
		--ivcap:in-dir ${PROJECT_DIR} \
		--ivcap:out-dir ${PROJECT_DIR}/DATA/run
	@echo ">>> Output should be in '${PROJECT_DIR}/DATA/run'"

run-segformer:
	make -f ${PROJECT_DIR}/Makefile \
		TEST_MODEL=examples/models/segformer_b5_seagrass_13/model.artifact.tgz \
		TEST_IMG=examples/indonesia_flores \
		run-collection

run-docker: #docker-build
	@echo ""
	@echo ">>>>>>> On Mac, please ensure that this directory is mounted into minikube (if that's what you are using)"
	@echo ">>>>>>>    minikube mount ${PROJECT_DIR}:${PROJECT_DIR}"
	@echo ""
	mkdir -p ${PROJECT_DIR}/DATA/runs && rm -rf ${PROJECT_DIR}/DATA/runs/*
	env DOCKER_DEFAULT_PLATFORM=linux/amd64 \
	docker run -it \
		-e IVCAP_INSIDE_CONTAINER="" \
		-e IVCAP_ORDER_ID=urn:ivcap:order:0000 \
		-e IVCAP_NODE_ID=n0 \
		-v ${PROJECT_DIR}:/data/in \
		-v ${PROJECT_DIR}/DATA/runs:/data/out \
		${DOCKER_NAME} \
		--ivcap:in-dir /data/in \
		--ivcap:out-dir /data/out \
		--model /data/in/${TEST_MODEL} \
		--image /data/in/${TEST_IMG}

run-docker-segformer:
	make -f ${PROJECT_DIR}/Makefile \
		TEST_MODEL=examples/models/segformer_b5_seagrass_13/model.artifact.tgz \
		TEST_IMG=examples/indonesia_flores \
		run-docker

MODEL_ARTIFACT=urn:ivcap:artifact:145fdbdd-06f6-45ca-a241-1ba846d33f0c
IMAGE_ARTIFACT=urn:ivcap:artifact:07793994-bbff-49d5-979b-0843b6b4093c

run-ivcap:
	ivcap order create ${SERVICE_ID} \
	  model=${MODEL_ARTIFACT} \
		image=${IMAGE_ARTIFACT}

run-data-proxy:
	mkdir -p ${PROJECT_DIR}/DATA/run && rm -rf ${PROJECT_DIR}/DATA/run/*
	env IVCAP_INSIDE_CONTAINER="Yes" \
		IVCAP_ORDER_ID=urn:ivcap:order:0000 \
		IVCAP_NODE_ID=n0 \
		http_proxy=http://localhost:9999 \
	  https_proxy=http://localhost:9999 \
		IVCAP_STORAGE_URL=http://artifact.local \
	  IVCAP_CACHE_URL=http://cache.local \
		PYTHONPATH=../../ivcap-sdk-python/sdk_service/src \
	python ${SERVICE_FILE} \
	  --device cpu \
	  --model ${MODEL_ARTIFACT} \
		--image ${IMAGE_ARTIFACT} \
		--ivcap:in-dir ${PROJECT_DIR}/DATA/run \
		--ivcap:out-dir ${PROJECT_DIR}/DATA/run \
		--ivcap:cache-dir ${PROJECT_DIR}/DATA/run
	@echo ">>> Output should be in '${PROJECT_DIR}/DATA/run'"

run-batch:
	mkdir -p ${PROJECT_DIR}/DATA/out
	find ${TEST_IMG_DIR} -name "*.JPG" \
	| while read f; do \
		echo ">>>> Processing $$f - $(shell date)"; \
		python ${PROJECT_DIR}/${SERVICE_FILE} \
			--device cpu \
			--model /tmp/model.tgz \
			--image $$f \
			--ivcap:in-dir ${PROJECT_DIR}/DATA/in; \
			--ivcap:out-dir ${PROJECT_DIR}/DATA/out; \
	done


docker-build:
	@echo "Building docker image ${DOCKER_NAME}"
	env DOCKER_DEFAULT_PLATFORM=linux/amd64 \
	docker build --no-cache --pull \
		--build-arg GIT_COMMIT=${GIT_COMMIT} \
		--build-arg GIT_TAG=${GIT_TAG} \
		--build-arg BUILD_DATE="$(shell date)" \
		-t ${DOCKER_NAME} \
		-f ${PROJECT_DIR}/Dockerfile \
		${PROJECT_DIR} ${DOCKER_BUILD_ARGS}
	@echo "\nFinished building docker image ${DOCKER_NAME}\n"

docker-publish: docker-build
	@echo "====> If 'unauthorized: authentication required' log into ACR with 'az acr login --name cipmain'"
	docker tag ${DOCKER_NAME} ${DOCKER_DEPLOY}
	docker push ${DOCKER_DEPLOY}

service-register: FORCE
	env IVCAP_SERVICE_ID=${SERVICE_ID} \
		IVCAP_PROVIDER_ID=$(shell ivcap context get provider-id) \
		IVCAP_ACCOUNT_ID=$(shell ivcap context get account-id) \
		IVCAP_CONTAINER=${DOCKER_DEPLOY} \
	python ${SERVICE_FILE} --ivcap:print-service-description \
	| ivcap service update --create ${SERVICE_ID} --format yaml -f - --timeout 600

service-description: FORCE
	env IVCAP_SERVICE_ID=${SERVICE_ID} \
		IVCAP_PROVIDER_ID=$(shell ivcap context get provider-id) \
		IVCAP_ACCOUNT_ID=$(shell ivcap context get account-id) \
		IVCAP_CONTAINER=${DOCKER_DEPLOY} \
	python ${SERVICE_FILE} --ivcap:print-service-description
FORCE:
