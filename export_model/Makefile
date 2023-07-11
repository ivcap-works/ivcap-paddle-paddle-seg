CONTAINER_NAME=paddle-paddle-export
PROVIDER_NAME=bluecarbon

# don't foget to login 'az acr login --name cipmain'
DOCKER_REGISTRY=cipmain.azurecr.io

PROJECT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

MODEL_DIR=${PROJECT_DIR}/../examples/fiji_lau/ocrnet_48-seagrass
MODEL_NAME=ocrnet_hrnetw48_seagrass_test

GIT_COMMIT := $(shell git rev-parse --short HEAD)
GIT_TAG := $(shell git describe --abbrev=0 --tags ${TAG_COMMIT} 2>/dev/null || true)

DOCKER_NAME=$(shell echo ${CONTAINER_NAME} | sed -E 's/-/_/g')
DOCKER_VERSION=${GIT_COMMIT}
DOCKER_TAG=$(shell echo ${PROVIDER_NAME} | sed -E 's/[-:]/_/g')/${DOCKER_NAME}:${DOCKER_VERSION}
DOCKER_DEPLOY=${DOCKER_REGISTRY}/${DOCKER_TAG}

run: ${PROJECT_DIR}/model.tgz

${PROJECT_DIR}/model.tgz:
	python ${PROJECT_DIR}/export.py \
		--config ${MODEL_DIR}/${MODEL_NAME}.yml \
		--model ${MODEL_DIR}/model.pdparams \
		--save-path ${PROJECT_DIR}/model.tgz


segformer:
	python ${PROJECT_DIR}/export.py \
		--cv-config ../examples/models/segformer_b5_seagrass_13/paddle_segmentation_v2-5_mix_vision_transformer_b5_public_seagrass_fiji_indo.json \
		--save-path ../examples/models/segformer_b5_seagrass_13/model.artifact.tgz

deploy-model: ${PROJECT_DIR}/model.tgz
	@echo "... uploading model - may take a while"
	$(eval BUILD_ART=$(shell ivcap --silent artifact create -n ${MODEL_NAME}.model -f ${PROJECT_DIR}/model.tgz --chunk-size 20000000))
	@echo "ArtifactID: $(BUILD_ART)"
	sed -e "s/@@ARTIFACT@@/$(BUILD_ART)/" model-meta.json \
	 | ivcap --timeout 600 metadata add $(BUILD_ART) -f -

# ivcap artifact create -n ${MODEL_NAME}.tgz -f ${PROJECT_DIR}/model.tgz --chunk-size 20000000

docker-run: #docker-build
	docker run -it \
		-v ${MODEL_DIR}:/model \
		-v /tmp:/output \
		${DOCKER_NAME} \
		--config /model/ocrnet_hrnetw48_seagrass_test.yml \
		--model-path /model/model.pdparams \
		--save-path /output/model.tgz

# Minikube on the Mac may not mount user directories under the same path
docker-run-minikube:
	docker run -it \
		-v /src/IVCAP/ivcap-paddle-paddle/export_model/../examples/fiji_lau/ocrnet_48-seagrass:/model \
		-v /tmp:/output \
		${DOCKER_NAME} \
		--config /model/ocrnet_hrnetw48_seagrass_test.yml \
		--model-path /model/model.pdparams \
		--save-path /output/model.tgz
	@echo ">>> Model is REALLY saved in /tmp on minikube's VM (minikube ssh)"

docker-build:
	@echo "Building docker image ${DOCKER_NAME}"
	docker build \
		--build-arg GIT_COMMIT=${GIT_COMMIT} \
		--build-arg GIT_TAG=${GIT_TAG} \
		--build-arg BUILD_DATE="$(shell date)" \
		-t ${DOCKER_NAME} \
		-f ${PROJECT_DIR}/Dockerfile \
		${PROJECT_DIR} ${DOCKER_BILD_ARGS}
	@echo "\nFinished building docker image ${DOCKER_NAME}\n"

docker-publish: docker-build
	@echo "====> If 'unauthorized: authentication required' log into ACR with 'az acr login --name cipmain'"
	docker tag ${DOCKER_NAME} ${DOCKER_DEPLOY}
	docker push ${DOCKER_DEPLOY}

FORCE:
