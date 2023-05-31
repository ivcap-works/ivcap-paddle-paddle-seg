# Utility to Export PaddlePaddle model for use by IVCAP service

This simple script (and docker image) takes a PaddlePaddle model produced
by a training run and creates a standalone model file (tgz archive) containing
all the necessary information required by the IVCAP `paddle-paddle-servicing' service.

## Usage

First install all the necessary requirements listed in `requirements.txt`.

Assuming we have the result of a training regime stored in `MODEL_DIR` with the following content:

```
% ls -1 ${MODEL_DIR}
model.pdparams
ocrnet_48-seagrass.json
ocrnet_hrnetw48_seagrass_test.yml
train.txt
val.txt
```

we can then package it with:

```
python export.py \
  --config ${MODEL_DIR}/ocrnet_hrnetw48_seagrass_test.yml \
  --model-path ${MODEL_DIR}/model.pdparams \
  --save-path /tmp/model.tgz
```

where `--config` refers to the config file used for training, `--model-path` refers
to the parameters learned, and `--save-path` identifies the path and name of the
created export file.

Additional options can be displayed by running the command with the `-h` flag:

```
% python export.py -h
usage: export.py [-h] --config CFG [--save-path SAVE_PATH]
                 [--model-path MODEL_PATH] [--without-argmax] [--with-softmax]
                 [--input-shape INPUT_SHAPE [INPUT_SHAPE ...]]

Model export.

optional arguments:
  -h, --help            show this help message and exit
  --config CFG          The config file.
  --save-path SAVE_PATH
                        The file name for the exported model
  --model-path MODEL_PATH
                        The path of model for export
  --without-argmax      Do not add the argmax operation at the end 
                        of the network
  --with-softmax        Add the softmax operation at the end of 
                        the network
  --input-shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Export the model with fixed input shape, such 
                        as 1 3 1024 1024.
```

After a model is exported, we should upload it to IVCAP as an artifact:

```
ivcap artifact create -f /tmp/model.tgz -n model-seagrass-v1
```

Please note down the artifact id (e.g. `urn:ivcap:artifact:....`) as it will be required
as the `model` argument for the infer service.
