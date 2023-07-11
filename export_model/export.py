# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path
import json

import paddle
import yaml
import tempfile
import tarfile

from paddleseg.cvlibs import Config
from paddleseg.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    # params of training
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file. (if not set, assumed to be in --cv-config)",
        default=None,
        type=str,
        required=False)
    parser.add_argument(
        "--cv-config",
        dest="cv_cfg",
        help="The cv-pipeline config file.",
        default=None,
        type=str,
        required=False)
    parser.add_argument(
        '--save-path',
        dest='save_path',
        help='The file name for the exported model',
        type=str,
        default='./model.artifact.tgz')
    parser.add_argument(
        '--model',
        dest='model',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument(
        '--without-argmax',
        dest='without_argmax',
        help='Do not add the argmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        '--with-softmax',
        dest='with_softmax',
        help='Add the softmax operation at the end of the network',
        action='store_true')
    parser.add_argument(
        "--input-shape",
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)

    return parser.parse_args()


class SavedSegmentationNet(paddle.nn.Layer):
    def __init__(self, net, without_argmax=False, with_softmax=False):
        super().__init__()
        self.net = net
        self.post_processer = PostPorcesser(without_argmax, with_softmax)

    def forward(self, x):
        outs = self.net(x)
        outs = self.post_processer(outs)
        return outs


class PostPorcesser(paddle.nn.Layer):
    def __init__(self, without_argmax, with_softmax):
        super().__init__()
        self.without_argmax = without_argmax
        self.with_softmax = with_softmax

    def forward(self, outs):
        new_outs = []
        for out in outs:
            if self.with_softmax:
                out = paddle.nn.functional.softmax(out, axis=1)
            if not self.without_argmax:
                out = paddle.argmax(out, axis=1)
            new_outs.append(out)
        return new_outs

def load_net(args, shape):
    cfg = Config(args.cfg)
    net = cfg.model

    if args.model:
        para_state_dict = paddle.load(args.model)
        net.set_dict(para_state_dict)
        logger.info('Loaded trained params of model successfully.')

    if not args.without_argmax or args.with_softmax:
        new_net = SavedSegmentationNet(net, args.without_argmax,
                                       args.with_softmax)
    else:
        new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(
            shape=shape, dtype='float32')])
    return [new_net, cfg]
    
def main(args):
    os.environ['PADDLESEG_EXPORT_STAGE'] = 'True'

    seg_classes = []
    def_colors = []
    
    if args.cv_cfg:
        with open(args.cv_cfg) as f:
            j = json.load(f)
            dir = os.path.dirname(args.cv_cfg)
            if not args.cfg:
                cfg = j["paddlepaddlesegmentation_config_filename"]
                args.cfg = os.path.join(dir, cfg)
            if not args.model:
                model = j["paddlepaddlesegmentation_model_filename"]
                args.model = os.path.join(dir, model)
            seg_classes = j.get("segmentationpostprocessing_classes", [])
            def_colors = j.get("segmentationpostprocessing_default_class_colours", [])

    t = str.maketrans("/ ", "--", "")
    for i, name in enumerate(seg_classes):
        n = name.translate(t).lower()
        id = f"urn:ibenthos:segmentation:class:{n}"
        seg_classes[i] = {
            "id": id,
            "name": name,
            "def_color": def_colors[i] if len(def_colors) > i else [],
        }

    if args.input_shape is None:
        shape = [None, 3, None, None]
    else:
        shape = args.input_shape
            
    [net, cfg] = load_net(args, shape)
            
    meta = {
        "$schema": "urn:ibenthos:schema:paddle.seg.model.1",
        "name": Path(args.cfg).stem,
        "model": cfg.dic["model"],
        "classes": seg_classes,
        "shape": shape,
        "artifact": "@@ARTIFACT@@"
    }
    jp = Path(args.save_path)
    jp = jp.with_name(f"{jp.stem}-meta.json")
    with open(jp, "w") as fp:
        json.dump(meta, fp, indent=2) 

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, 'model')
        paddle.jit.save(net, save_path)

        yml_file = os.path.join(tmp_dir, 'deploy.yaml')
        with open(yml_file, 'w') as file:
            transforms = cfg.export_config.get('transforms', [{
                'type': 'Normalize'
            }])
            data = {
                'Deploy': {
                    'transforms': transforms,
                    'model': 'model.pdmodel',
                    'params': 'model.pdiparams'
                }
            }
            yaml.dump(data, file)

        meta_file = os.path.join(tmp_dir, 'meta.json')
        with open(meta_file, 'w') as file:
            meta.pop("artifact",  None)
            json.dump(meta, file, indent=2) 

        with tarfile.open(args.save_path, 'w:gz') as tar:
            tar.add(tmp_dir, arcname='.')

    logger.info(f'Model is saved in {args.save_path}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
