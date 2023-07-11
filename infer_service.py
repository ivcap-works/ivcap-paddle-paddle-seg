# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 CSIRO. All Rights Reserved.
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

# Original source code taken from https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/deploy/python/infer.py

from functools import reduce
import json
from typing import Any, Optional
import warnings

from predictor import DeployConfig, Predictor, adjust_image, use_auto_tune, set_logger
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

from paddleseg.utils.visualize import get_pseudo_color_map
from PIL.ImageStat import Stat

from ivcap_sdk_service import Service, Parameter, Option, Type, ServiceArgs
from ivcap_sdk_service import register_service, deliver_data, SupportedMimeTypes
from ivcap_sdk_service import get_config as ivcap_config, create_metadata, PythonWorkflow
import logging
import tarfile
import tempfile

import os

logger = None # set when called by SDK

######
# 1. Service description
#
SERVICE = Service(
    name = "infer-with-paddle-paddle",
    description = "A service which applies a PaddlePaddle model to a set of images",
    parameters = [
        Parameter(
            name='model', 
            type=Type.ARTIFACT, 
            description='Model to use (tgz archive of all needed components)'),
        Parameter(
            name='images', 
            type=Type.COLLECTION, 
            description='Image to analyse'),
        Parameter(
            name='max-img-size', 
            type=Type.INT, 
            description="Reduce 'image' to this many pixels. If set to -1, leave unchanged",
            default=1000000),
        Parameter(
            name='batch-size', 
            type=Type.INT, 
            description='Mini batch size of one gpu or cpu.',
            default=1),
        Parameter(
            name='device',
            type=Type.OPTION,
            options=[Option(value='cpu'), Option(value='gpu')],
            default="cpu",
            description="Select which device to inference, defaults to gpu."),
        Parameter(
            name='use-trt',
            type=Type.BOOL,
            description='Whether to use Nvidia TensorRT to accelerate prediction.'),
        Parameter(
            name='precision',
            type=Type.OPTION,
            options=[Option(value='fp32'), Option(value='fp16'), Option(value='int8')],
            default="fp32",
            description='The tensorrt precision.'),
        Parameter(
            name='min-subgraph-size',
            default=3,
            type=Type.INT,
            description='The min subgraph size in tensorrt prediction.'),
        Parameter(
            name='enable-auto-tune',
            type=Type.BOOL,
            description='Whether to enable tuned dynamic shape. We uses some images to collect '
            'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'),
        # Parameter(
        #     name='auto-tuned-shape-file',
        #     type=Type.STRING,
        #     default="auto_tune_tmp.pbtxt",
        #     description='The temp file to save tuned dynamic shape.'),
        Parameter(
            name='cpu-threads',
            default=10,
            type=Type.INT,
            description='Number of threads to predict when using cpu.'),
        Parameter(
            name='enable-mkldnn',
            type=Type.BOOL,
            description='Enable to use mkldnn to speed up when using cpu.'),
        Parameter(
            name='benchmark',
            type=Type.BOOL,
            description="Whether to log some information about environment, model, configuration and performance."),
        Parameter(
            name='model-name',
            type=Type.STRING,
            description='When `--benchmark` is True, the specified model name is displayed.',
            optional=True),
        Parameter(
            name='with-argmax',
            type=Type.BOOL,
            description='Perform argmax operation on the predict result.'),
        Parameter(
            name='print-detail',
            type=Type.BOOL,
            description='Print GLOG information of Paddle Inference.') ,   
    ],
    workflow = PythonWorkflow(min_memory="5Gi"),
)

######
# 2. Service entry point
#
def service(args: ServiceArgs, svc_logger: logging):
    """Called after the service has started and all paramters have been parsed and validated

    Args:
        args (ServiceArgs): A Dict where the key is one of the `Parameter` defined in the above `SERVICE`
        svc_logger (logging): Logger to use for reporting information on the progress of execution
    """
    global logger
    logger = svc_logger
    set_logger(svc_logger)

    with tempfile.TemporaryDirectory() as tmp_dir:
        io_mgr = IOManager(args, tmp_dir)

        # collect dynamic shape by auto_tune
        # if use_auto_tune(args):
        #     tune_img_nums = 10
        #     auto_tune(args, imgs_list, tune_img_nums)

        # create and run predictor
        predictor = Predictor(args, io_mgr.get_config())
        predictor.run(io_mgr)

        if use_auto_tune(args) and \
            os.path.exists(args.auto_tuned_shape_file):
            os.remove(args.auto_tuned_shape_file)

        if args.benchmark:
            predictor.autolog.report()


######
# 3. I/O and interface to Predictor
#
class IOManager:
    def __init__(self, args: ServiceArgs, tmp_dir: str):
        self.args = args
        #logger.info(f"image name: '{img_name}' path: '{args.image.path}' - isfile: {os.path.isfile(args.image.path)}")
        #self.img_list, _ = get_image_list(args.image.path)
        self.img_list = []
        self.images = {}
        for img in args.images:
            imgA = adjust_image(img, args.max_img_size)
            self.img_list.append(imgA)
            self.images[imgA] = img
        logger.info(f"Image list '{self.img_list}'")
        self.batch_size = args.batch_size

        self.save_dir = '/tmp'

        logger.info(f"Opening model '{args.model.name}'.")
        tf = tarfile.open(args.model.as_local_file(), 'r|gz')
        tf.extractall(tmp_dir)
        deployPath = os.path.join(tmp_dir, 'deploy.yaml')
        self.cfg = DeployConfig(deployPath)
        
        with open(os.path.join(tmp_dir, 'meta.json')) as f:
            self.meta = json.load(f)
            self.classes = self.meta.get("classes", None)

    def __repr__(self):
        return f"IOManager(batch_size={self.batch_size }, save_dir={self.save_dir}, img_list={self.img_list})"

    def get_config(self) -> DeployConfig:
        return self.cfg

    def  get_colormap(self) -> Optional[Any]:
        c = self.meta.get("classes", None)
        if not c:
            return None
        
        def r(p, el):
            for c in el["def_color"]:
                p.append(c)
            return p
        
        cm = reduce(r, c, [])
        return cm

    def save_imgs(self, results, imgs_path):
        logger.debug(f"... save_imgs shape: {results.shape} img_path: {imgs_path}")

        cm = self.get_colormap()
        for i in range(results.shape[0]):
            result = results[i]
            img = self.images[imgs_path[i]]
            img_name = img.name
            pseudo_img = get_pseudo_color_map(result, cm)
            stats = Stat(pseudo_img)
            logger.debug(f'... 0/1: {stats.h[:2]} count: {stats.count} shape: {result.shape}')
            basename = os.path.basename(img_name)
            basename, _ = os.path.splitext(basename)
            basename = f'{basename}.pseudo.png'

            meta = create_metadata('urn:ibenthos:schema:paddle.seg.inference.1', {
                'image': img_name,
                'model': self.args.model.name,
                'width': result.shape[0],
                'height': result.shape[1],
                'cover': self.get_cover(stats),
                #'params': self.args._asdict(),
                'order-id': ivcap_config().ORDER_ID,
            })
            url = deliver_data(basename, lambda f: pseudo_img.save(f, format='png'), SupportedMimeTypes.JPEG, metadata=meta) 
            logger.debug(f"Saved pseudo colored image ({pseudo_img}) type as '{url}'")

    def get_cover(self, stats: Stat):
        cover = []
        count = stats.count[0]
        for i, cl in enumerate(self.classes):
            m = cl.copy()
            m['color'] = m.pop("def_color",  None)
            m['cover'] = 1.0 * stats.h[i] / count
            cover.append(m)
        return cover
            
            
        
    def __iter__(self):
        return self.Iter(self)

    class Iter:
        def __init__(self, outer):
            self.outer = outer
            self.n = 0

        def __next__(self):
            iml = self.outer.img_list
            batch_size = self.outer.batch_size
            if self.n < len(iml):
                result = iml[self.n:self.n + batch_size]
                self.n += batch_size
                logger.debug(f"... supplying images: {result}")
                return result
            else:
                raise StopIteration


######
# 4. Service registration
#
# Register this service with IVCAP which in turn
# will call 'service' with the relevant parameters
# defined in 'SERVICE'
#
register_service(SERVICE, service)

