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

import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import yaml
import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig
import os

import paddleseg.transforms as T
from paddleseg.cvlibs import manager
# from paddleseg.utils import get_sys_env, get_image_list
# from paddleseg.utils.visualize import get_pseudo_color_map
# from PIL.ImageStat import Stat
from PIL import Image

# from ivcap_sdk_service import Service, Parameter, Option, Type, ServiceArgs
# from ivcap_sdk_service import register_service, deliver_data, SupportedMimeTypes
# from ivcap_sdk_service import get_config as ivcap_config, create_metadata, PythonWorkflow
import logging
# import tarfile
import tempfile

from typing import Dict

import codecs
import os
# import json

logger = None # set when called by SDK

def set_logger(l: logging):
    global logger
    logger = l

def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
        and args.device == "gpu" and args.use_trt and args.enable_auto_tune

def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args), "Do not support auto_tune, which requires " \
        "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = np.array([cfg.transforms(imgs[i])[0]])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "Auto tune failed. Usually, the error is out of GPU memory "
                "for the model or image is too large. \n")
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune success.\n")


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self.load_transforms(self.dic['Deploy'][
            'transforms'])
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)

class Predictor:
    def __init__(self, args: Dict, cfg: DeployConfig):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = cfg # DeployConfig(args.cfg)

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set --enable_auto_tune=True to use auto_tune. \n")
            exit()

        if hasattr(args, 'benchmark') and args.benchmark:
            import auto_log
            pid = os.getpid()
            self.autolog = auto_log.AutoLogger(
                model_name=args.model_name,
                model_precision=args.precision,
                batch_size=args.batch_size,
                data_shape="dynamic",
                save_path=None,
                inference_config=self.pred_cfg,
                pids=pid,
                process_name=None,
                gpu_ids=0,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args.print_detail:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

        if self.args.use_trt:
            logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                os.path.exists(self.args.auto_tuned_shape_file):
                logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, io_manager):
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        first = True
        for img_batch in io_manager:
            # warm up
            if first and args.benchmark:
                for j in range(5):
                    data = np.array([
                        self._preprocess(img)
                        for img in img_batch
                    ])
                    input_handle.reshape(data.shape)
                    input_handle.copy_from_cpu(data)
                    self.predictor.run()
                    results = output_handle.copy_to_cpu()
                    results = self._postprocess(results)
            first = False

            # inference
            if args.benchmark:
                self.autolog.times.start()

            data = np.array([
                self._preprocess(p) for p in img_batch
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            if args.benchmark:
                self.autolog.times.stamp()

            self.predictor.run()

            if args.benchmark:
                self.autolog.times.stamp()

            results = output_handle.copy_to_cpu()
            results = self._postprocess(results)

            if args.benchmark:
                self.autolog.times.end(stamp=True)

            io_manager.save_imgs(results, img_batch)
        logger.info("Done")

    def _preprocess(self, img):
        logger.debug(f"... _preprocess {img}, self.cfg.transforms: {self.cfg.transforms}")
        t = self.cfg.transforms({"img": img})
        return t["img"]

    def _postprocess(self, results):
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results

######
#
# Utility function

def adjust_image(image, max_size):
    """Ensures that image is of max size if defined in --max-size

    Args:
        path (str): File name of image
    """
    imgPath = image.as_local_file()
    logger.info(f"Checking if image '{image.name}' needs adjusting (max-size: {max_size})")
    if max_size < 0:
        # keep image
        return imgPath

    img = Image.open(imgPath)
    format = img.format
    width, height = img.size
    size = width * height
    if size > max_size:
        # too big
        scale = math.sqrt(max_size / size)
        w = int(scale * width)
        h = int(scale * height)
        logger.info(f"Downscaling image by '{scale}' ({w}x{h})")
        img = img.resize((w,h))
        imgf = tempfile.NamedTemporaryFile("w+b", delete=False)
        img.save(imgf, format)
        imgPath = imgf.name
    return imgPath