from typing import Optional, Generator, Union

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import shutil
from .logger import Logger
_logger = Logger(__name__).get_logger()

try:
    import cv2
except:
    pass
import numpy as np
import onnx
from onnx_tf.backend import prepare
import torch
import tensorflow as tf

from .torch2onnx import Torch2onnxConverter


class Torch2TFLiteConverter(Torch2onnxConverter):
    def __init__(
            self,
            torch_model_path: str,
            tf_model_path: str = None,
            tflite_model_save_path: str = None,
            sample_file_path: Optional[str] = None,
            target_shape: tuple = (224, 224, 3),
            seed: int = 10,
            normalize: bool = False,
            op_fuse: bool = True,
            representative_dataset: Generator = None,
            evaluate_loader = None
    ):
        """Convert pytorch model to TFLite model.

            Args:
                torch_model_path: the path to load pytorch model.
                tf_model_path: where to save tensorflow model,
                        if None, save to /tmp/model_converter/tf_model.
                sample_file_path: input path (e.g., ./cat.jpg), 
                        if None, random data will be generated according to target_shape.
                target_shape: input shape size.
                seed: random seed number.
                normalize: whether to normalize the sample_data.
                op_fuse: whether to fuse the operator when pytorch model is 
                        converted to onnx model (e.g., Conv2D-BatchNorm -> Conv2D).
                        Note that the operator will be fused when converted to tflite model.
                representative_dataset: generator type, yield data in function.
                        If given, the tensorflow model will be quantized, 
                        and the representative data is used to calibrate quantization model.
                        (nchw)
                evaluate_loader: pytorch dataloader to evaluate the quantized tflite model, if None, the model will not be evaluated.
        """
        super().__init__(torch_model_path=torch_model_path, sample_file_path=sample_file_path, \
                            target_shape=target_shape, seed=seed, normalize=normalize, op_fuse=op_fuse)

        self.tflite_model_path = tflite_model_save_path if tflite_model_save_path is not None \
                                    else os.path.join(self.tmpdir, 'model.lite') 
        self.tf_model_path = tf_model_path if tf_model_path is not None \
                                    else os.path.join(self.tmpdir, 'tf_model') 
        self.representative_dataset = representative_dataset
        self.evaluate_loader = evaluate_loader

    def convert(self):
        self.torch2onnx()
        self.onnx_inference_shapes()
        self.onnx2tf()
        self.tf2tflite()
        torch_output = self.inference_torch()
        interpreter = self.load_tflite()
        tflite_output = self.inference_tflite(interpreter)
        self.calc_error(torch_output, tflite_output)
        if self.evaluate_loader is not None:
            self.evaluate(interpreter)

    def load_sample_input(
            self,
            file_path: Optional[str] = None,
            target_shape: tuple = (224, 224, 3),
            seed: int = 10,
            normalize: bool = True
    ):
        if file_path is not None:
            if (len(target_shape) == 3 and target_shape[-1] == 1) or len(target_shape) == 2:
                imread_flags = cv2.IMREAD_GRAYSCALE
            elif len(target_shape) == 3 and target_shape[-1] == 3:
                imread_flags = cv2.IMREAD_COLOR
            else:
                imread_flags = cv2.IMREAD_ANYCOLOR + cv2.IMREAD_ANYDEPTH
            try:
                img = cv2.resize(
                    src=cv2.imread(file_path, imread_flags),
                    dsize=target_shape[:2],
                    interpolation=cv2.INTER_LINEAR
                )
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif len(img.shape) == 2:
                    img = img[..., np.newaxis]

                if normalize:
                    img = img * 1. / 255
                img = img.astype(np.float32)

                sample_data_np = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
                sample_data_torch = torch.from_numpy(sample_data_np)
                _logger.info(f'Sample input successfully loaded from, {file_path}')

            except Exception:
                _logger.error(f'Can not load sample input from, {file_path}')
                sys.exit(-1)

        else:
            _logger.info(f'Sample input file path not specified, random data will be generated')
            np.random.seed(seed)
            data = np.random.random(target_shape).astype(np.float32)
            sample_data_np = np.transpose(data, (2, 0, 1))[np.newaxis, :, :, :]
            sample_data_torch = torch.from_numpy(sample_data_np)
            _logger.info(f'Sample input randomly generated')

        return {'sample_data_np': sample_data_np, 'sample_data_torch': sample_data_torch}

    def load_tflite(self):

        interpret = tf.lite.Interpreter(self.tflite_model_path)
        interpret.allocate_tensors()
        _logger.info(f'TFLite interpreter successfully loaded from, {self.tflite_model_path}')
        return interpret

    def onnx2tf(self) -> None:
        onnx_model = onnx.load(self.onnx_model_path)
        onnx.checker.check_model(onnx_model)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(self.tf_model_path)
        _logger.info('Change onnx model tensorflow model')

    def tf2tflite(self) -> None:
        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path)

        if self.representative_dataset is not None:
            # quantization config
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset

            # assert all operations support int8, or raise error
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]

            # set input and output as int8
            # converter.inference_input_type = tf.uint8
            # converter.inference_output_type = tf.uint8
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()
        with open(self.tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        _logger.info(f'Tflite model is saved to {self.tflite_model_path}')

    def inference_tflite(self, interpreter: tf.lite.Interpreter) -> np.ndarray:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], self.sample_data['sample_data_np'])
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])
        return y_pred

    @staticmethod
    def calc_error(result_torch, result_tflite):
        mse = ((result_torch - result_tflite) ** 2).mean(axis=None)
        mae = np.abs(result_torch - result_tflite).mean(axis=None)
        _logger.info(f'MSE (Mean-Square-Error): {mse}\tMAE (Mean-Absolute-Error): {mae}')

    def evaluate(self, interpreter: tf.lite.Interpreter):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        correct_num = 0
        for index, (image, label) in enumerate(self.evaluate_loader):
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
                label = label.cpu().numpy()
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            y_pred = interpreter.get_tensor(output_details[0]['index'])

            if index % 100 == 0:
                _logger.info(f'inference: {index}/{len(self.evaluate_loader) * 1}')
            if y_pred.argmax() == label[0]:
                correct_num += 1

        accuracy = correct_num / (len(self.evaluate_loader) * 1)    # 1为batch_size
        _logger.info(f'tflite int8 model accuracy: {accuracy}')


if __name__ == '__main__':
    '''
    Args
        --torch-path Path to local PyTorch model, please save whole model e.g. torch.save(model, PATH)
        --tf-lite-path Save path for Tensorflow Lite model
        --target-shape Model input shape to create static-graph (default: (224, 224, 3)
        --sample-file Path to sample image file. If model is not about computer-vision, please use leave empty and only enter --target-shape
        --seed Seeds RNG to produce random input data when --sample-file does not exists
        --log=INFO To see what happens behind
    '''

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-path', type=str, required=True)
    parser.add_argument('--tflite-path', type=str, required=True)
    parser.add_argument('--target-shape', type=tuple, nargs=3, default=(224, 224, 3))
    parser.add_argument('--sample-file', type=str)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    conv = Torch2TFLiteConverter(
        args.torch_path,
        args.tflite_path,
        args.sample_file,
        args.target_shape,
        args.seed
    )
    conv.convert()
    sys.exit(0)
