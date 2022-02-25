from typing import Optional
import os, sys
import numpy as np
import shutil
import logging
logging.getLogger().setLevel(logging.INFO) # 为了打印log需要添加的

try:
    import cv2
except:
    pass
import torch
from torch.onnx import TrainingMode
import onnx
from onnx import shape_inference


class Torch2onnxConverter:
    def __init__(
            self,
            torch_model_path: str,
            onnx_model_save_path: str = None,
            sample_file_path: Optional[str] = None,
            target_shape: tuple = (3, 224, 224),
            seed: int = 10,
            normalize: bool = True,
            op_fuse: bool = True
    ):
        """Convert pytorch model to onnx model

            Args:
                torch_model_path: the path to load pytorch model.
                onnx_model_save_path: the path to save onnx model, 
                        if None, save to /tmp/model_converter/model.onnx.
                sample_file_path: input path (e.g., ./cat.jpg), 
                        if None, random data will be generated according to target_shape.
                target_shape: input shape size.
                seed: random seed number.
                normalize: whether to normalize the input.
                op_fuse: whether to fuse the operator when pytorch model is 
                        converted to onnx model (e.g., Conv2D-BatchNorm -> Conv2D).
                        Note that the operator will be fused when converted to tflite model.
        """
        self.torch_model_path = torch_model_path
        self.sample_file_path = sample_file_path
        self.target_shape = target_shape
        self.op_fuse = op_fuse

        if onnx_model_save_path is None:
            self.tmpdir = '/tmp/model_converter/'
            self.__check_tmpdir()
            self.onnx_model_path = os.path.join(self.tmpdir, 'model.onnx')
        else:
            self.onnx_model_path = onnx_model_save_path
        self.torch_model = self.load_torch_model()
        self.sample_data = self.load_sample_input(sample_file_path, target_shape, seed, normalize)

    def convert(self):
        self.torch2onnx()
        self.onnx_inference_shapes()

    def __check_tmpdir(self):
        try:
            if os.path.exists(self.tmpdir) and os.path.isdir(self.tmpdir):
                shutil.rmtree(self.tmpdir)
                logging.info(f'Old temp directory removed')
            os.makedirs(self.tmpdir, exist_ok=True)
            logging.info(f'Temp directory created at {self.tmpdir}')
        except Exception:
            logging.error('Can not create temporary directory, exiting!')
            sys.exit(-1)

    def load_torch_model(self) -> torch.nn.Module:
        try:
            if self.torch_model_path.endswith('.pth') or self.torch_model_path.endswith('.pt'):
                model = torch.load(self.torch_model_path, map_location='cpu')
                model = model.eval()
                logging.info('PyTorch model successfully loaded and mapped to CPU')
                return model
            else:
                logging.error('Specified file path not compatible with torch2tflite, exiting!')
                sys.exit(-1)
        except Exception:
            logging.error('Can not load PyTorch model. Please make sure'
                          'that model saved like `torch.save(model, PATH)`')
            sys.exit(-1)

    def load_sample_input(
            self,
            file_path: Optional[str] = None,
            target_shape: tuple = (3, 224, 224),
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

                sample_data_np = img[np.newaxis, :, :, :]
                sample_data_torch = torch.from_numpy(sample_data_np)
                logging.info(f'Sample input successfully loaded from, {file_path}')

            except Exception:
                logging.error(f'Can not load sample input from, {file_path}')
                sys.exit(-1)

        else:
            logging.info(f'Sample input file path not specified, random data will be generated')
            np.random.seed(seed)
            data = np.random.random(target_shape).astype(np.float32)
            sample_data_np = data[np.newaxis, :, :, :]
            sample_data_torch = torch.from_numpy(sample_data_np)
            logging.info(f'Sample input randomly generated')

        return {'sample_data_np': sample_data_np, 'sample_data_torch': sample_data_torch}

    def torch2onnx(self) -> None:
        torch.onnx.export(
            model=self.torch_model,
            args=self.sample_data['sample_data_torch'],
            f=self.onnx_model_path,
            verbose=False,
            export_params=True,
            do_constant_folding=False,
            training=TrainingMode.EVAL if self.op_fuse else TrainingMode.TRAINING,
            input_names=['input'],
            opset_version=11,
            output_names=['output'])

    def add_value_info_for_constants(self, model : onnx.ModelProto):
        """
        Currently onnx.shape_inference doesn't use the shape of initializers, so add
        that info explicitly as ValueInfoProtos.
        Mutates the model.
        Args:
            model: The ModelProto to update.
        """
        # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
        if model.ir_version < 4:
            return

        def add_const_value_infos_to_graph(graph : onnx.GraphProto):
            inputs = {i.name for i in graph.input}
            existing_info = {vi.name: vi for vi in graph.value_info}
            for init in graph.initializer:
                # Check it really is a constant, not an input
                if init.name in inputs:
                    continue

                # The details we want to add
                elem_type = init.data_type
                shape = init.dims

                # Get existing or create new value info for this constant
                vi = existing_info.get(init.name)
                if vi is None:
                    vi = graph.value_info.add()
                    vi.name = init.name

                # Even though it would be weird, we will not overwrite info even if it doesn't match
                tt = vi.type.tensor_type
                if tt.elem_type == onnx.TensorProto.UNDEFINED:
                    tt.elem_type = elem_type
                if not tt.HasField("shape"):
                    # Ensure we set an empty list if the const is scalar (zero dims)
                    tt.shape.dim.extend([])
                    for dim in shape:
                        tt.shape.dim.add().dim_value = dim

            # Handle subgraphs
            for node in graph.node:
                for attr in node.attribute:
                    # Ref attrs refer to other attrs, so we don't need to do anything
                    if attr.ref_attr_name != "":
                        continue

                    if attr.type == onnx.AttributeProto.GRAPH:
                        add_const_value_infos_to_graph(attr.g)
                    if attr.type == onnx.AttributeProto.GRAPHS:
                        for g in attr.graphs:
                            add_const_value_infos_to_graph(g)

        return add_const_value_infos_to_graph(model.graph)

    def onnx_inference_shapes(self) -> None:
        """
        Add inference shapes to the onnx graph,
        and it can be visualized by Netron(https://github.com/lutzroeder/netron)
        """ 

        # 加载模型
        onnx_model = onnx.load(self.onnx_model_path)

        # 添加inference shape info
        self.add_value_info_for_constants(onnx_model)

        # 调用API添加inference shape信息并保存模型
        inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(inferred_onnx_model, self.onnx_model_path)
        logging.info(f'Onnx model is saved to {self.onnx_model_path}')

    def inference_torch(self) -> np.ndarray:
        y_pred = self.torch_model(self.sample_data['sample_data_torch'])
        return y_pred.detach().cpu().numpy()