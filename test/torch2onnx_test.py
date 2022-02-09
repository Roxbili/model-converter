import torch
from torchvision.models import resnet18

from converter import Torch2onnxConverter

# init
model = resnet18()

# save model
model.eval()
tmp_path = '/tmp/resnet18.pth'
torch.save(model, tmp_path)

# convert
converter = Torch2onnxConverter(tmp_path, 'model.onnx', target_shape=(3,224,224))
converter.convert()