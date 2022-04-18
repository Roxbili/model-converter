import torch
from torchvision.models import resnet18

from converter import Torch2TFLiteConverter

# init
model = resnet18()

# save model
model.eval()
tmp_path = '/tmp/resnet18.pth'
torch.save(model, tmp_path)

# float32 convert
converter = Torch2TFLiteConverter(tmp_path, tflite_model_save_path='model_float32.lite', target_shape=(224,224,3))
converter.convert()

# int8 convert
dataset = torch.rand(16,3,32,32, dtype=torch.float32)
def representative_dataset():
    for data in dataset:
        data = data.unsqueeze(0) # (1,3,32,32)
        yield [data]

converter = Torch2TFLiteConverter(tmp_path, tflite_model_save_path='model_int8.lite', target_shape=(224,224,3),
                                    representative_dataset=representative_dataset, normalize=True)
converter.convert()