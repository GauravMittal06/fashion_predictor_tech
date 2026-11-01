"""Download every pretrained model the prototype uses (run once; needs internet)."""
import torch

print('1/5 person detector (Faster R-CNN MobileNetV3)', flush=True)
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)

print('2/5 Marqo-FashionSigLIP', flush=True)
import open_clip
open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')

print('3/5 Chronos-2', flush=True)
from chronos import Chronos2Pipeline
Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu')

print('4/5 SDXL-Turbo (fp16)', flush=True)
from diffusers import AutoPipelineForText2Image
AutoPipelineForText2Image.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16')
print('5/5 FashionCLIP 2.0 (second tagger expert)', flush=True)
from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained('patrickjohncyh/fashion-clip')
CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')
print('done', flush=True)
