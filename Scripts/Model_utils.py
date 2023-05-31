import os
import sys
import tensorflow as tf
import keras
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
from models.common import DetectMultiBackend
from models.yolo import Model
from utils.general import intersect_dicts
from utils.torch_utils import select_device
import yaml


# import tensorrt as trt

def test_model_size(model, saved_model_path, device):
    gpu_model_memory_mb = 0
    if device == 'cuda':
        # Get the GPU memory allocated by the model in bytes
        torch.cuda.synchronize()
        vram_used = torch.cuda.memory_allocated(device)
        model_parameter_bytes = sum(p.element_size() * p.nelement() for p in model.parameters())
        element_size_bytes = model.buffers().__iter__().__next__().element_size()
        model_memory_bytes = model_parameter_bytes + element_size_bytes * sum(
            b.nelement() for b in model.buffers())
        other_memory_bytes = vram_used - model_memory_bytes
        
        gpu_model_memory_mb = model_memory_bytes / (1024 * 1024)
        gpu_other_memory_mb = other_memory_bytes / (1024 * 1024)
        print(f'GPU memory used but not by model : {gpu_other_memory_mb:.2f} MB')
    
    # Get the size of the model object in bytes
    model_size = sys.getsizeof(model)
    parameter_size = sum(p.numel() * p.element_size() for p in model.parameters())
    object_size_mb = (model_size + parameter_size) / (1024 * 1024)

    # Get the size of the model on disk
    disk_size_mb = os.stat(saved_model_path).st_size / (1024 * 1024)
    
    return object_size_mb, gpu_model_memory_mb, disk_size_mb


def load_pytorch_model(model_path, device, yaml_path=None, exclude=[]):
    if 'vgg16' in model_path.split(os.sep)[-1]:
        print(f'Loading VGG16 model at {model_path}')
        model_dict = torch.load(model_path, map_location='cpu')
        number_of_classes = [x for x in model_dict['state_dict'].values()][-1].shape[0]
        model = vgg16()
        num_features = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, number_of_classes, bias=True))
        model.classifier = classifier
        model.load_state_dict(model_dict['state_dict'])
        del model_dict
        imgsz = 32
        # model = model['model']
        # for m in model.modules():
        #     if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
        #         m.inplace = True  # pytorch 1.7.0 compatibility
        #     elif type(m) is nn.Upsample:
        #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    
    elif 'resnet50' in model_path.split(os.sep)[-1]:
        print(f'Loading Resnet50 model at {model_path}')
        model_dict = torch.load(model_path, map_location='cpu')
        number_of_classes = [x for x in model_dict['state_dict'].values()][-1].shape[0]
        model = resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, number_of_classes)
        model.load_state_dict(model_dict['state_dict'])
        del model_dict
        imgsz = 28
    
    elif 'yolo' in model_path.split(os.sep)[-1]:
        print(f'Loading YOLO model at {model_path}')
        # model = DetectMultiBackend(model_path, device=torch.device(device), dnn=False, data=yaml_path, fp16=False)
        resume = True
        nc = 52
        
        yolo_hyp = {'giou': 3.54, 'cls': 1.0, 'cls_pw': 0.5, 'obj': 64.3, 'obj_pw': 1.0, 'iou_t': 0.225,
                    'lr0': 0.01,
                    'lrf': 0.0005,
                    'momentum': 0.937, 'weight_decay': 0.0005, 'fl_gamma': 0.0, 'hsv_h': 0.0138, 'hsv_s': 0.664,
                    'hsv_v': 0.464}
        
        ckpt = torch.load(model_path, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        print('ckpt.keys() :')
        for x in ckpt.keys():
            print(x)
        ckpt['model'].model.nc = nc  # set nc to 80
        model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=yolo_hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (yolo_hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    
    model = model.to(device)
    print(f'Loaded model on {device}')
    return model


def load_tensorflow_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# def convert_model_to_trt():
#     # Step 3: Convert the PyTorch model to TensorRT
#     trt_logger = trt.Logger(trt.Logger.WARNING)
#
#     # Create a TensorRT builder and network
#     trt_builder = trt.Builder(trt_logger)
#     trt_network = trt_builder.create_network()
#
#     # Configure optimization parameters
#     trt_builder.max_batch_size = 1
#     trt_builder.max_workspace_size = 1 << 30  # 1GB
#
#     # Convert the PyTorch model to TensorRT
#     trt_converter = trt.pytorch.PyTorchConverter(trt_builder, trt_logger)
#     trt_engine = trt_converter.convert(model)
#
#     # Step 4: Serialize and save the TensorRT engine
#     trt_engine_path = 'path/to/save/tensorrt/engine.trt'
#     trt_serialized_engine = trt_engine.serialize()
#     with open(trt_engine_path, 'wb') as f:
#         f.write(trt_serialized_engine)
#
#     print('TensorRT engine is saved successfully!')