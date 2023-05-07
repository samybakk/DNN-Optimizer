import tensorflow as tf
import keras
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights,vgg16, VGG16_Weights

def load_pytorch_model(model_path, device):
    try:
        model_dict = torch.load(model_path, map_location='cpu')
        number_of_classes = [x for x in model_dict['state_dict'].values()][-1].shape[0]
        if model_path.split('/')[-1].startswith('vgg16'):
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
        
        elif model_path.split('/')[-1].startswith('resnet50'):
            model = resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, number_of_classes)
        model.load_state_dict(model_dict['state_dict'])
        
        # model = model['model']
        # for m in model.modules():
        #     if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
        #         m.inplace = True  # pytorch 1.7.0 compatibility
        #     elif type(m) is nn.Upsample:
        #         m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    
    except:
        model = torch.jit.load(model_path, map_location=torch.device(device=device))
    
    model = model.to(device)
    return model

def load_tensorflow_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model