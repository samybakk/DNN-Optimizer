import tensorflow.lite as tflite
import torchvision.models.quantization
from typing import Type, Any, Callable, Union, List, Optional

from torch import Tensor
from torch.ao.quantization import QuantStub, DeQuantStub, MinMaxObserver, default_observer

from Dataset_utils import *


def quantize_model_tensorflow(model, desired_format,logger):
    converter = tflite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if desired_format == 'int8':
        desired_format = tf.qint8
    elif desired_format == 'int16':
        desired_format = tf.qint16
    elif desired_format == 'int32':
        desired_format = tf.qint32
    elif desired_format == 'float16':
        desired_format = tf.float16
    else:
        desired_format = tf.float32
    
    converter.target_spec.supported_types = [desired_format]
    tflite_quant_model = converter.convert()
    return tflite_quant_model



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        out = self.skip_add.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.quant(x) # add quant
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x) # add dequant

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    progress: bool,
    num_classes,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers,num_classes, **kwargs)

    return model


def resnet50_quantizable(progress: bool = True, num_classes=10, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
        :param num_classes: Number of classes in the model
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], progress,num_classes,
                   **kwargs)

class QuantizedModel(nn.Module):
    def __init__(self, model):
        super(QuantizedModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()
        
        # self.names = model.names
        # self.model.nc = model.model.nc
        # Transfer all attributes from the original model
        # for attr_name in dir(model):
        #     attr_value = getattr(model, attr_name)
        #     if not callable(attr_value) and not attr_name.startswith('__'):
        #         print('attribute :',attr_name, attr_value)
        #         setattr(self, attr_name, attr_value)
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def quantize_model_pytorch(model, desired_format, device,train_loader,logger):
    logger.info('\n\nQuantizing the model\n')
    if desired_format == 'int8':
        desired_format = torch.qint8
    elif desired_format == 'uint8':
        desired_format = torch.quint8
    elif desired_format == 'int16':
        desired_format = torch.float16
        print('int16 not available for pytorch | using float16 instead')
    elif desired_format == 'int32':
        desired_format = torch.qint32
    elif desired_format == 'float16':
        desired_format = torch.float16
    else:
        desired_format = torch.float32
    
    if True :
        torch_quant_model = resnet50_quantizable(num_classes=10).to('cpu')
        torch_quant_model.load_state_dict(model.state_dict())
        torch_quant_model.eval()
        module_names = []
        is_previous_conv = False

        for name, module in torch_quant_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                module_names.append(name)
                is_previous_conv = True
                continue
    
            if is_previous_conv and isinstance(module, torch.nn.BatchNorm2d):
                module_names.append(name)
    
            is_previous_conv = False
    
        modules_to_fuse = []
        for i in range(0,len(module_names) - 1,2):
            modules_to_fuse.append([module_names[i], module_names[i + 1]])

        torch_quant_model = torch.quantization.fuse_modules(torch_quant_model, modules_to_fuse)
        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        custom_qconfig = torch.quantization.QConfig(
            activation=MinMaxObserver.with_args(dtype=desired_format),
            weight=torch.quantization.default_per_channel_weight_observer.with_args(dtype=desired_format)
        )
        torch_quant_model.qconfig = custom_qconfig
        

        torch.quantization.prepare(torch_quant_model, inplace=True)
        torch_quant_model.eval()

        with torch.no_grad():
            for enum,(data, target) in enumerate(train_loader):
                data,target = data.to(device),target.to(device)
                torch_quant_model(data)
                if (enum + 1) % 100 == 0:
                    print(
                        f'Quantization  Observer : {enum+1} / {len(train_loader)}')

        torch.quantization.convert(torch_quant_model, inplace=True)
    
    elif device == 'cpu' or True:
        torch_quant_model = quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=desired_format).cpu()
        # torch_quant_model = QuantizedModel(torch_quant_model).to(device)
    else:
    
        # Create a quantized model
        model = QuantizedModel(model.to(device))
        
        model.eval()
    
        # Define the quantization configuration
        quantization_scheme = 'x86' # only int8 for now
        qconfig = torch.quantization.get_default_qconfig(quantization_scheme)
        model.qconfig = qconfig
    
        # Prepare and convert the model for quantization
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        torch_quant_model = model.to(device)
    
        # torch_quant_model = torchvision.models.quantization.resnet50(quantize=True).to(device)

        
    return torch_quant_model


def quantization_aware_training_pytorch(model,train_loader,val_loader, device, num_epochs,logger):
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    fused_model = torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]],
                                                inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
    quant_model = QuantizedModel(fused_model).to(device)
    if device == 'cuda':
        quant_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') #onednn
    else :
        quant_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    quant_model = torch.quantization.prepare_qat(quant_model.train(),inplace=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        quant_model.train()
        for enum,(inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()
            
            outputs = quant_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (enum + 1) % 100 == 0:
                print(
                    f'Quantization epoch {epoch + 1} / {num_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {running_loss / (enum + 1):.4f}')

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        quant_model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(torch.device(device=device))
                val_labels = val_labels.type(torch.LongTensor).to(torch.device(device=device))
                val_outputs = quant_model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_labels)
        
                total_val_loss += val_batch_loss.item() * val_inputs.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                total_val_correct += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples

        print(
            f'\nQuantization | Validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} %\n')
        logger.info(
            f'Quantization | Validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} %')
    
    quant_model.eval()
    final_quant_model = torch.quantization.convert(quant_model.eval(),inplace=True)
    #model = torch.ao.quantization.convert(model.eval(),inplace=True)
    final_quant_model.to(device)
    
    return final_quant_model