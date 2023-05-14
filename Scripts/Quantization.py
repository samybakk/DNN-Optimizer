import tensorflow.lite as tflite
import torchvision.models.quantization

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


class QuantizedModel(nn.Module):
    def __init__(self, model):
        super(QuantizedModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def quantize_model_pytorch(model, desired_format, device,logger):
    logger.info('\n\nQuantizing the model\n')
    if desired_format == 'int8':
        desired_format = torch.qint8
    elif desired_format == 'int16':
        desired_format = torch.float16
        print('int16 not available for pytorch | using float16 instead')
    elif desired_format == 'int32':
        desired_format = torch.qint32
    elif desired_format == 'float16':
        desired_format = torch.float16
    else:
        desired_format = torch.float32
    
    # if device == 'cpu':
    #     torch_quant_model = quantization.quantize_dynamic(model, {nn.Linear}, dtype=desired_format)
    #
    # else:
    
    # Create a quantized model
    # model = QuantizedModel(model)
    torch_quant_model = torchvision.models.quantization.resnet50(quantize=True)
    # model.eval()
    #
    # # Define the quantization configuration
    # quantization_scheme = 'x86' # only int8 for now
    # qconfig = torch.quantization.get_default_qconfig(quantization_scheme)
    # model.qconfig = qconfig
    #
    # # Prepare and convert the model for quantization
    # torch.quantization.prepare(model, inplace=False)
    # torch.quantization.convert(model)
    # torch_quant_model = model.to(device)

        
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