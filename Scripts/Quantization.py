import tensorflow.lite as tflite
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


def quantize_model_pytorch(model, desired_format, device,val_loader,logger):
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
    
    if device == 'cpu':
        torch_quant_model = quantization.quantize_dynamic(model, {nn.Linear}, dtype=desired_format)
    
    else:
        # Create a quantized model
        model = QuantizedModel(model)
        model.eval()
        
        # Define the quantization configuration
        quantization_scheme = 'x86'
        qconfig = torch.quantization.get_default_qconfig(quantization_scheme)
        model.qconfig = qconfig
        
        # Prepare and convert the model for quantization
        torch.quantization.prepare(model, inplace=False)
        torch.quantization.convert(model)
        torch_quant_model = model.to(device)

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        val_criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(torch.device(device=device))
                val_labels = val_labels.type(torch.LongTensor).to(torch.device(device=device))
                val_outputs = model(val_inputs)
                val_batch_loss = val_criterion(val_outputs, val_labels)
        
                total_val_loss += val_batch_loss.item() * val_inputs.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                total_val_correct += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples

        print(
            f'\nQuantization | Validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} %\n')
        logger.info(f'Quantization | Validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} %')
        
    return torch_quant_model