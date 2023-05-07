import torch.nn.utils.prune as prune
import numpy as np
from utilities import *
from Dataset_utils import *
import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot





# a funciton that takes a tensorflow model and returns a pruned version of it using pruning_ratio as the pruning ratio without re-training the model and without data
def basic_prune_model_tensorflow(model, pruning_ratio,logger):
    # Get weights of the model
    weights = model.get_weights()
    
    # Flatten the weights into a 1D array for sorting
    flattened_weights = np.concatenate([layer.flatten() for layer in weights])
    
    # Compute the number of weights to prune
    num_weights_to_prune = int(len(flattened_weights) * pruning_ratio)
    
    # Sort the weights in descending order
    sorted_weights = np.sort(np.abs(flattened_weights))[::-1]
    
    # Find the threshold weight value to prune
    threshold = sorted_weights[num_weights_to_prune]
    
    # Set the weights above the threshold to zero
    pruned_weights = [np.where(np.abs(w) >= threshold, w, 0) for w in weights]
    
    # Set the pruned weights back into the model
    model.set_weights(pruned_weights)
    
    return model


def prune_model_tensorflow(model, pruning_ratio,PEpochs,batch_size,convert_tflite,quantization,PWInstance,logger):


    train_data, train_labels, test_data, test_labels, end_step = load_mnist(epochs=PEpochs, batch_size=batch_size,
    validation_split=0.1)

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=pruning_ratio / 3,
                                                                 final_sparsity=pruning_ratio,
                                                                 begin_step=0,
                                                                 end_step=end_step)}

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    model_for_pruning.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    model_for_pruning.summary()
    logdir = tempfile.mkdtemp()

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        TfPlotPruning(PWInstance)]

    model_for_pruning.fit(train_data, train_labels,batch_size=batch_size, epochs=PEpochs, validation_split=0.1,
    callbacks=callbacks)
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    if convert_tflite == True and quantization == False:
        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        tflite_model = converter.convert()
        return tflite_model

    else:
        return model_for_export


def basic_prune_finetune_model_pytorch(model, pruning_ratio, pruning_epochs, device, train_loader, val_loader,logger):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # n=2  pour les conv2d --> pruning avec l2 norm
            prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)

        if isinstance(module, torch.nn.Linear):
            # n=1  pour les linear layers --> pruning avec l1 norm
            prune.ln_structured(module, name='weight', amount=pruning_ratio, n=1, dim=0)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(pruning_epochs):
        sum_loss = 0.0
        for enum, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device=device), labels.type(torch.LongTensor).to(device=device)
            optimizer.zero_grad()
            # Convert the input tensor to the same data type as the weights tensor
            weight_tensor = next(model.parameters()).data
            inputs = inputs.to(weight_tensor.dtype)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            if (enum + 1) % 100 == 0:
                print(f'epoch {epoch+1} / {pruning_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {sum_loss / (enum + 1):.4f}')

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(torch.device(device=device))
                val_labels = val_labels.type(torch.LongTensor).to(torch.device(device=device))
                val_outputs = model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_labels)
        
                total_val_loss += val_batch_loss.item() * val_inputs.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                total_val_correct += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples

        print(
            f'\nPruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy*100:.2f} %\n')
    
    # Supprime les poids de facon permanente
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def prune_dynamic_model_pytorch(model, pruning_ratio, pruning_epochs, device, train_loader, val_loader,logger):
    logger.info(f'\n\nPruning dynamic model\n')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    pr_per_epoch = pruning_ratio / pruning_epochs
    model.train()
    for epoch in range(pruning_epochs):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # n=2  pour les conv2d --> pruning avec l2 norm
                prune.ln_structured(module, name='weight', amount=pr_per_epoch, n=2, dim=0)

            if isinstance(module, torch.nn.Linear):
                # n=1  pour les linear layers --> pruning avec l1 norm
                prune.ln_structured(module, name='weight', amount=pr_per_epoch, n=1, dim=0)
        
        sum_loss = 0.0
        for enum, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device=device), labels.type(torch.LongTensor).to(device=device)
            optimizer.zero_grad()
            
            # Convert the input tensor to the same data type as the weights tensor
            weight_tensor = next(model.parameters()).data
            inputs = inputs.to(weight_tensor.dtype)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            if (enum + 1) % 100 == 0:
                print(
                    f'Pruning epoch {epoch+1} / {pruning_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {sum_loss / (enum + 1):.4f}')

        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(torch.device(device=device))
                val_labels = val_labels.type(torch.LongTensor).to(torch.device(device=device))
                val_outputs = model(val_inputs)
                val_batch_loss = criterion(val_outputs, val_labels)
        
                total_val_loss += val_batch_loss.item() * val_inputs.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                total_val_correct += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)

        val_loss = total_val_loss / total_val_samples
        val_accuracy = total_val_correct / total_val_samples

        print(
            f'\nPruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy*100:.2f} %\n')
        logger.info(f'Pruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy*100:.2f} %')
    
    # Supprime les poids de facon permanente
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def basic_magnitude_prune_model_pytorch(model, pruning_ratio,logger):
    masks = []
    model_parameters = [x for x in model.parameters()]
    for param in model_parameters:
        masks.append(torch.ones_like(param))
    
    for param, mask in zip(model_parameters, masks):
        sorted_param = param.view(-1).abs().sort()[0]
        n_elements_to_prune = int(sorted_param.numel() * pruning_ratio)
        pruning_threshold = sorted_param[n_elements_to_prune]
        mask[param.abs() < pruning_threshold] = 0
    
    for param, mask in zip(model_parameters, masks):
        param.data.mul_(mask)
    
    return model


def GAL_prune_pytorch(model, pruning_ratio, PEpochs, batch_size, device, PWInstance,logger):
    backbone_id = get_pytorch_backbone_layers_id(model)
    backbone_model, head_model = extract_backbone_layers(model, backbone_id)
    pruned_model = prune_backbone_with_gal(backbone_model, pruning_ratio, PEpochs, device)
    
    return pruned_model