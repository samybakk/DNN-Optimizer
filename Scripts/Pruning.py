import torch.nn
import torch.nn.utils.prune as prune
import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import ConstantSparsity
from torch.optim import lr_scheduler

from utilities import *
from Dataset_utils import *
import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from utils.loss import ComputeLoss




# a function that takes a tensorflow model and returns a pruned version of it using pruning_ratio as the pruning ratio without re-training the model and without data
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


def prune_model_tensorflow(model, pruning_ratio,PEpochs,batch_size,convert_tflite,quantization,logger):


    # train_data, train_labels, test_data, test_labels, end_step = load_mnist(epochs=PEpochs, batch_size=batch_size)
    #
    # prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    # pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=pruning_ratio / 3,
    #                                                              final_sparsity=pruning_ratio,
    #                                                              begin_step=0,
    #                                                              end_step=end_step)}
    #
    # model_for_pruning = prune_low_magnitude(model, **pruning_params)
    # model_for_pruning.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                           metrics=['accuracy'])
    #
    # model_for_pruning.summary()
    # logdir = tempfile.mkdtemp()
    #
    # callbacks = [
    #     tfmot.sparsity.keras.UpdatePruningStep(),
    #     tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),]
    #
    # model_for_pruning.fit(train_data, train_labels,batch_size=batch_size, epochs=PEpochs, validation_split=0.1,
    # callbacks=callbacks)
    # model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    #
    # if convert_tflite == True and quantization == False:
    #     converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    #     tflite_model = converter.convert()
    #     return tflite_model
    #
    # else:
    #     return model_for_export
    train_data, train_labels, val_data, val_labels, end_step = load_mnist(epochs=PEpochs, batch_size=batch_size)

    if PEpochs == 0 :
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(pruning_ratio, 0),
            'block_size': (1, 1),
            'block_pooling_type': 'AVG'
        }

    else :
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.00,
                                                               final_sparsity=pruning_ratio,
                                                               begin_step=0,
                                                               end_step=end_step),
            'block_size': (1, 1),
            'block_pooling_type': 'AVG'
        }

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy,
                              metrics=['accuracy'])

   
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]

    model_for_pruning.fit(train_data, train_labels, batch_size=batch_size, epochs=PEpochs, validation_data=(val_data, val_labels), callbacks=callbacks)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    return model_for_export
    
def pytorch_random_unstructured_prune(model, pruning_ratio, pruning_epochs, device, train_loader, val_loader,logger,is_yolo):
    if is_yolo:
        batch_size = 8  # Batch size for training
        img_size = 640  # Input image size
        nc = 52  # Number of classes
        hyp_path = 'yolov5/data/hyps/hyp.scratch-low.yaml'
        with open(hyp_path) as f:
            yolo_hyp = yaml.load(f, Loader=yaml.SafeLoader)
        
        yolo_hyp['label_smoothing'] = 0.0
        model.nc = nc  # attach number of classes to model
        model.hyp = yolo_hyp  # attach hyperparameters to model
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        
        yolo_optimizer = torch.optim.SGD(model.parameters(), lr=yolo_hyp['lr0'], momentum=yolo_hyp['momentum'],
                                         weight_decay=yolo_hyp['weight_decay'])
        yolo_compute_loss = ComputeLoss(model)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(yolo_optimizer, milestones=[round(yolo_hyp['lrf'] * 0.8),
                                                                                     round(yolo_hyp['lrf'] * 0.9)],
                                                         gamma=0.1)
    
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    for module, parameter_name in parameters_to_prune:
        prune.random_unstructured(module, name=parameter_name, amount=pruning_ratio)

    model.train()
    for epoch in range(pruning_epochs):
        if is_yolo:
            mloss = torch.zeros(3, device=device)  # mean losses
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
            print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            for i, (imgs, targets, paths, _) in pbar:
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
            
                # Forward pass
                pred = model(imgs)  # forward
                loss, loss_items = yolo_compute_loss(pred, targets)
            
                # Backward pass
                loss.backward()
            
                # Optimize
                yolo_optimizer.step()
                yolo_optimizer.zero_grad()
            
                # Print progress
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch + 1}/{pruning_epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # print(f'Epoch {epoch}, Batch {batch_i}, Loss: {loss.item()}')
            
                # Update the learning rate
            scheduler.step()
    
        else:
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
                        f'Pruning epoch {epoch + 1} / {pruning_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {sum_loss / (enum + 1):.4f}')
        
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
        
            scheduler.step()
        
            print(
                f'\nPruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pruning_ratio * 100:.0f} % of weights pruned\n')
            logger.info(
                f'Pruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pruning_ratio * 100:.0f} % of weights pruned')

    # Supprime les poids de facon permanente
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def pytorch_random_structured_prune(model, pruning_ratio, pruning_epochs, device, train_loader, val_loader,logger,is_yolo):
    if is_yolo:
        batch_size = 8  # Batch size for training
        img_size = 640  # Input image size
        nc = 52  # Number of classes
        hyp_path = 'yolov5/data/hyps/hyp.scratch-low.yaml'
        with open(hyp_path) as f:
            yolo_hyp = yaml.load(f, Loader=yaml.SafeLoader)
        
        yolo_hyp['label_smoothing'] = 0.0
        model.nc = nc  # attach number of classes to model
        model.hyp = yolo_hyp  # attach hyperparameters to model
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        
        yolo_optimizer = torch.optim.SGD(model.parameters(), lr=yolo_hyp['lr0'], momentum=yolo_hyp['momentum'],
                                         weight_decay=yolo_hyp['weight_decay'])
        yolo_compute_loss = ComputeLoss(model)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(yolo_optimizer, milestones=[round(yolo_hyp['lrf'] * 0.8),
                                                                                     round(yolo_hyp['lrf'] * 0.9)],
                                                         gamma=0.1)
    
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    for module, parameter_name in parameters_to_prune:
        prune.random_structured(module, name=parameter_name, amount=pruning_ratio)

    model.train()
    for epoch in range(pruning_epochs):
        if is_yolo:
            mloss = torch.zeros(3, device=device)  # mean losses
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
            print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            for i, (imgs, targets, paths, _) in pbar:
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
            
                # Forward pass
                pred = model(imgs)  # forward
                loss, loss_items = yolo_compute_loss(pred, targets)
            
                # Backward pass
                loss.backward()
            
                # Optimize
                yolo_optimizer.step()
                yolo_optimizer.zero_grad()
            
                # Print progress
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch + 1}/{pruning_epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # print(f'Epoch {epoch}, Batch {batch_i}, Loss: {loss.item()}')
            
                # Update the learning rate
            scheduler.step()
    
        else:
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
                        f'Pruning epoch {epoch + 1} / {pruning_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {sum_loss / (enum + 1):.4f}')
        
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
        
            scheduler.step()
        
            print(
                f'\nPruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pruning_ratio * 100:.0f} % of weights pruned\n')
            logger.info(
                f'Pruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pruning_ratio * 100:.0f} % of weights pruned')

    # Supprime les poids de facon permanente
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def pytorch_magnitude_prune(model, pruning_ratio, pruning_epochs, device, train_loader, val_loader,logger,is_yolo):
    

    if is_yolo:
        batch_size = 8  # Batch size for training
        img_size = 640  # Input image size
        nc = 52  # Number of classes
        hyp_path = 'yolov5/data/hyps/hyp.scratch-low.yaml'
        with open(hyp_path) as f:
            yolo_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    
        yolo_hyp['label_smoothing'] = 0.0
        model.nc = nc  # attach number of classes to model
        model.hyp = yolo_hyp  # attach hyperparameters to model
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    
        yolo_optimizer = torch.optim.SGD(model.parameters(), lr=yolo_hyp['lr0'], momentum=yolo_hyp['momentum'],
                                         weight_decay=yolo_hyp['weight_decay'])
        yolo_compute_loss = ComputeLoss(model)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(yolo_optimizer, milestones=[round(yolo_hyp['lrf'] * 0.8),
                                                                                     round(yolo_hyp['lrf'] * 0.9)],
                                                         gamma=0.1)
        
    else :
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=pruning_ratio)
    
    
    
    model.train()
    for epoch in range(pruning_epochs):
        if is_yolo:
            mloss = torch.zeros(3, device=device)  # mean losses
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
            print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            for i, (imgs, targets, paths, _) in pbar:
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
            
                # Forward pass
                pred = model(imgs)  # forward
                loss, loss_items = yolo_compute_loss(pred, targets)
            
                # Backward pass
                loss.backward()
            
                # Optimize
                yolo_optimizer.step()
                yolo_optimizer.zero_grad()
            
                # Print progress
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch + 1}/{pruning_epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # print(f'Epoch {epoch}, Batch {batch_i}, Loss: {loss.item()}')
            
                # Update the learning rate
            scheduler.step()
    
        else:
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
                        f'Pruning epoch {epoch + 1} / {pruning_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {sum_loss / (enum + 1):.4f}')
        
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

            scheduler.step()
        
            print(
                f'\nPruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pruning_ratio  * 100:.0f} % of weights pruned\n')
            logger.info(
                f'Pruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pruning_ratio  * 100:.0f} % of weights pruned')

    # Supprime les poids de facon permanente
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def prune_dynamic_model_pytorch(model, pruning_ratio, pruning_epochs, device, train_loader, val_loader,logger,is_yolo,magnitude_pruning,channel_pruning):
    logger.info(f'\n\nPruning dynamic model\n')

    if is_yolo:
        batch_size = 8  # Batch size for training
        img_size = 640  # Input image size
        nc = 52  # Number of classes
        hyp_path = 'yolov5/data/hyps/hyp.scratch-low.yaml'
        with open(hyp_path) as f:
            yolo_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    
        yolo_hyp['label_smoothing'] = 0.0
        model.nc = nc  # attach number of classes to model
        model.hyp = yolo_hyp  # attach hyperparameters to model
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    
        yolo_optimizer = torch.optim.SGD(model.parameters(), lr=yolo_hyp['lr0'], momentum=yolo_hyp['momentum'],
                                         weight_decay=yolo_hyp['weight_decay'])
        yolo_compute_loss = ComputeLoss(model)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(yolo_optimizer, milestones=[round(yolo_hyp['lrf'] * 0.8),
                                                                                     round(yolo_hyp['lrf'] * 0.9)],
                                                         gamma=0.1)

    else:
        normal_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(normal_optimizer, step_size=1, gamma=0.1)
    
    pr_per_epoch = pruning_ratio / pruning_epochs
    model.train()
    for epoch in range(pruning_epochs):
        for name, module in model.named_modules():
            
            
            # if magnitude_pruning and channel_pruning :
            if isinstance(module, torch.nn.Conv2d):
                # n=2  pour les conv2d --> pruning avec l2 norm
                prune.ln_structured(module, name='weight', amount=pr_per_epoch, n=2,dim=0)

            if isinstance(module, torch.nn.Linear):
                # n=1  pour les linear layers --> pruning avec l1 norm
                prune.ln_structured(module, name='weight', amount=pr_per_epoch, n=1,dim=0)
            # elif magnitude_pruning :
            #     if isinstance(module, torch.nn.Conv2d):
            #         # n=2  pour les conv2d --> pruning avec l2 norm
            #         prune.ln_structured(module, name='weight', amount=pr_per_epoch, n=2,dim=0)
            #
            #     if isinstance(module, torch.nn.Linear):
            #         # n=1  pour les linear layers --> pruning avec l1 norm
            #         prune.ln_structured(module, name='weight', amount=pr_per_epoch, n=1,dim=0)
            #
            # elif  channel_pruning:
            #     for name, module in model.named_modules():
            #         if isinstance(module, torch.nn.Conv2d):
            #             # Channel-wise pruning with L1-norm for convolutional layers
            #             prune.ln_structured(module, name='weight', amount=pr_per_epoch, n=1, dim=1)
         
        
        if is_yolo :
            mloss = torch.zeros(3, device=device)  # mean losses
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
            print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            for i, (imgs, targets, paths, _) in pbar:
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
        
                # Forward pass
                pred = model(imgs)  # forward
                loss, loss_items = yolo_compute_loss(pred, targets)
        
                # Backward pass
                loss.backward()
        
                # Optimize
                yolo_optimizer.step()
                yolo_optimizer.zero_grad()
        
                # Print progress
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch+1}/{pruning_epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # print(f'Epoch {epoch}, Batch {batch_i}, Loss: {loss.item()}')
        
                # Update the learning rate
            scheduler.step()
    
        else :
            sum_loss = 0.0
            for enum, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device=device), labels.type(torch.LongTensor).to(device=device)
                normal_optimizer.zero_grad()
        
                # Convert the input tensor to the same data type as the weights tensor
                weight_tensor = next(model.parameters()).data
                inputs = inputs.to(weight_tensor.dtype)
        
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
        
                loss.backward()
                normal_optimizer.step()
                if (enum + 1) % 100 == 0:
                    print(
                        f'Pruning epoch {epoch + 1} / {pruning_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {sum_loss / (enum + 1):.4f}')
    
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

            scheduler.step()
    
            print(
                f'\nPruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pr_per_epoch * (epoch + 1) * 100:.0f} % of weights pruned\n')
            logger.info(
                f'Pruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pr_per_epoch * (epoch + 1) * 100:.0f} % of weights pruned')

    # Supprime les poids de facon permanente
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def prune_global_model_pytorch(model, pruning_ratio, logger):
    logger.info(f'\n\nPruning global model\n')
    
    
    
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio
    )

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def prune_global_structured_model_pytorch(model, pruning_ratio, logger):
    logger.info(f'\n\nPruning global model\n')
    
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.LnStructured,
        amount=pruning_ratio
    )
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
    return model


def prune_global_dynamic_model_pytorch(model, pruning_ratio, pruning_epochs, device, train_loader, val_loader, logger,is_yolo):
    logger.info(f'\n\nPruning global model\n')

    if is_yolo:
        batch_size = 8  # Batch size for training
        img_size = 640  # Input image size
        nc = 52  # Number of classes
        hyp_path = 'yolov5/data/hyps/hyp.scratch-low.yaml'
        with open(hyp_path) as f:
            yolo_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    
        yolo_hyp['label_smoothing'] = 0.0
        model.nc = nc  # attach number of classes to model
        model.hyp = yolo_hyp  # attach hyperparameters to model
        # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    
        yolo_optimizer = torch.optim.SGD(model.parameters(), lr=yolo_hyp['lr0'], momentum=yolo_hyp['momentum'],
                                         weight_decay=yolo_hyp['weight_decay'])
        yolo_compute_loss = ComputeLoss(model)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(yolo_optimizer, milestones=[round(yolo_hyp['lrf'] * 0.8),
                                                                                     round(yolo_hyp['lrf'] * 0.9)],
                                                         gamma=0.1)

    else:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    
    pr_per_epoch = pruning_ratio / pruning_epochs
    for epoch in range(pruning_epochs):
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pr_per_epoch
        )

        if is_yolo:
            mloss = torch.zeros(3, device=device)  # mean losses
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
            print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            for i, (imgs, targets, paths, _) in pbar:
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
        
                # Forward pass
                pred = model(imgs)  # forward
                loss, loss_items = yolo_compute_loss(pred, targets)
        
                # Backward pass
                loss.backward()
        
                # Optimize
                yolo_optimizer.step()
                yolo_optimizer.zero_grad()
        
                # Print progress
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch + 1}/{pruning_epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # print(f'Epoch {epoch}, Batch {batch_i}, Loss: {loss.item()}')
        
                # Update the learning rate
            scheduler.step()

        else:
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
                        f'Pruning epoch {epoch + 1} / {pruning_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {sum_loss / (enum + 1):.4f}')
    
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

            scheduler.step()
            
            print(
                f'\nPruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pr_per_epoch * (epoch + 1) * 100:.0f} % of weights pruned\n')
            logger.info(
                f'Pruning epoch {epoch + 1} / {pruning_epochs}  | validation loss : {val_loss:.4f} | validation accuracy : {val_accuracy * 100:.2f} % | {pr_per_epoch * (epoch + 1) * 100:.0f} % of weights pruned')

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