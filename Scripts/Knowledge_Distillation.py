import os
from copy import deepcopy
from datetime import datetime

import shutil

from Distiller import *
from Dataset_utils import *
import torch.nn.functional as F

from utils.torch_utils import de_parallel, ModelEMA
from train import main as train_yolov5_kd


def distill_model_tensorflow(model, teacher_model, train_data, train_labels, val_data, val_labels, temperature, alpha, epochs, PWInstance):
    
    distiller = TfDistiller(model, teacher_model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=alpha,
        temperature=temperature)
    
    callbacks = [TfPlotDK(PWInstance)]
    
    distiller.fit(train_data, train_labels, epochs=epochs, callbacks=callbacks)
    
    print('Evaluating the distilled model')
    distiller.evaluate(val_data, val_labels)
    return model


def distill_model_yolo(student_model, teacher_model, dataset_path, KD_epochs, logger,nbr_classes,save_name,batch_size,device):
    logger.info("\n\nStarting Knowledge Distillation\n")
    
    yolo_optimizer = torch.optim.SGD(student_model.parameters(), 0.01,
                                     momentum=0.937,
                                     weight_decay=0.0005)
    ema = ModelEMA(student_model)
    saved_model_path = 'Models' + os.sep + 'Temp' + os.sep + save_name + '_yolov5.pt'
    ckpt = {
        'epoch': 0,
        'best_fitness': 0,
        'model': deepcopy(de_parallel(student_model)).half(),
        'ema': deepcopy(ema.ema).half(),
        'updates': ema.updates,
        'optimizer': yolo_optimizer.state_dict(),
        'opt': None,
        'git': None,  # {remote, branch, commit} if a git repo
        'date': datetime.now().isoformat()}
    
    # Save last, best and delete
    torch.save(ckpt, saved_model_path)
    del ckpt
    
    class Args:
        def __init__(self, dataset_path, teacher_model, KD_epochs):
            self.weights = saved_model_path
            self.teacher_weight = teacher_model
            self.cfg = ''
            self.data = dataset_path + '/data.yaml'
            self.hyp = 'yolov5-knowledge-distillation/data/hyps/hyp.scratch-low.yaml'
            self.epochs = KD_epochs
            self.batch_size = batch_size
            self.imgsz = 640
            self.rect = False
            self.resume = False
            self.nosave = False
            self.noval = False
            self.noautoanchor = False
            self.noplots = False
            self.evolve = None
            self.bucket = ''
            self.cache = 'ram'
            self.image_weights = False
            self.device = '0'
            self.multi_scale = False
            self.single_cls = False
            self.optimizer = 'SGD'
            self.sync_bn = False
            self.workers = 2
            self.project = 'results' + os.sep + save_name
            
            self.name = 'yolov5kd'
            
            self.exist_ok = False
            self.quad = False
            self.cos_lr = False
            self.label_smoothing = 0.0
            self.patience = 100
            self.freeze = [0]
            self.save_period = -1
            self.seed = 0
            self.local_rank = -1
            self.entity = None
            self.upload_dataset = False
            self.bbox_interval = -1
            self.artifact_alias = 'latest'
    
    args_dict = Args(dataset_path, teacher_model, KD_epochs)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    train_yolov5_kd(args_dict)
    load_path = 'results' + os.sep + save_name+ os.sep + 'yolov5kd'+ os.sep + 'weights' + os.sep + 'best.pt'
    model_dict = torch.load( load_path , map_location='cpu')
    model_name = saved_model_path.split(os.sep)[-1]
    model = load_pytorch_model(model_dict, device, model_name, number_of_classes=nbr_classes)
    
    os.remove(saved_model_path)
    
    return model


def distill_model_pytorch(student_model, teacher_model, temperature, alpha, KD_epochs, device,train_loader, val_loader, PWInstance,logger):
    logger.info("\n\nStarting Knowledge Distillation\n")
    teacher_model.eval()
    student_model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=0.0001, momentum=0.9)
    plotdk = PTPlotDK(PWInstance)
    for epoch in range(KD_epochs):
        loss_sum = 0.0
        for enum, (data, target) in enumerate(train_loader):
            data, target = data.to(torch.device(device=device)), target.type(torch.LongTensor).to(torch.device(device=device))
            
            optimizer.zero_grad()
            student_pred = student_model(data)
            
            with torch.no_grad():
                teacher_pred = teacher_model(data)

            student_loss = criterion(student_pred, target)
            teacher_loss = nn.KLDivLoss()(F.log_softmax(student_pred/temperature, dim=0),
                             F.softmax(teacher_pred/temperature, dim=0)) * ( temperature * temperature)

            # student_loss = student_loss_sum / data.size(0)
            # teacher_loss = teacher_loss_sum / data.size(0)
            
            loss = alpha * student_loss + (1 - alpha) * teacher_loss
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            if (enum + 1) % 100 == 0:
                print(
                    f'KD Epoch {epoch + 1} / {KD_epochs} : {enum + 1} / {len(train_loader)} | batch loss : {loss:.4f} | average loss : {loss_sum / (enum + 1):.4f}')

        student_model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = student_model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)
        val_accuracy = total_correct / total_samples
        print(f'KD Epoch {epoch + 1}, Validation Accuracy: {val_accuracy * 100:.2f} %')
        logger.info(f'KD Epoch {epoch + 1}, Validation Accuracy: {val_accuracy * 100:.2f} %')
        plotdk.on_epoch_end(epoch + 1, 100 * val_accuracy)
        student_model.train()
        
    return student_model


def exp_distill_model_pytorch(student_model, teacher_model, dataset_path, batch_size, temperature, alpha, epochs,device, PWInstance, logger):
    # Set models to train mode
    teacher_model.train()
    student_model.train()
    
    # Define temperature and weighting for distillation loss
    train_loader, test_loader = load_pytorch_dataset(dataset_path, batch_size=batch_size)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    
    # Iterate over training data
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Compute teacher model outputs and logits
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
                teacher_logits = teacher_outputs[:, :, 4:]
            
            # Compute student model outputs and logits
            student_outputs = student_model(data)
            student_logits = student_outputs[:, :, 4:]
            
            # Compute classification and regression losses
            cls_loss = F.binary_cross_entropy_with_logits(student_logits[:, :, :1], targets[:, :, :1])
            reg_loss = F.mse_loss(student_logits[:, :, 1:], targets[:, :, 1:])
            
            # Compute distillation loss
            soft_teacher_logits = F.softmax(teacher_logits / temperature, dim=2)
            soft_student_logits = F.softmax(student_logits / temperature, dim=2)
            dist_loss = F.kl_div(soft_student_logits, soft_teacher_logits,
                                 reduction='batchmean') * temperature * temperature
            
            # Combine losses and apply backpropagation
            loss = alpha * dist_loss + (1 - alpha) * (cls_loss + reg_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in test_loader:
        images = images.view(-1, 640 * 640).to(torch.device(device=device))
        labels = labels.to(torch.device(device=device))
        
        output = student_model(images)
        loss = nn.functional.cross_entropy(output, labels)
        loss_sum += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
class TfPlotDK(keras.callbacks.Callback):
    def __init__(self, PW):
        self.PW = PW
        self.PW.on_dk_epoch_end_signal.connect(self.PW.update_dk_graph_data)

    def on_epoch_end(self, epoch, logs={}):
        self.PW.on_dk_epoch_end_signal.emit(epoch+1, logs['sparse_categorical_accuracy'])
        
class PTPlotDK():
    def __init__(self, PW):
        self.PW = PW
        self.PW.on_dk_epoch_end_signal.connect(self.PW.update_dk_graph_data)

    def on_epoch_end(self, epoch, loss):
        self.PW.on_dk_epoch_end_signal.emit(epoch, loss)