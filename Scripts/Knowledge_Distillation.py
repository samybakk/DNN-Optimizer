from tqdm import tqdm

from Distiller import *
from Dataset_utils import *
import torch.nn.functional as F

from yolov5.utils.loss import ComputeLoss


def distill_model_tensorflow(model, teacher_model, dataset_path, batch_size, temperature, alpha, epochs, PWInstance,logger):
    # train_data, train_labels, test_data, test_labels = load_and_prep_images(dataset_path,(28,28), batch_size, batch_size, 0.1,True)
    train_data, train_labels, test_data, test_labels = load_mnist(epochs, batch_size, 0.1)
    
    distiller = TfDistiller(model, teacher_model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=alpha,
        temperature=temperature)
    
    
    distiller.fit(train_data, train_labels, epochs=epochs)
    # distiller.fit(train_data, epochs=epochs, callbacks=callbacks)
    
    print('Evaluating the distilled model')
    distiller.evaluate(test_data, test_labels)
    return model


def distill_model_pytorch(student_model, teacher_model, temperature, alpha, KD_epochs, device,train_loader, val_loader,is_yolo,logger):
    logger.info("\n\nStarting Knowledge Distillation\n")
    teacher_model.eval()
    student_model.train()

    if is_yolo:
        batch_size = 8  # Batch size for training
        img_size = 640  # Input image size
        nc = 52  # Number of classes
        hyp_path = 'yolov5/data/hyps/hyp.scratch-low.yaml'
        with open(hyp_path) as f:
            yolo_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    
        yolo_hyp['label_smoothing'] = 0.0
        student_model.nc = nc  # attach number of classes to model
        student_model.hyp = yolo_hyp  # attach hyperparameters to model

        teacher_model.nc = nc  # attach number of classes to model
        teacher_model.hyp = yolo_hyp  # attach hyperparameters to model

        # for param in teacher_model.parameters():
        #     param.requires_grad = False
        #
        # from utils.general import non_max_suppression
        #
        # criterion = nn.KLDivLoss(reduction='batchmean')
        # optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
        #
        # for epoch in range(KD_epochs):
        #     train_loss = 0.0
        #     val_loss = 0.0
        #
        #     # mloss = torch.zeros(3, device=device)  # mean losses
        #     pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        #     print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        #     for i, (imgs, targets, paths, _) in pbar:
        #         inputs = imgs.to(device).float() / 255.0
        #         targets = targets.to(device)
        #         inputs, targets = inputs.to(device), targets.to(device)
        #
        #         optimizer.zero_grad()
        #
        #         with torch.no_grad():
        #             teacher_outputs = teacher_model(inputs) #/ temperature
        #             # teacher_outputs = non_max_suppression(teacher_outputs, conf_thres=0.001, iou_thres=0.6)[0]
        #             teacher_outputs_clone = teacher_outputs#.clone()
        #
        #         student_outputs = student_model(inputs)  #/ temperature
        #         # student_outputs = non_max_suppression(student_outputs, conf_thres=0.001, iou_thres=0.6)[0]
        #         student_output_clone = student_outputs#.clone()
        #
        #         if teacher_outputs is None or student_outputs is None:
        #             continue
        #
        #         # Compute loss for class probabilities
        #         class_loss = criterion(F.log_softmax(student_output_clone[..., 5:], dim=-1), F.softmax(teacher_outputs_clone[..., 5:], dim=-1))
        #         class_loss *= alpha * temperature * temperature
        #
        #         # Compute loss for bounding box coordinates
        #         bbox_loss = criterion(student_output_clone[..., :4], teacher_outputs_clone[..., :4])
        #         bbox_loss *= alpha * temperature * temperature
        #
        #         # Compute total loss
        #         loss = class_loss + bbox_loss
        #
        #         loss.backward()
        #         optimizer.step()
        #
        #         train_loss += loss.item()
        #
        #     train_loss /= len(train_loader)
        #
        #     with torch.no_grad():
        #         for i, (inputs, targets, _) in enumerate(val_loader):
        #             inputs, targets = inputs.to(device), targets.to(device)
        #
        #             teacher_outputs = teacher_model(inputs)
        #             teacher_outputs = non_max_suppression(teacher_outputs, conf_thres=0.001, iou_thres=0.6)[0]
        #
        #             student_outputs = student_model(inputs)
        #             student_outputs = non_max_suppression(student_outputs, conf_thres=0.001, iou_thres=0.6)[0]
        #
        #             if teacher_outputs is None or student_outputs is None:
        #                 continue
        #
        #             # Compute loss for class probabilities
        #             class_loss = criterion(F.log_softmax(student_outputs[..., 5:], dim=-1), F.softmax(teacher_outputs[..., 5:], dim=-1))
        #             class_loss *= alpha * temperature * temperature
        #
        #             # Compute loss for bounding box coordinates
        #             bbox_loss = criterion(student_outputs[..., :4], teacher_outputs[..., :4])
        #             bbox_loss *= alpha * temperature * temperature
        #
        #             # Compute total loss
        #             loss = class_loss + bbox_loss
        #
        #             val_loss += loss.item()
        #
        #         val_loss /= len(val_loader)
        #
        #     logger.info(f'Epoch {epoch + 1}/{KD_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        teacher_model.train()
        dump_image = torch.zeros((1, 3, img_size, img_size), device=device)
        targets = torch.Tensor([[0, 0, 0, 0, 0, 0]]).to(device)
        _, features, _ = student_model(dump_image)  # forward
        _, teacher_feature, _ = teacher_model(dump_image)

        _, student_channel, student_out_size, _, _ = features.shape
        _, teacher_channel, teacher_out_size, _, _ = teacher_feature.shape

        stu_feature_adapt = nn.Sequential(
            nn.Conv2d(student_channel, teacher_channel, 3, padding=1, stride=int(student_out_size / teacher_out_size)),
            nn.ReLU()).to(device)

        yolo_optimizer = torch.optim.SGD(student_model.parameters(), lr=yolo_hyp['lr0'], momentum=yolo_hyp['momentum'],
                                         weight_decay=yolo_hyp['weight_decay'])
        yolo_compute_loss = ComputeLoss(student_model)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(yolo_optimizer, milestones=[round(yolo_hyp['lrf'] * 0.8),
                                                                                     round(yolo_hyp['lrf'] * 0.9)],
                                                         gamma=0.1)
        
        for epoch in range(KD_epochs):
            teacher_model.eval()
            student_model.train()
            mloss = torch.zeros(3, device=device)  # mean losses
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
            print(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            
            for i, (imgs, targets, paths, _) in pbar:
                imgs = imgs.to(device).float() / 255.0
                targets = targets.to(device)
            
                # Forward pass
                pred, features, _ = student_model(imgs, target=targets)  # forward
                _, teacher_feature, mask = teacher_model(imgs, target=targets)
                loss, loss_items = yolo_compute_loss(pred, targets, teacher_feature.detach(), stu_feature_adapt(features),
                                                mask.detach())  # loss scaled by batch_size

                # Backward pass
                loss.backward()
            
                # Optimize
                yolo_optimizer.step()
                yolo_optimizer.zero_grad()
            
                # Print progress
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch + 1}/{KD_epochs}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # print(f'Epoch {epoch}, Batch {batch_i}, Loss: {loss.item()}')
            
                # Update the learning rate
            scheduler.step()
    
    else :
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(KD_epochs):
            loss_sum = 0.0
            for enum, (data, target) in enumerate(train_loader):
                data, target = data.to(torch.device(device=device)), target.type(torch.LongTensor).to(torch.device(device=device))
                
                optimizer.zero_grad()
                student_pred = student_model(data)
                
                with torch.no_grad():
                    teacher_pred = teacher_model(data)
    
                # student_loss_sum = 0.0
                # teacher_loss_sum = 0.0
                student_loss = criterion(student_pred, target)
                # for teacher_pred, target_pred, student_pred in zip(teacher_output, target, output):
                #     student_loss_sum += nn.functional.cross_entropy(student_pred, target_pred)
                #     _, student_pred = torch.max(student_pred, 0)
                #     _, teacher_pred = torch.max(teacher_pred, 0)
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