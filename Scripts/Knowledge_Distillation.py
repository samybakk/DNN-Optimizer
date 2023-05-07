from Distiller import *
from Dataset_utils import *
import torch.nn.functional as F


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
    
    callbacks = [TfPlotDK(PWInstance)]
    
    distiller.fit(train_data, train_labels, epochs=epochs, callbacks=callbacks)
    # distiller.fit(train_data, epochs=epochs, callbacks=callbacks)
    
    print('Evaluating the distilled model')
    distiller.evaluate(test_data, test_labels)
    return model


def distill_model_pytorch(student_model, teacher_model, temperature, alpha, KD_epochs, device,train_loader, val_loader, PWInstance,logger):
    logger.info("\n\nStarting Knowledge Distillation\n")
    teacher_model.train()
    student_model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9)
    plotdk = PTPlotDK(PWInstance)
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
        print(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy * 100:.2f} %')
        logger.info(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy * 100:.2f} %')
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