import torch
import yaml
from utils.general import TQDM_BAR_FORMAT
from utils.loss import ComputeLoss
from tqdm import tqdm


def train_yolov5(model, nc, device):
    hyp_path = 'yolov5/data/hyps/hyp.scratch-low.yaml'
    with open(hyp_path) as f:
        yolo_hyp = yaml.load(f, Loader=yaml.SafeLoader)
    
    yolo_hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = yolo_hyp  # attach hyperparameters to model
    yolo_optimizer = torch.optim.SGD(model.parameters(), lr=yolo_hyp['lr0'], momentum=yolo_hyp['momentum'],
                                     weight_decay=yolo_hyp['weight_decay'])
    yolo_compute_loss = ComputeLoss(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(yolo_optimizer, milestones=[round(yolo_hyp['lrf'] * 0.8),
                                                                                 round(yolo_hyp['lrf'] * 0.9)],
                                                     gamma=0.1)


def yolov5_run_epoch(train_loader, model, yolo_optimizer, yolo_compute_loss,scheduler, epoch, pruning_epochs, device) :
    mloss = torch.zeros(3, device=device)  # mean losses
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
    print('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size')
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