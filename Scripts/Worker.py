import re

from PyQt5.QtCore import QObject
import zipfile
from keras.models import load_model as tf_load
from Pruning import *
from Quantization import *
from Knowledge_Distillation import *
from Model_utils import *
from logger_utils import *
from PyQt5.QtCore import QRunnable, pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from yolov5.utils.torch_utils import de_parallel, ModelEMA


class WorkerSignals(QObject):
    finished = pyqtSignal(QMainWindow)
    error_occurred = pyqtSignal(str)


class Worker(QRunnable):
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, Dict, worker_id):
        super().__init__()
        self.worker_id = worker_id
        self.Dict = Dict
        self.signals = WorkerSignals()
    
    def run(self):
        Dict = self.Dict
        Dict['PWInstance'].update_progress_signal.connect(Dict['PWInstance'].update_progress)
        Dict['PWInstance'].undo_progress_signal.connect(Dict['PWInstance'].undo_progress)
        
        # Create a log directory if it doesn't exist
        log_dir = 'results/' + Dict['save_name']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=False)
        
        else:
            counter = 1
            while True:
                new_dir = f"{log_dir}_{counter}"
                
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir, exist_ok=False)
                    break
                
                counter += 1
            log_dir = new_dir
        
        # Create a logger
        logger = setup_logger(log_dir)
        for key, value in Dict.items():
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                logger.info(f'{key} : {value}')
        
        yolo = 'yolo' in Dict['model_path'].split(os.sep)[-1]
        dataset = Dict['dataset_path'].split(os.sep)[-1] != ''
        
        if Dict['framework'] == 'tf':
            model = tf_load(Dict['model_path'])
            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model initialized')
            if dataset:
                train_data, train_labels, val_data, val_labels = load_tensorflow_data(Dict['dataset_path'],Dict['train_fraction'],Dict['validation_fraction'],
                                                                                      batch_size=Dict['batch_size'])
                initial_accuracy, initial_avg_eval_time = evaluate_tensorflow_model(model, val_data, val_labels)
                print(f'Accuracy of initial model : {initial_accuracy}')
                print(f'Average inference time of initial model : {initial_avg_eval_time}')
                logger.info(f'Accuracy of initial model : {initial_accuracy}')
                logger.info(f'Average inference time of initial model : {initial_avg_eval_time}')
            else:
                train_data, train_labels, val_data, val_labels = None, None, None, None
            
            if Dict['Pruning']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Pruning...')
                model = prune_model_tensorflow(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                               Dict['batch_size'], train_data,
                                               train_labels, val_data, val_labels)
                
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Pruning completed')
            
            if Dict['Knowledge_Distillation']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Knowledge transfer...')
                teacher_model = tf_load(Dict['teacher_model_path'])
                model = distill_model_tensorflow(model, teacher_model, train_data, train_labels, val_data, val_labels,
                                                 Dict['KD_temperature'], Dict['KD_alpha'], Dict['KD_epochs'],
                                                 Dict['PWInstance'])
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Knowledge transfer completed')
            
            if Dict['Quantization']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Quantization...')
                model = quantize_model_tensorflow(model, Dict['desired_format'], logger)
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Quantization completed')
            
            
            if not Dict['Quantization'] and not Dict['convert_tflite']:
                Dict['PWInstance'].final_size_signal.connect(Dict['PWInstance'].update_final_size)
                
                final_accuracy, final_avg_eval_time = evaluate_tensorflow_model(model, val_data, val_labels)
                print(f'Accuracy of final model : {final_accuracy}')
                print(f'Average inference time of final model : {final_avg_eval_time}')
                logger.info(f'Accuracy of final model : {final_accuracy}')
                logger.info(f'Average inference time of final model : {final_avg_eval_time}')
            
            Dict['PWInstance'].update_progress_signal.emit('Saving final model...')
            
            if Dict['convert_tflite'] or Dict['Quantization']:
                open(log_dir + Dict['save_name'] + '.tflite', "wb").write(model)
                saved_model_path = log_dir + Dict['save_name'] + '.tflite'
                
                if Dict['Compressed']:
                    with zipfile.ZipFile(log_dir + Dict['save_name'] + '.zip', 'w',
                                         compression=zipfile.ZIP_DEFLATED) as f:
                        f.write(log_dir + Dict['save_name']+ '.tflite')
                    saved_model_path = Dict['save_name'] + '.zip'
                    if not Dict['save_unziped']:
                        os.remove(log_dir + Dict['save_name']+ '.tflite')
            
            elif Dict['Compressed']:
                model.save(log_dir + Dict['save_name'] + '.h5')
                with zipfile.ZipFile(log_dir + Dict['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                    f.write(log_dir + Dict['save_name'] + '.h5')
                saved_model_path = log_dir + Dict['save_name'] + '.zip'
                
                if not Dict['save_unziped']:
                    os.remove(log_dir + Dict['save_name'] + '.h5')
            
            else:
                
                model.save(log_dir + Dict['save_name'] + '.h5')
                saved_model_path = log_dir + Dict['save_name'] + '.h5'
            
            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model saved')
            
            Dict['PWInstance'].final_size_signal.emit(round(os.stat(saved_model_path).st_size / (1024 ^ 3), 3))
            self.signals.finished.emit(Dict['PWInstance'])
        
        elif Dict['framework'] == 'torch' and yolo:
            
            yaml_file_path = os.path.join(Dict['dataset_path'], 'data.yaml')
            with open(yaml_file_path, 'r') as file:
                content = file.read()
            
            current_dir = os.path.dirname(__file__).split(os.sep)[:-1]
            current_dir = os.sep.join(current_dir)
            print('current_dir : ' + current_dir)
            
            absolute_path = os.path.join(current_dir, yaml_file_path)
            print('absolute_path : ' + absolute_path)
            yaml_dir = os.path.dirname(absolute_path)
            print('yaml_dir : ' + yaml_dir)
            
            if os.path.isdir(os.path.join(yaml_dir, 'images')):
                new_train_path = os.path.join(yaml_dir, 'images', 'train')
                new_val_path = os.path.join(yaml_dir, 'images', 'valid')
                new_test_path = os.path.join(yaml_dir, 'images', 'test')
            else:
                new_train_path = os.path.join(yaml_dir, 'train', 'images')
                new_val_path = os.path.join(yaml_dir, 'valid', 'images')
                new_test_path = os.path.join(yaml_dir, 'test', 'images')
            
            print('new_train_path : ' + new_train_path)
            print('new_val_path : ' + new_val_path)
            print('new_test_path : ' + new_test_path)
            
            logger.info('new_train_path : ' + new_train_path)
            logger.info('new_val_path : ' + new_val_path)
            logger.info('new_test_path : ' + new_test_path)
            
            content = re.sub(r'train: .+', fr'train: {re.escape(new_train_path)}', content)
            content = re.sub(r'val: .+', fr'val: {re.escape(new_val_path)}', content)
            content = re.sub(r'test: .+', fr'test: {re.escape(new_test_path)}', content)
            
            print('saving the modified content back to the .yaml file')
            with open(yaml_file_path, 'w') as file:
                file.write(content)
            
            match = re.search(r"nc\s*:\s*(\d+)", content, re.IGNORECASE)
            if match:
                nc_value = int(match.group(1))
                print('Number of classes = ' + str(nc_value))
                logger.info('Number of classes = ' + str(nc_value))
            
            else:
                print("No 'nc' value found in the YAML file.Using Default value of 80")
                logger.info("No 'nc' value found in the YAML file.Using Default value of 80")
                nc_value = 80
            
            # Loading the dataset has been moved inside pruning because it's the only place where it's used
            #train_loader, val_loader = load_pytorch_dataset(Dict['dataset_path'], batch_size=Dict['batch_size'],
            #                                                train_fraction=Dict['train_fraction'],
            #                                                val_fraction=Dict['validation_fraction'])
            
            # Loading The model
            try:
                model_dict = torch.load(Dict['model_path'], map_location='cpu')
                model_name = Dict['model_path'].split(os.sep)[-1]
                model = load_pytorch_model(model_dict, Dict['device'], model_name, number_of_classes=nc_value)
            except Exception as e:
                print(f"\n\nError loading the PyTorch model at {Dict['model_path']} :")
                print(str(e) + '\n')
                self.signals.error_occurred.emit(str(e))
            if dataset:
                initial_inference_speed, initial_val_accuracy = test_inference_speed_and_accuracy(Dict['model_path'],
                                                                                                  None,
                                                                                                  device=Dict['device'],
                                                                                                  Dict=Dict,
                                                                                                  logger=logger,
                                                                                                  half=Dict['half'],
                                                                                                  is_yolo=yolo)
                logger.info(
                    f"Speed of the initial model : {initial_inference_speed[0]}ms pre-process, {initial_inference_speed[1]}ms inference, {initial_inference_speed[2]}ms NMS per image ")
                logger.info(
                    f'Accuracy of the initial model | Precision : {100 * initial_val_accuracy[0]:.2f} % | Recall : {100 * initial_val_accuracy[1]:.2f} % | mAP50 : {100 * initial_val_accuracy[2]:.2f} % | mAP50-95 : {100 * initial_val_accuracy[3]:.2f} %')
            
            # Testing initial model size
            model.eval()
            initial_object_size_mb, initial_gpu_model_memory_mb, initial_disk_size_mb = test_model_size(model, Dict[
                'model_path'], Dict['device'])
            print(f"Initial model GPU memory allocated: {initial_gpu_model_memory_mb:.3f} MB")
            print(f"Initial model object size: {initial_object_size_mb:.3f} MB")
            print(f"Initial model disk size: {initial_disk_size_mb:.3f} MB")
            logger.info(f"Initial model GPU memory allocated: {initial_gpu_model_memory_mb:.3f} MB")
            logger.info(f"Initial model object size: {initial_object_size_mb:.3f} MB")
            logger.info(f"Initial model disk size: {initial_disk_size_mb:.3f} MB")

            # ---------------------------------Beginning of optimization---------------------------------
            
            if Dict['Pruning']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Pruning...')
                train_loader, val_loader = load_pytorch_dataset(Dict['dataset_path'], batch_size=Dict['batch_size'],
                                                                train_fraction=Dict['train_fraction'],
                                                                val_fraction=Dict['validation_fraction'])
                
                if Dict['pruning_type'] == 'random_unstructured_pruning':
                    model = pytorch_random_unstructured_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                    Dict['device'], train_loader, val_loader, logger, yolo, Dict['PWInstance'], nc_value)
                
                if Dict['pruning_type'] == 'random_structured_pruning':
                    model = pytorch_random_structured_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                    Dict['device'], train_loader, val_loader, logger, yolo, Dict['PWInstance'], nc_value)
                
                if Dict['pruning_type'] == 'magnitude_pruning':
                    model = pytorch_magnitude_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                    Dict['device'], train_loader, val_loader, logger, yolo, Dict['PWInstance'], nc_value)
                elif Dict['pruning_type'] == 'dynamic_pruning':
                    model = prune_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                    Dict['device'], train_loader, val_loader, logger, yolo, Dict['PWInstance'], nc_value)
                
                elif Dict['pruning_type'] == 'global_pruning':
                    model = prune_global_model_pytorch(model, Dict['pruning_ratio'], logger)
                
                elif Dict['pruning_type'] == 'global_dynamic_pruning':
                    model = prune_global_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                               Dict['device'], train_loader, val_loader, logger, yolo, Dict['PWInstance'],nc_value)
                
                else:
                    print('ERROR : No pruning method selected')
                
                sparsity_list = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        sparsity_list.append(
                            100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()))
                logger.info("Average Sparsity in layers : {:.2f}%".format(sum(sparsity_list) / len(sparsity_list)))

                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Pruning completed')
            
            if Dict['Knowledge_Distillation']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Knowledge transfer...')
                
                model = distill_model_yolo(model, Dict['teacher_model_path'], Dict['dataset_path'],
                                           Dict['KD_epochs'], logger=logger,nbr_classes=nc_value,save_name=Dict['save_name'],batch_size=Dict['batch_size'],device=Dict['device'])
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Knowledge transfer completed')
            
            if Dict['Quantization']:
                Dict['PWInstance'].update_progress_signal.emit('Quantization not yet supported for yolo models')
                
            # ---------------------------------End of optimization---------------------------------
            Dict['PWInstance'].final_size_signal.connect(Dict['PWInstance'].update_final_size)
            Dict['PWInstance'].update_progress_signal.emit('Saving final model...')
            if dataset:
                yolo_optimizer = torch.optim.SGD(model.parameters(), 0.01,
                                                 momentum=0.937,
                                                 weight_decay=0.0005)
                ema = ModelEMA(model)
                
                saved_model_path = log_dir + os.sep + Dict['save_name'] + '.pt'
                ckpt = {
                    'epoch': 0,
                    'best_fitness': 0,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': yolo_optimizer.state_dict(),
                    'opt': None,
                    'git': None,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}
                
                # Save last, best and delete
                torch.save(ckpt, saved_model_path)
                del ckpt
                final_inference_speed, final_val_accuracy = test_inference_speed_and_accuracy(saved_model_path,
                                                                                              None,
                                                                                              device=Dict['device'],
                                                                                              Dict=Dict,
                                                                                              logger=logger,
                                                                                              half=Dict['half'],
                                                                                              is_yolo=yolo)
                logger.info(
                    f"Speed of the final model : {final_inference_speed[0]}ms pre-process, {final_inference_speed[1]}ms inference, {final_inference_speed[2]}ms NMS per image ")
                logger.info(
                    f'Accuracy of the final model | Precision : {100 * final_val_accuracy[0]:.2f} % | Recall : {100 * final_val_accuracy[1]:.2f} % | mAP50 : {100 * final_val_accuracy[2]:.2f} % | mAP50-95 : {100 * final_val_accuracy[3]:.2f} %')
            
            if Dict['Compressed']:
                # Open a gzip file for writing
                with zipfile.ZipFile(log_dir + os.sep + Dict['save_name'] + '.zip', 'w',
                                     compression=zipfile.ZIP_DEFLATED) as f:
                    # Save the model to the file
                    f.write(saved_model_path, arcname=Dict['save_name'] + '.pt')
                if not Dict['save_unziped']:
                    os.remove(saved_model_path)
                saved_model_path = log_dir + os.sep + Dict['save_name'] + '.zip'

            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model saved')
            
            # Testing final model size
            model.eval()
            object_size_mb, gpu_size_mb, disk_size_mb = test_model_size(model, saved_model_path, Dict['device'])
            
            print(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
            print(f"Final model object size: {object_size_mb:.3f} MB")
            print(f"Final model disk size: {disk_size_mb:.3f} MB")
            logger.info(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
            logger.info(f"Final model object size: {object_size_mb:.3f} MB")
            logger.info(f"Final model disk size: {disk_size_mb:.3f} MB")

            Dict['PWInstance'].final_size_signal.emit(disk_size_mb)
            self.signals.finished.emit(Dict['PWInstance'])
        
        elif Dict['framework'] == 'torch':
            torch.manual_seed(0)
            try:
                train_loader, val_loader = load_pytorch_dataset(Dict['dataset_path'], batch_size=Dict['batch_size'],
                                                                train_fraction=Dict['train_fraction'],
                                                                val_fraction=Dict['validation_fraction']
                                                                )
            except Exception as e:
                if Dict['dataset_path'].split(os.sep)[-1] == '':
                    print('No Dataset selected, Continuing with the optimization process without testing the model')
                    print('WARNING : Some optimization processes might not work correctly without a dataset')
                else:
                    Dict['PWInstance'].update_progress_signal.emit('Dataset not found')
                    print("\n\nError loading the dataset :")
                    print(str(e) + '\n')
                    self.signals.error_occurred.emit(str(e))
            
            # Loading The model
            try:
                model_dict = torch.load(Dict['model_path'], map_location='cpu')
                model_name = Dict['model_path'].split(os.sep)[-1]
                model = load_pytorch_model(model_dict, Dict['device'], model_name)
            
            except Exception as e:
                print(f"\n\nError loading the PyTorch model at {Dict['model_path']} :")
                print(str(e) + '\n')
                self.signals.error_occurred.emit(str(e))
            
            # Testing initial model inference speed
            if dataset:
                
                initial_inference_speed, initial_val_accuracy = test_inference_speed_and_accuracy(model, val_loader,
                                                                                                  device=Dict[
                                                                                                      'device'],
                                                                                                  Dict=Dict,
                                                                                                  logger=logger,
                                                                                                  half=Dict['half'],
                                                                                                  is_yolo=yolo)
                print(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
                print(f'Accuracy of the initial model : {100 * initial_val_accuracy:.2f} %')
                logger.info(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
                logger.info(f'Accuracy of the initial model : {100 * initial_val_accuracy:.2f} %')
            
            else:
                # Testing final model inference speed
                initial_inference_speed = test_inference_speed(model, device=Dict['device'], logger=logger)
                print(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
                logger.info(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
            
            # Testing initial model size
            model.eval()
            initial_object_size_mb, initial_gpu_model_memory_mb, initial_disk_size_mb = test_model_size(model, Dict[
                'model_path'], Dict['device'])
            print(f"Initial model GPU memory allocated: {initial_gpu_model_memory_mb:.3f} MB")
            print(f"Initial model object size: {initial_object_size_mb:.3f} MB")
            print(f"Initial model disk size: {initial_disk_size_mb:.3f} MB")
            logger.info(f"Initial model GPU memory allocated: {initial_gpu_model_memory_mb:.3f} MB")
            logger.info(f"Initial model object size: {initial_object_size_mb:.3f} MB")
            logger.info(f"Initial model disk size: {initial_disk_size_mb:.3f} MB")
            
            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model initialized')
            
            # ------------------------------BEGINNING OF OPTIMIZATION PROCESS----------------------------
            
            if Dict['Pruning']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Pruning...')

                if Dict['pruning_type'] == 'random_unstructured_pruning':
                    model = pytorch_random_unstructured_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                              Dict['device'], train_loader, val_loader, logger, yolo,
                                                              Dict['PWInstance'])

                if Dict['pruning_type'] == 'random_structured_pruning':
                    model = pytorch_random_structured_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                            Dict['device'], train_loader, val_loader, logger, yolo,
                                                            Dict['PWInstance'])

                if Dict['pruning_type'] == 'magnitude_pruning':
                    model = pytorch_magnitude_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                    Dict['device'], train_loader, val_loader, logger, yolo,
                                                    Dict['PWInstance'])
                elif Dict['pruning_type'] == 'dynamic_pruning':
                    model = prune_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                        Dict['device'], train_loader, val_loader, logger, yolo,
                                                        Dict['PWInstance'])

                elif Dict['pruning_type'] == 'global_pruning':
                    model = prune_global_model_pytorch(model, Dict['pruning_ratio'], logger)

                elif Dict['pruning_type'] == 'global_dynamic_pruning':
                    model = prune_global_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                               Dict['device'], train_loader, val_loader, logger, yolo,
                                                               Dict['PWInstance'])

                else:
                    print('ERROR : No pruning method selected')
                
                sparsity_list = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                        sparsity_list.append(
                            100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()))
                logger.info("Average Sparsity in layers : {:.2f}%".format(sum(sparsity_list) / len(sparsity_list)))
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Pruning completed')
            
            if Dict['Knowledge_Distillation']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Knowledge transfer...')
                
                try:
                    teacher_model_dict = torch.load(Dict['teacher_model_path'], map_location='cpu')
                    teacher_model_name = Dict['teacher_model_path'].split(os.sep)[-1]
                    teacher_model = load_pytorch_model(teacher_model_dict, Dict['device'], teacher_model_name)
                
                except Exception as e:
                    print(f"\n\nError loading the PyTorch Teacher model in {Dict['teacher_model_path']} :")
                    print(str(e) + "\n")
                    self.signals.error_occurred.emit(str(e))
                
                model = distill_model_pytorch(model, teacher_model,
                                              Dict['KD_temperature'], Dict['KD_alpha'], Dict['KD_epochs'],
                                              Dict['device'], train_loader, val_loader,
                                              Dict['PWInstance'], logger=logger)
                
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Knowledge transfer completed')
            
            if Dict['Quantization']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Quantization...')
                Dict['device'] = 'cpu'
                if Dict['quantization_type'] == 'Static':
                    model = static_quantize_model_pytorch(model, Dict['desired_format'], Dict['device'], train_loader,
                                                          Dict['quantization_epochs'],
                                                          logger=logger)
                
                elif Dict['quantization_type'] == 'Dynamic':
                    model = dynamic_quantize_model_pytorch(model, Dict['desired_format'], Dict['device'], logger=logger)
                elif Dict['quantization_type'] == 'Quantization Aware Training':
                    model = quantization_aware_training_pytorch(model, train_loader, val_loader, Dict['device'],
                                                                Dict['quantization_epochs'], logger=logger)
                
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Quantization completed')
            
            # ---------------------------------------END OF OPTIMIZATION--------------------------------------------
            
            Dict['PWInstance'].final_size_signal.connect(Dict['PWInstance'].update_final_size)
            Dict['PWInstance'].update_progress_signal.emit('Saving final model...')
            
            if dataset:
                # Testing final model inference speed and accuracy
                final_inference_speed, final_val_accuracy = test_inference_speed_and_accuracy(model, val_loader,
                                                                                              device=Dict['device'],
                                                                                              Dict=Dict,
                                                                                              logger=logger,
                                                                                              half=Dict['half'],
                                                                                              is_yolo=yolo)
                print(f"Inference speed of the final model : {final_inference_speed:.4f} seconds")
                print(f'Accuracy of the final model : {100 * final_val_accuracy:.2f} %')
                logger.info(f'\nInference speed of the final model : {final_inference_speed:.4f} seconds')
                logger.info(f'Accuracy of the final model : {100 * final_val_accuracy:.2f} %')
                saved_model_path = log_dir + os.sep + Dict['save_name'] + '.pt'
                torch.save({'state_dict': model.state_dict()}, saved_model_path)
            else:
                # Testing final model inference speed
                final_inference_speed = test_inference_speed(model, device=Dict['device'], logger=logger)
                print(f"Inference speed of the final model : {final_inference_speed:.4f} seconds")
                logger.info(f'\nInference speed of the final model : {final_inference_speed:.4f} seconds')
                saved_model_path = log_dir + os.sep + Dict['save_name'] + '.pt'
                torch.save({'state_dict': model.state_dict()}, saved_model_path)
            
            if Dict['Compressed']:
                # Open a gzip file for writing
                with zipfile.ZipFile(log_dir + '/' + Dict['save_name'] + '.zip', 'w',
                                     compression=zipfile.ZIP_DEFLATED) as f:
                    # Save the model to the file
                    f.write(saved_model_path, arcname=Dict['save_name'] + '.pt')
                if not Dict['save_unziped']:
                    os.remove(saved_model_path)
                saved_model_path = log_dir + '/' + Dict['save_name'] + '.zip'
            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model saved')
            
            # Testing final model size
            model.eval()
            object_size_mb, gpu_size_mb, disk_size_mb = test_model_size(model, saved_model_path, Dict['device'])
            
            print(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
            print(f"Final model object size: {object_size_mb:.3f} MB")
            print(f"Final model disk size: {disk_size_mb:.3f} MB")
            logger.info(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
            logger.info(f"Final model object size: {object_size_mb:.3f} MB")
            logger.info(f"Final model disk size: {disk_size_mb:.3f} MB")
            
            Dict['PWInstance'].final_size_signal.emit(disk_size_mb)
            self.signals.finished.emit(Dict['PWInstance'])
            
        
        