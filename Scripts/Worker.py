import os
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import zipfile
from keras.models import load_model as tf_load
from Dataset_utils import *
import model_classes
import logging
from Pruning import *
from Quantization import *
from Knowledge_Distillation import *
from Model_utils import *
from logger_utils  import *

class Worker(QObject):
    finished = pyqtSignal()

    
    
    @pyqtSlot(dict)
    def processing(self, Dict):
    
        Dict['PWInstance'].update_progress_signal.connect(Dict['PWInstance'].update_progress)
        Dict['PWInstance'].undo_progress_signal.connect(Dict['PWInstance'].undo_progress)
        
        # Create a log directory if it doesn't exist
        log_dir = 'results/' + Dict['save_name']
        os.makedirs(log_dir, exist_ok=True)

        # Create a logger
        logger = setup_logger(log_dir)
        for key, value in Dict.items():
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float) or isinstance(value, bool):
                logger.info(f'{key} : {value}')


        if Dict['framework'] == 'tf':
            model = tf_load(Dict['model_path'])
            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model initialized')
            
            if Dict['Knowledge_Distillation']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Knowledge transfer...')
                teacher_model = tf_load(Dict['teacher_model_path'])
                model = distill_model_tensorflow(model, teacher_model, Dict['dataset_path'], Dict['batch_size'],
                                                 Dict['KD_temperature'], Dict['KD_alpha'], Dict['KD_epochs'],
                                                 Dict['PWInstance'],logger)
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Knowledge transfer completed')

            if Dict['Pruning']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Pruning...')
                # model = prune_model_tensorflow(model, dictio['pruning_ratio'], dictio['pruning_epochs'],
                #                        dictio['batch_size'],dictio['convert_tflite'],
                #                        dictio['Quantization'], dictio['PWInstance'],logger)

                model = basic_prune_model_tensorflow(model, Dict['pruning_ratio'],logger)

                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Pruning completed')

            if Dict['Quantization']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Quantization...')
                model = quantize_model_tensorflow(model, Dict['desired_format'],logger)
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Quantization completed')

            Dict['PWInstance'].final_size_signal.connect(Dict['PWInstance'].update_final_size)

            Dict['PWInstance'].update_progress_signal.emit('Saving final model...')
            os.chdir('./Models/Output Models')

            if Dict['convert_tflite'] or Dict['Quantization'].isChecked():
                open(log_dir + '.tflite', "wb").write(model)
                saved_model_path = log_dir + Dict['save_name'] + '.tflite'
    
                if Dict['Compressed']:
                    with zipfile.ZipFile(log_dir + Dict['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                        f.write(log_dir + '.tflite')
                    saved_model_path = Dict['save_name'] + '.zip'
                    if not Dict['save_unziped']:
                        os.remove(log_dir + '.tflite')

            elif Dict['Compressed']:
                model.save(log_dir + '.h5')
                with zipfile.ZipFile(log_dir + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                    f.write(log_dir + '.h5')
                saved_model_path = log_dir + '.zip'
    
                if not Dict['save_unziped']:
                    os.remove(log_dir + '.h5')

            else:
    
                model.save(log_dir + '.h5')
                saved_model_path = log_dir + '.h5'

            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model saved')

            Dict['PWInstance'].final_size_signal.emit(str(round(os.stat(saved_model_path).st_size / (1024 ^ 2), 3)))

            os.chdir('../../')
            self.finished.emit()
            return model
                
        elif Dict['framework'] == 'torch':
            try :
                train_loader, val_loader = load_pytorch_dataset(Dict['dataset_path'], batch_size=Dict['batch_size'])
                
            except:
                Dict['PWInstance'].update_progress_signal.emit('Dataset not found')
                return
            
            # Loading The model
            try :
                model = load_pytorch_model(Dict['model_path'],Dict['device'])
            except Exception as e:
                print("Error loading the PyTorch model :")
                print(str(e))
                
            #Testing intial model inference speed
            if Dict['dataset_path'].split('/')[-1] != '':
    
                # Testing final model inference speed and accuracy
                initial_inference_speed, initial_val_accuracy = test_inference_speed_and_accuracy(model, val_loader,device=Dict['device'],logger = logger)
                print(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
                print(f'Accuracy of the initial model : {100 * initial_val_accuracy:.2f} %')
                logger.info(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
                logger.info(f'Accuracy of the initial model : {100 * initial_val_accuracy:.2f} %')

            else:
                # Testing final model inference speed
                initial_inference_speed = test_inference_speed(model, device=Dict['device'],logger = logger)
                print(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
                logger.info(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
            
            
            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model initialized')
            

            if Dict['Pruning']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Pruning...')
                
                # model = prune_model_pytorch(model, Dict['pr'], Dict['epochs'], Dict['batch_size'],
                #                        Dict['q'].isChecked(), Dict['PWInstance'],logger = logger)
                # model = GAL_prune_pytorch(model, Dict['pr'], Dict['epochs'], Dict['batch_size'],Dict['device']
                #                             ,Dict['PWInstance'],logger = logger)
                model = prune_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'], Dict['device'],train_loader, val_loader,logger = logger)
                #model = basic_prune_finetune_model_pytorch(model, Dict['pr'], Dict['epochs'],Dict['device'],train_loader, val_loader,logger = logger)
                
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Pruning completed')

            if Dict['Quantization']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Quantization...')
                
                model = quantize_model_pytorch(model, Dict['desired_format'], Dict['device'],val_loader,logger = logger)
                
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Quantization completed')

            if Dict['Knowledge_Distillation']:
                Dict['PWInstance'].update_progress_signal.emit('Starting Knowledge transfer...')
                
                try :
                    teacher_model = load_pytorch_model(Dict['model_path'],Dict['device'])
                except Exception as e:
                    print("Error loading the PyTorch Teacher model :")
                    print(str(e))
                    
                model = distill_model_pytorch(model, teacher_model,
                                              Dict['KD_temperature'], Dict['KD_alpha'], Dict['KD_epochs'],
                                              Dict['device'],train_loader, val_loader,
                                              Dict['PWInstance'],logger = logger)
                
                Dict['PWInstance'].undo_progress_signal.emit()
                Dict['PWInstance'].update_progress_signal.emit('Knowledge transfer completed')

            Dict['PWInstance'].final_size_signal.connect(Dict['PWInstance'].update_final_size)

            Dict['PWInstance'].update_progress_signal.emit('Saving final model...')

            
            # _, test_loader = torch_load_mnist(batch_size=1)
            #if test_model == True:
            if Dict['dataset_path'].split('/')[-1] != '':
                
                # Testing final model inference speed and accuracy
                final_inference_speed, final_val_accuracy = test_inference_speed_and_accuracy(model, val_loader,
                                                                                          device=Dict['device'],logger = logger)
                print(f"Inference speed of the final model : {final_inference_speed:.4f} seconds")
                print(f'Accuracy of the final model : {100 * final_val_accuracy:.2f} %')
                logger.info(f'\nInference speed of the final model : {final_inference_speed:.4f} seconds')
                logger.info(f'Accuracy of the final model : {100 * final_val_accuracy:.2f} %')
            else:
                # Testing final model inference speed
                final_inference_speed= test_inference_speed(model,device=Dict['device'],logger = logger)
                print(f"Inference speed of the final model : {final_inference_speed:.4f} seconds")
                logger.info(f'\nInference speed of the final model : {final_inference_speed:.4f} seconds')

            saved_model_path = log_dir + '/' + Dict['save_name'] + '.pt'
            torch.save({'state_dict': model.state_dict()},  saved_model_path)
            

            if Dict['Compressed']:
                # Open a gzip file for writing
                with zipfile.ZipFile( log_dir + '/' + Dict['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                    # Save the model to the file
                    f.write(saved_model_path, arcname=Dict['save_name'] + '.pt')
                if not Dict['save_unziped']:
                    os.remove(saved_model_path)
                saved_model_path =  log_dir + '/' + Dict['save_name'] + '.zip'
            Dict['PWInstance'].undo_progress_signal.emit()
            Dict['PWInstance'].update_progress_signal.emit('Model saved')

            Dict['PWInstance'].final_size_signal.emit(str(round(os.stat(saved_model_path).st_size / (1024 ^ 2), 3)))

            self.finished.emit()
            return model

    