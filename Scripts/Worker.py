import os
import sys
import re
from copy import deepcopy
from datetime import datetime
import zipfile
from keras.models import load_model as tf_load
from Pruning import *
from Quantization import *
from Knowledge_Distillation import *
from Model_utils import *
from logger_utils  import *


from utils.torch_utils import de_parallel, ModelEMA


def worker(Dict):
    
    # Create a log directory if it doesn't exist
    log_dir = 'results/' + Dict['save_name']

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=False)

    else :
        counter = 1
        while True:
            new_dir = f"{log_dir}_{counter}"
        
            if not os.path.exists(new_dir):
                os.makedirs(new_dir,exist_ok=False)
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
        
        if dataset :
            train_data, train_labels, val_data, val_labels = load_tensorflow_data( Dict['dataset_path'],img_size=(32,32),batch_size=Dict['batch_size'])
            initial_accuracy, initial_avg_eval_time = evaluate_tensorflow_model(model, val_data, val_labels)
            print(f'Accuracy of initial model : {initial_accuracy}')
            print(f'Average inference time of initial model : {initial_avg_eval_time}')
            logger.info(f'Accuracy of initial model : {initial_accuracy}')
            logger.info(f'Average inference time of initial model : {initial_avg_eval_time}')
        else :
            train_data, train_labels, val_data, val_labels = None,None,None,None
        
        if Dict['Pruning']:
            logger.info('Starting pruning')
            model = prune_model_tensorflow(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                   Dict['batch_size'],Dict['convert_tflite'],
                                   Dict['Quantization'],logger)

            # model = basic_prune_model_tensorflow(model, Dict['pruning_ratio'],logger)
        
        if Dict['Knowledge_Distillation']:
            logger.info('Starting quantization')
            teacher_model = tf_load(Dict['teacher_model_path'])
            model = distill_model_tensorflow(model, teacher_model, Dict['dataset_path'], Dict['batch_size'],
                                             Dict['KD_temperature'], Dict['KD_alpha'], Dict['KD_epochs'],
                                             logger)

        

        if Dict['Quantization']:
            logger.info('Starting knowledge distillation')
            model = quantize_model_tensorflow(model, Dict['desired_format'],logger)

        final_accuracy, final_avg_eval_time= evaluate_tensorflow_model(model,val_data, val_labels)
        print(f'Accuracy of final model : {final_accuracy}')
        print(f'Average inference time of final model : {final_avg_eval_time}')
        logger.info(f'Accuracy of final model : {final_accuracy}')
        logger.info(f'Average inference time of final model : {final_avg_eval_time}')
        
        
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

        os.chdir('../../')
        
    elif Dict['framework'] == 'torch' and yolo :
        # Load the content of the .yaml file
        yaml_file_path = os.path.join(Dict['dataset_path'], 'data.yaml')
        with open(yaml_file_path, 'r') as file:
            content = file.read()
    
        # Get the directory path of the .yaml file
        current_dir = os.path.dirname(__file__).split(os.sep)[:-1]  # Get the current directory of the script
        current_dir = os.sep.join(current_dir)
        print('current_dir : ' + current_dir)
    
        absolute_path = os.path.join(current_dir, yaml_file_path)  # Get the absolute path
        print('absolute_path : ' + absolute_path)
        yaml_dir = os.path.dirname(absolute_path)
        print('yaml_dir : ' + yaml_dir)
    
        # Define the new paths relative to the .yaml file directory
        new_train_path = os.path.join(yaml_dir,'images', 'train' )
        new_val_path = os.path.join(yaml_dir,'images', 'valid')
        new_test_path = os.path.join(yaml_dir, 'images','test')
    
        print('new_train_path : ' + new_train_path)
        print('new_val_path : ' + new_val_path)
        print('new_test_path : ' + new_test_path)
        # Modify the paths relative to the .yaml file directory
        content = re.sub(r'train: .+', fr'train: {re.escape(new_train_path)}', content)
        content = re.sub(r'val: .+', fr'val: {re.escape(new_val_path)}', content)
        content = re.sub(r'test: .+', fr'test: {re.escape(new_test_path)}', content)
    
        # Save the modified content back to the .yaml file
        print('saving the modified content back to the .yaml file')
        with open(yaml_file_path, 'w') as file:
            file.write(content)

        train_loader, val_loader = load_pytorch_dataset(Dict['dataset_path'], batch_size=Dict['batch_size'],
                                                        train_fraction=Dict['train_fraction'],
                                                        val_fraction=Dict['validation_fraction'],
                                                        half=Dict['half'],
                                                        logger=logger)

        # Loading The model
        model = load_pytorch_model(Dict['model_path'], Dict['device'], Dict['half'],
                                   yaml_path=Dict['dataset_path'] + '/data.yaml')
        
        if dataset :
            initial_inference_speed, initial_val_accuracy = test_inference_speed_and_accuracy(Dict['model_path'],
                                                                                              val_loader,
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

        if Dict['Pruning']:
            print('Starting Pruning ...')
    
            if Dict['pruning_type'] == 'random_unstructured_pruning':
                model = pytorch_random_unstructured_prune(model, Dict['pruning_ratio'])
    
            if Dict['pruning_type'] == 'random_structured_pruning':
                model = pytorch_random_structured_prune(model, Dict['pruning_ratio'])
    
            if Dict['pruning_type'] == 'magnitude_pruning':
                model = pytorch_magnitude_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                Dict['device'], train_loader, val_loader, logger, yolo)
            elif Dict['pruning_type'] == 'dynamic_pruning':
                model = prune_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                    Dict['device'], train_loader, val_loader, logger=logger,
                                                    is_yolo=yolo)
    
            elif Dict['pruning_type'] == 'global_pruning':
                model = prune_global_model_pytorch(model, Dict['pruning_ratio'], logger)
    
            elif Dict['pruning_type'] == 'global_dynamic_pruning':
                model = prune_global_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                           Dict['device'], train_loader, val_loader, logger, yolo)
    
            else:
                print('ERROR : No pruning method selected')
    
            sparsity_list = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    sparsity_list.append(100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()))
            print("Average Sparsity in layers : {:.2f}%".format(sum(sparsity_list) / len(sparsity_list)))
    
            print('Pruning Done')

        if Dict['Knowledge_Distillation']:
            print('Starting Knowledge Distillation ...')
    
            model = distill_model_yolo(model, Dict['teacher_model_path'],Dict['dataset_path'],
                                           Dict['KD_epochs'],logger=logger)
            print('Knowledge Distillation Done')
            
        
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
            final_inference_speed, final_val_accuracy = test_inference_speed_and_accuracy(saved_model_path, val_loader,
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
            with zipfile.ZipFile( log_dir + os.sep + Dict['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                # Save the model to the file
                f.write(saved_model_path, arcname=Dict['save_name'] + '.pt')
            if not Dict['save_unziped']:
                os.remove(saved_model_path)
            saved_model_path =  log_dir + os.sep + Dict['save_name'] + '.zip'

        # Testing final model size
        model.eval()
        object_size_mb, gpu_size_mb, disk_size_mb = test_model_size(model, saved_model_path, Dict['device'])

        print(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
        print(f"Final model object size: {object_size_mb:.3f} MB")
        print(f"Final model disk size: {disk_size_mb:.3f} MB")
        logger.info(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
        logger.info(f"Final model object size: {object_size_mb:.3f} MB")
        logger.info(f"Final model disk size: {disk_size_mb:.3f} MB")

    elif Dict['framework'] == 'torch':
        torch.manual_seed(0)
            
        
        train_loader, val_loader = load_pytorch_dataset(Dict['dataset_path'], batch_size=Dict['batch_size'],
                                                            train_fraction=Dict['train_fraction'],
                                                            val_fraction=Dict['validation_fraction'],
                                                            half=Dict['half'],
                                                            logger=logger)
        
        
        # Loading The model
        model = load_pytorch_model(Dict['model_path'],Dict['device'],Dict['half'],yaml_path=Dict['dataset_path'] + '/data.yaml')
            
        #Testing intial model inference speed
        if dataset:
            initial_inference_speed, initial_val_accuracy = test_inference_speed_and_accuracy(model, val_loader,
                                                                                              device=Dict[
                                                                                                  'device'],
                                                                                              Dict=Dict,
                                                                                              logger=logger,
                                                                                              half = Dict['half'],
                                                                                              is_yolo=yolo)
            print(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
            print(f'Accuracy of the initial model : {100 * initial_val_accuracy:.2f} %')
            logger.info(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
            logger.info(f'Accuracy of the initial model : {100 * initial_val_accuracy:.2f} %')

        else:
            # Testing final model inference speed
            initial_inference_speed = test_inference_speed(model, device=Dict['device'],logger = logger)
            print(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")
            logger.info(f"Inference speed of the initial model : {initial_inference_speed:.4f} seconds")

        # Testing initial model size
        model.eval()
        initial_object_size_mb, initial_gpu_model_memory_mb, initial_disk_size_mb = test_model_size(model, Dict['model_path'], Dict['device'])
        print(f"Initial model GPU memory allocated: {initial_gpu_model_memory_mb:.3f} MB")
        print(f"Initial model object size: {initial_object_size_mb:.3f} MB")
        print(f"Initial model disk size: {initial_disk_size_mb:.3f} MB")
        logger.info(f"Initial model GPU memory allocated: {initial_gpu_model_memory_mb:.3f} MB")
        logger.info(f"Initial model object size: {initial_object_size_mb:.3f} MB")
        logger.info(f"Initial model disk size: {initial_disk_size_mb:.3f} MB")
        
        
        #BEGINNING OF OPTIMIZATION PROCESS--------------------------------------------------------------------------

        if Dict['Pruning']:
            print('Starting Pruning ...')
            
            if Dict['pruning_type']=='random_unstructured_pruning' :
                model = pytorch_random_unstructured_prune(model, Dict['pruning_ratio'])
            
            if Dict['pruning_type']=='random_structured_pruning' :
                model = pytorch_random_structured_prune(model, Dict['pruning_ratio'])
            
            if Dict['pruning_type']=='magnitude_pruning' :
                model = pytorch_magnitude_prune(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                           Dict['device'],train_loader,val_loader,logger,yolo)
            elif Dict['pruning_type']=='dynamic_pruning' :
                model = prune_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                    Dict['device'], train_loader, val_loader, logger=logger,
                                                    is_yolo=yolo)

            elif Dict['pruning_type']=='global_pruning' :
                model = prune_global_model_pytorch(model, Dict['pruning_ratio'], logger)
                
            elif Dict['pruning_type']=='global_dynamic_pruning' :
                model = prune_global_dynamic_model_pytorch(model, Dict['pruning_ratio'], Dict['pruning_epochs'],
                                                                Dict['device'], train_loader, val_loader, logger,yolo)
 
            else :
                print('ERROR : No pruning method selected')

            sparsity_list = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                    sparsity_list.append(100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()))
            print("Average Sparsity in layers : {:.2f}%".format(sum(sparsity_list)/len(sparsity_list)))
            
            print('Pruning Done')

        

        if Dict['Knowledge_Distillation']:
            print('Starting Knowledge Distillation ...')
            teacher_model = load_pytorch_model(Dict['teacher_model_path'],Dict['device'],Dict['half'],yaml_path=Dict['dataset_path'] + '/data.yaml')
                
            model = distill_model_pytorch(model, teacher_model,
                                          Dict['KD_temperature'], Dict['KD_alpha'], Dict['KD_epochs'],
                                          Dict['device'],train_loader, val_loader,is_yolo=yolo,
                                          logger = logger)
            print('Knowledge Distillation Done')

        if Dict['Quantization']:
            print('Starting Quantization ...')
            Dict['device'] = 'cpu'
            model = quantize_model_pytorch(model, Dict['desired_format'], Dict['device'], train_loader, logger=logger)

            # model = quantization_aware_training_pytorch(model,train_loader,val_loader,Dict['device'],1,logger = logger)
            print('Quantization Done')
        #END OF OPTIMIZATION----------------------------------------------------------------------------------------
        

        print('Evaluating and Saving the final model ...')
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
            final_inference_speed= test_inference_speed(model,device=Dict['device'],logger = logger)
            print(f"Inference speed of the final model : {final_inference_speed:.4f} seconds")
            logger.info(f'\nInference speed of the final model : {final_inference_speed:.4f} seconds')
            saved_model_path = log_dir + os.sep + Dict['save_name'] + '.pt'
            torch.save({'state_dict': model.state_dict()}, saved_model_path)
        
        

        if Dict['Compressed']:
            # Open a gzip file for writing
            with zipfile.ZipFile( log_dir + os.sep + Dict['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                # Save the model to the file
                f.write(saved_model_path, arcname=Dict['save_name'] + '.pt')
            if not Dict['save_unziped']:
                os.remove(saved_model_path)
            saved_model_path =  log_dir + os.sep + Dict['save_name'] + '.zip'
     

        # Testing final model size
        model.eval()
        object_size_mb, gpu_size_mb,disk_size_mb = test_model_size(model,saved_model_path, Dict['device'])
        
        print(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
        print(f"Final model object size: {object_size_mb:.3f} MB")
        print(f"Final model disk size: {disk_size_mb:.3f} MB")
        logger.info(f"Final model GPU memory allocated: {gpu_size_mb:.3f} MB")
        logger.info(f"Final model object size: {object_size_mb:.3f} MB")
        logger.info(f"Final model disk size: {disk_size_mb:.3f} MB")

            
        
        