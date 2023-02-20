import os
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import zipfile
from keras.models import load_model as tf_load
from Optimizer import *
import torch


class Worker(QObject):
    finished = pyqtSignal()
    
    @pyqtSlot(dict)
    def processing(self, dictio):
    
        dictio['PWInstance'].update_progress_signal.connect(dictio['PWInstance'].update_progress)
        dictio['PWInstance'].undo_progress_signal.connect(dictio['PWInstance'].undo_progress)

        if dictio['framework'] == 'tf':
            model = tf_load(dictio['model_path'])
            dictio['PWInstance'].undo_progress_signal.emit()
            dictio['PWInstance'].update_progress_signal.emit('Model initialized')
            
            if dictio['d'].isChecked():
                dictio['PWInstance'].update_progress_signal.emit('Starting Knowledge transfer...')
                teacher_model = tf_load(dictio['teacher_model_path'])
                model = distill_model_tensorflow(model, teacher_model,dictio['dataset_path'], dictio['batch_size'],
                                         dictio['temperature'], dictio['Alpha'], dictio['DKepochs'],
                                         dictio['PWInstance'])
                dictio['PWInstance'].undo_progress_signal.emit()
                dictio['PWInstance'].update_progress_signal.emit('Knowledge transfer completed')

            if dictio['p'].isChecked():
                dictio['PWInstance'].update_progress_signal.emit('Starting Pruning...')
                # model = prune_model_tensorflow(model, dictio['pr'], dictio['epochs'],
                #                        dictio['batch_size'],dictio['convert_tflite'],
                #                        dictio['q'].isChecked(), dictio['PWInstance'])

                model = basic_prune_model_tensorflow(model, dictio['pr'])

                dictio['PWInstance'].undo_progress_signal.emit()
                dictio['PWInstance'].update_progress_signal.emit('Pruning completed')

            if dictio['q'].isChecked():
                dictio['PWInstance'].update_progress_signal.emit('Starting Quantization...')
                model = quantize_model_tensorflow(model, dictio['df'])
                dictio['PWInstance'].undo_progress_signal.emit()
                dictio['PWInstance'].update_progress_signal.emit('Quantization completed')

            dictio['PWInstance'].final_size_signal.connect(dictio['PWInstance'].update_final_size)

            dictio['PWInstance'].update_progress_signal.emit('Saving final model...')
            os.chdir('./Models/Output Models')

            if dictio['convert_tflite'] or dictio['q'].isChecked():
                open(dictio['save_name'] + '.tflite', "wb").write(model)
                saved_model_path = dictio['save_name'] + '.tflite'
    
                if dictio['Compressed']:
                    with zipfile.ZipFile(dictio['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                        f.write(dictio['save_name'] + '.tflite')
                    saved_model_path = dictio['save_name'] + '.zip'
                    if not dictio['save_unziped']:
                        os.remove(dictio['save_name'] + '.tflite')

            elif dictio['Compressed']:
                model.save(dictio['save_name'] + '.h5')
                with zipfile.ZipFile(dictio['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                    f.write(dictio['save_name'] + '.h5')
                saved_model_path = dictio['save_name'] + '.zip'
    
                if not dictio['save_unziped']:
                    os.remove(dictio['save_name'] + '.h5')

            else:
    
                model.save(dictio['save_name'] + '.h5')
                saved_model_path = dictio['save_name'] + '.h5'

            dictio['PWInstance'].undo_progress_signal.emit()
            dictio['PWInstance'].update_progress_signal.emit('Model saved')

            dictio['PWInstance'].final_size_signal.emit(str(round(os.stat(saved_model_path).st_size / (1024 ^ 2), 3)))

            os.chdir('../../')
            self.finished.emit()
            return model
                
        elif dictio['framework'] == 'torch':
            model = torch.jit.load(dictio['model_path'], map_location=torch.device('cpu'))
            dictio['PWInstance'].undo_progress_signal.emit()
            dictio['PWInstance'].update_progress_signal.emit('Model initialized')
            if dictio['d'].isChecked():
                dictio['PWInstance'].update_progress_signal.emit('Starting Knowledge transfer...')
                teacher_model = torch.jit.load(dictio['teacher_model_path'], map_location=torch.device('cpu'))
                model = distill_model_pytorch(model, teacher_model,dictio['dataset_path'], dictio['batch_size'],
                                         dictio['temperature'], dictio['Alpha'], dictio['DKepochs'],
                                         dictio['PWInstance'])
                dictio['PWInstance'].undo_progress_signal.emit()
                dictio['PWInstance'].update_progress_signal.emit('Knowledge transfer completed')

            if dictio['p'].isChecked():
                dictio['PWInstance'].update_progress_signal.emit('Starting Pruning...')
                model = prune_model_pytorch(model, dictio['pr'], dictio['epochs'], dictio['batch_size'],
                                       dictio['q'].isChecked(), dictio['PWInstance'])
                dictio['PWInstance'].undo_progress_signal.emit()
                dictio['PWInstance'].update_progress_signal.emit('Pruning completed')

            if dictio['q'].isChecked():
                dictio['PWInstance'].update_progress_signal.emit('Starting Quantization...')
                model = quantize_model_pytorch(model, dictio['df'])
                dictio['PWInstance'].undo_progress_signal.emit()
                dictio['PWInstance'].update_progress_signal.emit('Quantization completed')

            dictio['PWInstance'].final_size_signal.connect(dictio['PWInstance'].update_final_size)

            dictio['PWInstance'].update_progress_signal.emit('Saving final model...')
            os.chdir('./Models/Output Models')

            # evaluating the model performance on mnist dataset
            correct = 0
            total = 0
            # _, test_loader = torch_load_mnist(batch_size=1)
            _,test_loader = load_pytorch_dataset(dictio['dataset_path'], batch_size=1)
            for images, labels in test_loader:
                images = images.view(-1, 28 * 28).to(torch.device('cpu'))
                labels = labels.to(torch.device('cpu'))
    
                output = model(images)
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            print(f'Accuracy of the final model: %.3f %%' % ((100 * correct) / total))

            # Save the model to a file
            model.save(dictio['save_name'] + '.pt')
            saved_model_path = dictio['save_name'] + '.pt'

            if dictio['Compressed']:
                # Open a gzip file for writing
                with zipfile.ZipFile(dictio['save_name'] + '.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
                    # Save the model to the file
                    f.write(dictio['save_name'] + '.pt')
                saved_model_path = dictio['save_name'] + '.zip'
                if not dictio['save_unziped']:
                    os.remove(dictio['save_name'] + '.pt')

            dictio['PWInstance'].undo_progress_signal.emit()
            dictio['PWInstance'].update_progress_signal.emit('Model saved')

            dictio['PWInstance'].final_size_signal.emit(str(round(os.stat(saved_model_path).st_size / (1024 ^ 2), 3)))

            os.chdir('../../')
            self.finished.emit()
            return model