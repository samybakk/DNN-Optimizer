import sys
sys.path.insert(0, './yolov5')
import argparse
from Model_utils import *
from Worker import worker




"""
Run this script to run the model optimization pipeline without interface.
Exemple of use:
python main_interface.py --current_model_path Models/Input Models/PCODD_yolov5_tiny.pt
    --Pruning True --Quantization False --Knowledge_Distillation False --current_dataset_path Datasets/PCODD
    --batch_size 2 --PruningRatio 0.5  --PruningEpochs 1 --DesiredFormat int8 --Teacher_Model_Path Models/Teacher Models/PCODD_yolov5l.pt
    --Temperature 6 --Alpha 0.8 --save_name First_ni_test

"""



def arg_parser():
    
    parser = argparse.ArgumentParser(description='Model Optimization')
    parser.add_argument('--current_model_path', type=str,
                        help='path to the model to be compressed')
    parser.add_argument('--Pruning', type=bool, default=False,
                        help='whether to prune the model')
    parser.add_argument('--Quantization', type=bool, default=False,
                        help='whether to quantize the model')
    parser.add_argument('--Knowledge_Distillation', type=bool, default=False,
                        help='whether to apply Knowledge Distillation')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for training')
    parser.add_argument('--PruningRatio', type=float, default=0.5,
                        help='ratio of pruning')
    parser.add_argument('--current_dataset_path', type=str,
                        help='path to the dataset')
    parser.add_argument('--train_fraction', type=float, default=1,
                        help='fraction of the dataset to be used for training')
    parser.add_argument('--validation_fraction', type=float, default=1,
                        help='fraction of the dataset to be used for validation')
    parser.add_argument('--PruningEpochs', type=int, default=1,
                        help='number of epochs for pruning')
    parser.add_argument('--KDEpochs', type=int, default=1,
                        help='number of epochs for Knowledge Distillation')
    parser.add_argument('--DesiredFormat', type=str, default='int8',
                        help='desired format of the quantized model')
    parser.add_argument('--Teacher_Model_Path', type=str,
                        help='path to the teacher model')
    parser.add_argument('--Temperature', type=int, default=1,
                        help='temperature for Knowledge Distillation')
    parser.add_argument('--save_name', type=str, default='Model_1',
                        help='name of the compressed model')
    parser.add_argument('--SaveUnziped', type=bool, default=False,
                        help='whether to save the compressed model in unzipped format')
    parser.add_argument('--ConvertTFLite', type=bool, default=False,
                        help='whether to convert the compressed model to tflite')
    parser.add_argument('--Compressed', type=bool, default=True,
                        help='whether to use the compressed model for training')
    parser.add_argument('--Alpha', type=float, default=0.8,
                        help='alpha for Knowledge Distillation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the optimization on')
    parser.add_argument('--framework', type=str, default='torch',
                        help='framework of the model')
    opt = parser.parse_args()

    return opt
    
def run(opt):
    
    dictio = {'model_path': opt.current_model_path,
              'Pruning': opt.Pruning,
              'Quantization': opt.Quantization,
              'Knowledge_Distillation': opt.Knowledge_Distillation, 'batch_size': opt.batch_size,
              'pruning_ratio': opt.PruningRatio, 'dataset_path': opt.current_dataset_path,
              'train_fraction': opt.train_fraction, 'validation_fraction': opt.validation_fraction,
              'pruning_epochs': opt.PruningEpochs,
              'desired_format': opt.DesiredFormat,
              'teacher_model_path': opt.Teacher_Model_Path,
              'KD_temperature': opt.Temperature, 'save_name': opt.save_name,
              'save_unziped': opt.SaveUnziped,
              'convert_tflite': opt.ConvertTFLite,
              'Compressed': opt.Compressed, 'KD_alpha': opt.Alpha,
              'KD_epochs': opt.KDEpochs, 'device': opt.device,
              'framework': opt.framework}

    worker(dictio)
    
    
    # index = 0
    # pr = [2,3,4,5,6,7,8,9]
    # pre = [2,3,4,5,5,5,5,5]
    # for pr, pre in zip(pr,pre):
    #     dictio = {'model_path': opt.current_model_path,
    #           'Pruning': opt.Pruning,
    #           'Quantization': opt.Quantization,
    #           'Knowledge_Distillation': opt.Knowledge_Distillation, 'batch_size': opt.batch_size,
    #           'pruning_ratio': pr/10, 'dataset_path': opt.current_dataset_path,
    #           'train_fraction': opt.train_fraction, 'validation_fraction': opt.validation_fraction,
    #           'pruning_epochs': pre,
    #           'desired_format': opt.DesiredFormat,
    #           'teacher_model_path': opt.Teacher_Model_Path,
    #           'KD_temperature': opt.Temperature, 'save_name': f'yolov5s_pruned_pr_{pr/10}_ep_{pre}',
    #           'save_unziped': opt.SaveUnziped,
    #           'convert_tflite': opt.ConvertTFLite,
    #           'Compressed': opt.Compressed, 'KD_alpha': opt.Alpha,
    #           'KD_epochs': opt.KDEpochs, 'device': opt.device,
    #           'framework': opt.framework}
    #
    #     print('\n\nNew Process | index : ',index)
    #     worker(dictio)
    #     index += 1
        
if __name__ == "__main__":
    opt =arg_parser()
    run(opt)
