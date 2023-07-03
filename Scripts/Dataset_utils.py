import os

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from torch.utils.data import SubsetRandomSampler
import torchvision
from utilities import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import torch
import time
from Model_utils import *
from val import run as val_run
from utils.dataloaders import *


def test_inference_speed(model, device, input_size=(1, 3, 32, 32), num_samples=100, num_warmups=10, logger=None):
    model.to(device)
    model.eval()
    
    x = torch.rand(size=input_size).to(device)
    
    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_avg = elapsed_time / num_samples
    
    return elapsed_time_avg


def test_inference_speed_and_accuracy(model, val_loader, device, Dict, logger,half, is_yolo):
    if isinstance(model, str):
        model_path = model
    
    if is_yolo:
        if device == 'cuda':
            device = '0'
        val_accuracy, val_accuracy_per_class, inference_speed = val_run(Dict['dataset_path'] + '/data.yaml', model_path,
                                                                        batch_size=1, imgsz=640, conf_thres=0.001,
                                                                        iou_thres=0.6,
                                                                        device=device, workers=1, save_json=False,
                                                                        plots=False)
        
        print(f'val_accuracy : {val_accuracy}')
        print(f'val_accuracy_per_class : {val_accuracy_per_class}')
        print(f'inference_speed : {inference_speed}')
    else:
        
        # Measure the inference time
        model = model.to(device=device)
        model = model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            
            start_time = time.time()
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if half:
                    inputs = inputs.half()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
                total_samples += inputs.size(0)
            model.train()
            
            # Calculate the inference speed and accuracy
            end_time = time.time()
            inference_speed = (end_time - start_time) / (len(val_loader) * inputs.size(0))
            val_accuracy = total_correct / total_samples
    
    return inference_speed, val_accuracy


def evaluate_tensorflow_model(model, val_data, val_labels):
    import time
    
    # Measure the average evaluation time
    num_samples = len(val_data)
    start_time = time.time()
    _, accuracy = model.evaluate(val_data, val_labels)
    end_time = time.time()
    avg_eval_time = (end_time - start_time) / num_samples
    
    return accuracy, avg_eval_time


def load_tensorflow_data(dataset_path, img_size=(32,32), batch_size=8):
    if 'cifar-10' in dataset_path:
        (training_images, training_labels), (
        validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
        # Normalize the image data
        training_images = training_images / 255.0
        validation_images = validation_images / 255.0

        # Convert labels to one-hot encoding
        num_classes = 10  # Number of classes in CIFAR-10 dataset
        training_labels = tf.keras.utils.to_categorical(training_labels, num_classes)
        validation_labels = tf.keras.utils.to_categorical(validation_labels, num_classes)
        return training_images, training_labels, validation_images, validation_labels
    
    # Create a dataset for the training data
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + '/train',
        image_size=img_size,
        batch_size=batch_size
    )
    
    # Create a dataset for the validation data
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path + '/val',
        image_size=img_size,
        batch_size=batch_size
    )
    
    # Normalize the data
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
    
    # Separate the data and labels
    train_data = train_dataset.unbatch().map(lambda x, y: x)
    train_labels = train_dataset.unbatch().map(lambda x, y: y)
    val_data = val_dataset.unbatch().map(lambda x, y: x)
    val_labels = val_dataset.unbatch().map(lambda x, y: y)
    
    return train_data, train_labels, val_data, val_labels

def load_pytorch_cifar10(dataset_path):
    import os
    import pickle
    
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    train_batch_list = []
    train_labels_list = []
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if 'batch' in file and 'meta' not in file:
                filepath = os.path.join(subdir, file)
                if 'test' in file:
                    test_batch = unpickle(filepath)
                else:
                    batch = unpickle(filepath)
                    train_batch_list.append(batch[b'data'])
                    train_labels_list.append(batch[b'labels'])
    # train_dataset,val_dataset = unpickle(dataset_path)
    
    images = np.concatenate(train_batch_list, dtype=np.float32)
    labels = np.concatenate(train_labels_list)
    
    test_images = test_batch[b'data'].astype(np.float32)
    test_labels = np.array(test_batch[b'labels'])
    
    # Reshape the data to the correct dimensions
    images = images.reshape((-1, 3, 32, 32))
    # images = np.transpose(images, (0, 2, 3, 1))
    
    test_images = test_images.reshape((-1, 3, 32, 32))
    # test_images = np.transpose(test_images, (0, 2, 3, 1))
    
    # Convert the data to PyTorch tensors
    images_tensor = torch.from_numpy(images).float() / 255.0
    labels_tensor = torch.from_numpy(labels)
    
    test_images_tensor = torch.from_numpy(test_images).float() / 255.0
    test_labels_tensor = torch.from_numpy(test_labels)
    
    # Define a dataset object to hold the data
    train_dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor)
    val_dataset = torch.utils.data.TensorDataset(test_images_tensor, test_labels_tensor)
    
    return train_dataset, val_dataset


def load_pytorch_dataset(dataset_path, batch_size=8, val_batch_size=16, train_fraction=1, val_fraction=1):
    print(f'Loading dataset at path {dataset_path}')
    if 'cifar-10' in dataset_path.split(os.sep)[-1]:
        train_dataset, val_dataset = load_pytorch_cifar10(dataset_path=dataset_path)
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define the transforms to apply to the validation data
        val_transforms = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        ])
        
        # Load the custom dataset
        try:
            train_dataset = torchvision.datasets.ImageFolder(root=dataset_path + '/train', transform=train_transforms)
            val_dataset = torchvision.datasets.ImageFolder(root=dataset_path + '/val', transform=val_transforms)
        except:
            with open(dataset_path + '/data.yaml') as f:
                data_dict = yaml.load(f, Loader=yaml.SafeLoader)
            train_path = data_dict['train']
            val_path = data_dict['val']
            imgsz = 640
            # gs = max(int(model.stride.max()), 32)  # grid size (max stride)
            stride = 32
            single_cls = False
            train_loader = create_dataloader(train_path,
                                             imgsz,
                                             batch_size,
                                             stride,
                                             single_cls,
                                             workers=2,
                                             prefix=f'train: ')[0]
            val_loader = create_dataloader(val_path,
                                           imgsz,
                                           batch_size,
                                           stride,
                                           single_cls,
                                           workers=2,
                                           prefix=f'train: ')[0]
            print('train_loader', train_loader)
            print('val_loader', val_loader)
            return train_loader, val_loader
    
    # Create a subset of the training dataset with a fraction of the samples
    train_num_samples = len(train_dataset)
    train_subset_size = int(train_num_samples * train_fraction)
    
    train_subset_indices = torch.randperm(train_num_samples)[:train_subset_size]
    train_subset_sampler = SubsetRandomSampler(train_subset_indices)
    
    # Create a subset of the validation dataset with a fraction of the samples
    val_num_samples = len(val_dataset)
    val_subset_size = int(val_num_samples * val_fraction)
    
    val_subset_indices = torch.randperm(val_num_samples)[:val_subset_size]
    val_subset_sampler = SubsetRandomSampler(val_subset_indices)
    
    # Create data loaders for the dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               sampler=train_subset_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                                             sampler=val_subset_sampler)
    
    print('finished loading dataset')
    return train_loader, val_loader


def load_and_prep_images(datadir, img_size, batch_size=64, test_batch_size=128, validation_split=0.1,
                         black_and_white=False):
    def preprocess_images(imgs, img_size=img_size, black_and_white=black_and_white):
        # imgs = imgs.astype('float32') / 255.0
        if black_and_white:
            imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY)
        imgs = cv2.resize(imgs, img_size, interpolation=cv2.INTER_CUBIC)
        return imgs
    
    datagen = ImageDataGenerator(preprocessing_function=preprocess_images, rescale=1. / 255,
                                 validation_split=validation_split)
    # Use the .flow_from_directory() method to load images from the directory
    # and apply the specified transformations
    train_gen = datagen.flow_from_directory(datadir,
                                            target_size=img_size,
                                            class_mode='categorical',
                                            batch_size=batch_size,
                                            color_mode='grayscale' if black_and_white else 'rgb',
                                            shuffle=False,
                                            subset='training')
    
    test_gen = datagen.flow_from_directory(datadir,
                                           target_size=img_size,
                                           class_mode='categorical',
                                           batch_size=test_batch_size,
                                           color_mode='grayscale' if black_and_white else 'rgb',
                                           shuffle=False,
                                           subset='validation')
    # Generate the data and labels from the generator
    train_images = [train_gen for _ in range(len(train_gen))]
    test_images = [test_gen for _ in range(len(test_gen))]
    train_labels = train_gen.classes
    test_labels = test_gen.classes
    #
    # train_images = train_images.reshape(train_images.shape[0], -1)
    # test_images = test_images.reshape(test_images.shape[0], -1)
    
    return train_images, train_labels, test_images, test_labels


def load_tensorflow_dataset(datadir, img_size, batch_size=64, test_batch_size=128, validation_split=0.1):
    """Loads the dataset from a directory.
    Args:
        datadir: String. The path to the dataset directory.
        img_size: Tuple. The size of the images to load.
        batch_size: Integer. The batch size to use for training.
        test_batch_size: Integer. The batch size to use for testing.
        train_split: Float. The fraction of the dataset to use for training.
    Returns:
        traindata: Tensor. The train data.
        trainlabels: Tensor. The train labels.
        testdata: Tensor. The test data.
        testlabels: Tensor. The test labels.
    """
    
    # Load the data
    trainData = tf.keras.utils.image_dataset_from_directory(
        datadir,
        labels='inferred',
        label_mode='categorical',
        validation_split=validation_split,
        subset='training',
        seed=1,
        image_size=img_size[:2],
        batch_size=batch_size,
        crop_to_aspect_ratio=True
    )
    testData = tf.keras.utils.image_dataset_from_directory(
        datadir,
        labels='inferred',
        label_mode='categorical',
        validation_split=validation_split,
        subset='validation',
        seed=1,
        image_size=img_size[:2],
        batch_size=test_batch_size,
        crop_to_aspect_ratio=True
    )
    
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    resizing_layer = tf.keras.layers.Resizing(img_size[0], img_size[1], interpolation='bilinear',
                                              crop_to_aspect_ratio=False)
    
    train_data = trainData.map(lambda x, y: (resizing_layer(normalization_layer(x)), y))
    test_data = testData.map(lambda x, y: (resizing_layer(normalization_layer(x)), y))
    
    # Return the datasets
    return train_data, test_data


def load_mnist():
    (train_data, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    
    train_data = train_data / 255.0
    test_data = test_images / 255.0
    
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
    
    return train_data, train_labels, test_data, test_labels


def torch_load_mnist(batch_size=128):
    train_data = dsets.MNIST(root='./data', train=True,
                             transform=transforms.ToTensor(), download=True)
    
    test_data = dsets.MNIST(root='./data', train=False,
                            transform=transforms.ToTensor())
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=False)
    
    return train_loader, test_loader