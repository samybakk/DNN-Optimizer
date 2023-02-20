import tempfile
import tensorflow as tf
import keras
from keras.datasets import mnist
import tensorflow_model_optimization as tfmot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from Distiller import TfDistiller
from torch import nn, optim, quantization
import torch
from torch.quantization import quantize_fx
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
import cv2

import torchvision.datasets as dsets
import torchvision.transforms as transforms


# a funciton that takes a tensorflow model and returns a pruned version of it using pruning_ratio as the pruning ratio without re-training the model and without data
def basic_prune_model_tensorflow(model, pruning_ratio):
    # Get weights of the model
    weights = model.get_weights()
    
    # Flatten the weights into a 1D array for sorting
    flattened_weights = np.concatenate([layer.flatten() for layer in weights])
    
    # Compute the number of weights to prune
    num_weights_to_prune = int(len(flattened_weights) * pruning_ratio)

    # Sort the weights in descending order
    sorted_weights = np.sort(np.abs(flattened_weights))[::-1]

    # Find the threshold weight value to prune
    threshold = sorted_weights[num_weights_to_prune]

    # Set the weights above the threshold to zero
    pruned_weights = [np.where(np.abs(w) >= threshold, w, 0) for w in weights]

    # Set the pruned weights back into the model
    model.set_weights(pruned_weights)
    
    return model


# def prune_model_tensorflow(model, pruning_ratio,PEpochs,batch_size,convert_tflite,quantization,PWInstance):
#
#
#     train_data, train_labels, test_data, test_labels, end_step = load_mnist(epochs=PEpochs, batch_size=batch_size,
#     validation_split=0.1)
#
#     prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
#     pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=pruning_ratio / 3,
#                                                                  final_sparsity=pruning_ratio,
#                                                                  begin_step=0,
#                                                                  end_step=end_step)}
#
#     model_for_pruning = prune_low_magnitude(model, **pruning_params)
#     model_for_pruning.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                               metrics=['accuracy'])
#
#     model_for_pruning.summary()
#     logdir = tempfile.mkdtemp()
#
#     callbacks = [
#         tfmot.sparsity.keras.UpdatePruningStep(),
#         tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
#         TfPlotPruning(PWInstance)]
#
#     model_for_pruning.fit(train_data, train_labels,batch_size=batch_size, epochs=PEpochs, validation_split=0.1,
#     callbacks=callbacks)
#     model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
#
#     if convert_tflite == True and quantization == False:
#         converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
#         tflite_model = converter.convert()
#         return tflite_model
#
#     else:
#         return model_for_export


def prune_model_pytorch(model, pruning_ratio, PEpochs, batch_size, quantization, PWInstance):
    masks = []
    for param in model.parameters():
        masks.append(torch.ones_like(param))
    
    # Iterate over the model's parameters
    for param, mask in zip(model.parameters(), masks):
        # Sort the elements of the parameter in ascending order
        sorted_param = param.view(-1).abs().sort()[0]
        
        # Calculate the number of elements to prune
        n_elements_to_prune = int(sorted_param.numel() * pruning_ratio)
        
        # Calculate the pruning threshold
        pruning_threshold = sorted_param[n_elements_to_prune]
        
        # Set all elements below the pruning threshold to zero
        mask[param.abs() < pruning_threshold] = 0
    
    # Apply the pruning mask to the model's parameters
    for param, mask in zip(model.parameters(), masks):
        param.data.mul_(mask)
    
    return model


def quantize_model_tensorflow(model, desired_format):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if desired_format == 'int8':
        desired_format = tf.qint8
    elif desired_format == 'int16':
        desired_format = tf.qint16
    elif desired_format == 'int32':
        desired_format = tf.qint32
    elif desired_format == 'float16':
        desired_format = tf.float16
    else:
        desired_format = tf.float32
    
    converter.target_spec.supported_types = [desired_format]
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def quantize_model_pytorch(model, desired_format):
    if desired_format == 'int8':
        desired_format = torch.qint8
    elif desired_format == 'int16':
        desired_format = torch.float16
        print('int16 not availabe for pytorch | using float16 instead')
    elif desired_format == 'int32':
        desired_format = torch.qint32
    elif desired_format == 'float16':
        desired_format = torch.float16
    else:
        desired_format = torch.float32
    torch_quant_model = quantization.quantize_dynamic(model, {nn.Linear}, dtype=desired_format)
    # qconfig_dict = {
    #     "": quantization.default_dynamic_qconfig}
    # model_prepared = quantize_fx.prepare_fx(torch_quant_model, qconfig_dict)
    # torch_quant_model = quantize_fx.convert_fx(model_prepared)
    return torch_quant_model


def distill_model_tensorflow(model, teacher_model,dataset_path,batch_size, temperature, alpha, epochs, PWInstance):
    train_data, train_labels, test_data, test_labels = load_and_prep_images(dataset_path,(28,28), batch_size, batch_size, 0.1,True)
    
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
    #distiller.fit(train_data, epochs=epochs, callbacks=callbacks)

    print('Evaluating the distilled model')
    distiller.evaluate(test_data, test_labels)
    return model


def distill_model_pytorch(model, teacher_model, dataset_path, batch_size, temperature, alpha, epochs, PWInstance):
    # take a torch student model and a torch teacher model and distill them
    student_model = model
    teacher_model = model # FIX ME I CHANGED THIS FOR TESTING should be teacher_model
    # teacher_model = teacher_model.to(device)
    # student_model = student_model.to(device)
    teacher_model.eval()
    student_model.train()
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    train_loader, test_loader = load_pytorch_dataset(dataset_path, batch_size=batch_size)
    # train_loader, test_loader = torch_load_mnist(batch_size=batch_size)
    plotdk = PTPlotDK(PWInstance)
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.squeeze(dim=1).view(-1, 28 * 28).to(torch.device('cpu')), target.to(torch.device('cpu'))
            optimizer.zero_grad()
            output = student_model(data)
            with torch.no_grad():
                teacher_output = teacher_model(data)
            distillation_loss = loss_fn(
                nn.functional.log_softmax(output / temperature, dim=1),
                nn.functional.softmax(teacher_output / temperature, dim=1),
            ) * (temperature ** 2)
            student_loss = nn.functional.cross_entropy(output, target)
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
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
            images = images.view(-1, 28 * 28).to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))
            
            output = student_model(images)
            loss = nn.functional.cross_entropy(output, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        plotdk.on_epoch_end(epoch + 1, 100 * correct / total)
        # plotdk.on_epoch_end(epoch+1, loss_sum/len(test_loader))
        print(f'Epoch : {epoch + 1} Accuracy of the model: %.3f %%' % ((100 * correct) / total))
    return student_model


def load_pytorch_dataset(dataset_path, batch_size=64, test_batch_size=128, train_split=0.9):
    # Set the device to use for training
    # device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    
    transform_fn = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load the custom dataset
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform_fn)
    
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders for the dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, test_loader


# def get_label(file_path):
#     # Convert the path to a list of path components
#     parts = tf.strings.split(file_path, os.path.sep)
#     # The second to last is the class-directory
#     one_hot = parts[-2] == class_names
#     # Integer encode the label
#     return tf.argmax(one_hot)
#
# def decode_img(img,img_size):
#   # Convert the compressed string to a 3D uint8 tensor
#   img = tf.io.read_file(img, channels=img_size[2])
#
#   # Resize the image to the desired size
#   return tf.image.resize(img, [1,img_size[0]*img_size[1]])
#
# def process_path(file_path,img_size):
#   label = get_label(file_path)
#   # Load the raw data from the file as a string
#   img = tf.io.read_file(file_path)
#   img = decode_img(img,img_size)
#   return img, label
def load_and_prep_images(datadir, img_size,batch_size=64, test_batch_size=128, validation_split=0.1,black_and_white = False):
    def preprocess_images(imgs,img_size=img_size,black_and_white=black_and_white):
        #imgs = imgs.astype('float32') / 255.0
        if black_and_white:
             imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY)
        imgs = cv2.resize(imgs, img_size, interpolation=cv2.INTER_CUBIC)
        return imgs
    
    datagen = ImageDataGenerator(preprocessing_function=preprocess_images,rescale=1./255,validation_split=validation_split)
    # Use the .flow_from_directory() method to load images from the directory
    # and apply the specified transformations
    train_gen = datagen.flow_from_directory(datadir,
                                            target_size=img_size,
                                            class_mode='categorical',
                                            batch_size=batch_size,
                                            color_mode = 'grayscale' if black_and_white else 'rgb',
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
    # image_count = len(list(data_dir.glob('*/*.jpg')))
    # list_ds = tf.data.Dataset.list_files(str(datadir / '*/*'), shuffle=False)
    # list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    #
    # val_size = int(image_count * validation_split)
    # train_ds = list_ds.skip(val_size)
    # val_ds = list_ds.take(val_size)
    # train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    # val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    
    
    # Load the data
    trainData = tf.keras.utils.image_dataset_from_directory(
        datadir,
        labels='inferred',
        label_mode='categorical',
        validation_split=validation_split,
        subset='training',
        seed=178,
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
        seed=178,
        image_size=img_size[:2],
        batch_size=test_batch_size,
        crop_to_aspect_ratio=True
    )

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    resizing_layer = tf.keras.layers.Resizing(img_size[0],img_size[1],interpolation='bilinear',crop_to_aspect_ratio=False)
    
    train_data = trainData.map(lambda x, y: (resizing_layer(normalization_layer(x)), y))
    test_data = testData.map(lambda x, y: (resizing_layer(normalization_layer(x)), y))
    


    # Return the datasets
    return train_data, test_data


def load_mnist(epochs, batch_size=128, validation_split=0.1):
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


class TfPlotPruning(keras.callbacks.Callback):
    def __init__(self, PW):
        self.PW = PW
        self.PW.on_pruning_epoch_end_signal.connect(self.PW.update_pruning_graph_data)
    
    def on_epoch_end(self, epoch, logs={}):
        self.PW.on_pruning_epoch_end_signal.emit(epoch, logs['val_accuracy'])


class TfPlotDK(keras.callbacks.Callback):
    def __init__(self, PW):
        self.PW = PW
        self.PW.on_dk_epoch_end_signal.connect(self.PW.update_dk_graph_data)
    
    def on_epoch_end(self, epoch, logs={}):
        self.PW.on_dk_epoch_end_signal.emit(epoch, logs['sparse_categorical_accuracy'])


class PTPlotPruning():
    def __init__(self, PW):
        self.PW = PW
        self.PW.on_dk_epoch_end_signal.connect(self.PW.update_pruning_graph_data)
    
    def on_epoch_end(self, epoch, loss):
        self.PW.on_dk_epoch_end_signal.emit(epoch, loss.item())


class PTPlotDK():
    def __init__(self, PW):
        self.PW = PW
        self.PW.on_dk_epoch_end_signal.connect(self.PW.update_dk_graph_data)
    
    def on_epoch_end(self, epoch, loss):
        self.PW.on_dk_epoch_end_signal.emit(epoch, loss)
