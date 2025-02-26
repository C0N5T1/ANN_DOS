import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from functools import reduce
from PIL import Image

import torch
from torchvision import datasets, transforms

############# INITIALIZATION #############

# Transformation (if needed)
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# initalizing the train datasets and sorting them by numbers (these are still tensors)
indices_0 = (mnist_train.targets == 0)
indices_1 = (mnist_train.targets == 1)
indices_2 = (mnist_train.targets == 2)
indices_3 = (mnist_train.targets == 3)
indices_4 = (mnist_train.targets == 4)
indices_5 = (mnist_train.targets == 5)
indices_6 = (mnist_train.targets == 6)
indices_7 = (mnist_train.targets == 7)
indices_8 = (mnist_train.targets == 8)
indices_9 = (mnist_train.targets == 9)

train_0 = mnist_train.data[indices_0]
train_1 = mnist_train.data[indices_1]
train_2 = mnist_train.data[indices_2]
train_3 = mnist_train.data[indices_3]
train_4 = mnist_train.data[indices_4]
train_5 = mnist_train.data[indices_5]
train_6 = mnist_train.data[indices_6]
train_7 = mnist_train.data[indices_7]
train_8 = mnist_train.data[indices_8]
train_9 = mnist_train.data[indices_9]

class_tensors = {
    0: train_0,
    1: train_1,
    2: train_2,
    3: train_3,
    4: train_4,
    5: train_5,
    6: train_6,
    7: train_7,
    8: train_8,
    9: train_9,
}

############# CREATING DATASETS #############

def create_binary_dataset(samples_per_class, shaped_data=True, normalize=False, reduce=False, shape=(10, 10)):
    """
    creates a Dataset of flattened np.ndarrays to use in the simulations. You can state the number of samples per Class you want
    use 'class_tensors' or 'normalized_class_tensors'
    """
    
    data = class_tensors
    
    dataset = []
    labels = []

    selected_classes = list(samples_per_class.keys())
    if len(selected_classes) != 2:
        raise ValueError("samples_per_class must contain exactly two classes for binary classification.")
    
    label_mapping = {selected_classes[0]: -1, selected_classes[1]: 1}
    
    for label, tensors in data.items():
        
        num_samples = samples_per_class.get(label, 0)
        
        if num_samples > 0:
            
            indices = torch.randperm(len(tensors))[:num_samples]
            selected_tensors = tensors[indices]
            
            if reduce:
                
                # if reduce is True the tensors are turned to array to images, rezised turned to arrays and flattened
                dataset.extend([np.array(Image.fromarray(tensor.numpy()).resize(shape, Image.LANCZOS)).flatten() for tensor in selected_tensors])
                
                #dataset.extend([tensor.numpy().astype(np.float64).flatten() for tensor in selected_tensors])
            
            else:
                dataset.extend([tensor.numpy().astype(np.float64).flatten() for tensor in selected_tensors])
            
            labels.extend([label_mapping[label]] * num_samples)
  
    if shaped_data:
        length = len(dataset[0])
        dat = np.array(dataset)
        dataset = [vec.reshape((length, 1)) for vec in dat]
    
    dat = np.array(dataset)/255
    lab = np.array(labels)
    
    if normalize:
        
        D_mean = np.mean(dat)
        D_std = np.std(dat)
        
        dat = (dat - D_mean)/D_std

    return dat, lab

def generate_random_dataset(P: int, N: int, teacher_vec: Optional[np.ndarray]=None):
    
    if teacher_vec is None:
        teacher_vec = np.random.choice([-1.0, 1.0], size=N)
    
    dataset = [np.random.choice([1, -1], N).astype(np.float64) for i in range(P)]
    dots = [np.dot(dataset[i], teacher_vec) for i in range(P)]
    labels = np.where(np.sign(dots) == 0, 1, np.sign(dots)).astype(np.float64)
    
    return np.array(dataset), np.array(labels)

# example use
"""
from ANN_DOS import *

samples_per_class = {0: 500, 1: 500}

dataset, labels = create_binary_dataset(class_tensors, samples_per_class)"""

############# SAVING/LOADING #############

def save_data_with_metadata_npz(data, filename, metadata):
    """
    Saves data and metadata to an .npz file.
    
    Parameters:
        data (np.ndarray): The data array to save.
        filename (str): The path for the .npz file.
        metadata (dict): Metadata to save with the data.
    """
    # Save data with metadata as a dictionary inside the .npz file
    np.savez(filename, data=data, metadata=metadata)

def load_data_with_metadata_npz(filename):
    with np.load(filename, allow_pickle=True) as data:
        dataset = data['data']
        metadata = data['metadata'].item()  # Retrieve dictionary from array
        
    return dataset, metadata

def print_metadata(filenames):
    
    if isinstance(filenames, str):
        filenames = [filenames]
    
    for name in filenames:
        _, metadata = load_data_with_metadata_npz(name)
        
        print(metadata)
        
############# HELPER FUNCTIONS #############

def isFlat(arr: np.ndarray, tolerance: float = 0.2) -> bool:
    
    mean = np.mean(arr)
    
    if np.min(arr) > mean * (1 - tolerance) and np.max(arr) < mean * (1 + tolerance):
        return True
    return False

############# WANG LANDAU ALGORITHM #############

def WangLandau(arr: np.ndarray, labels: np.ndarray, f: float = np.e, nodes: list = [1], epsilon: float = 1e-5, flatness: float = 0.2, loop=None, bin_len: int=1, continuous: bool=False, regen_ratio: float = 0.3):

    P = len(labels)                 # number of input vectors
    N = len(arr[0])                 # length of input vectors
    
    ln_epsilon = np.log(1 + epsilon)
    ln_f = np.log(f)
    
    labels = labels.reshape((P, 1))
    input_matrix = np.stack(arr, axis=1).reshape(N, P)       # stacking the input vectors for optimized matrix multiplications
    
    if bin_len == 1:
        num_bins = P + 1
    else:
        num_bins = int(P/bin_len)
    
    ln_g = np.zeros(num_bins)
    H = np.zeros(num_bins)  
    
    node_list = [N] + nodes    # making sure the nodes have the right first and last entries
    if node_list[-1] != 1:
        node_list.append(1)

    num_loops = int(np.sum([node_list[i] * node_list[i+1] for i in range(len(node_list)-1)]) * 10)   
    print(num_loops)  
    
    ##########################################################################
    # helper function for managing bins
    if bin_len == 1:
        def energy_to_bin(energy):
            return energy
    else:
        def energy_to_bin(energy):
            if energy == P:
                return int(np.floor(energy / bin_len)) - 1
            else:
                return int(np.floor(energy / bin_len))
       
    # helper function for creating the layers of the nn 
    if continuous:
        
        def create_layers(node_list):
            return [np.random.uniform(-1, 1, size=(node_list[i+1], node_list[i])) for i in range(len(node_list) - 1)]
        
        def update_layers(layers):
            new_layers = []
            for layer in layers:
                mask = np.random.rand(*layer.shape) < regen_ratio
                new_layer = layer.copy()
                new_layer[mask] = np.random.uniform(-1, 1, size=mask.sum())
                new_layers.append(new_layer)
            return new_layers
    else:
        
        def create_layers(node_list):
            return [(np.random.randint(0, 2, size=(node_list[i+1], node_list[i])) * 2 - 1).astype(float) for i in range(len(node_list) - 1)]
        
        def update_layers(layers):
            new_layers = []
            for layer in layers:
                mask = np.random.rand(*layer.shape) < regen_ratio
                new_layer = layer.copy()
                new_layer[mask] = (np.random.randint(0, 2, size=mask.sum()) * 2 -1).astype(float)
                new_layers.append(new_layer)
            return new_layers
        
    ##########################################################################    
    
    layers = create_layers(node_list)
        
    print(f'shape of layers: {[np.shape(layers[i]) for i in range(len(layers))]}')
    
    # more efficient matrix multiplication
    result_matrix = input_matrix  # Start with the stacked inputs
    for layer in layers:
        result_matrix = np.sign(layer @ result_matrix)  # Batch matrix multiplication
    outputs = result_matrix.reshape(P, 1)  # Flatten results into a 1D array
    
    E = energy_to_bin(np.sum(labels * np.sign(outputs) < 0))
    
    iterations = 1 
    
    while ln_f > ln_epsilon:
    
        random_numbers = np.log(np.random.rand(num_loops))
        
        # pre calculating all random layers in one loop
        layers_list = [layers]
        for _ in range(num_loops + 1):
            layers_list.append(update_layers(layers_list[-1]))    
            
        del layers_list[0]
        
        for i, random_number in enumerate(random_numbers):
            
            layers = layers_list[i]
            
            result_matrix = input_matrix  # Start with the stacked inputs
            for layer in layers:
                result_matrix = np.sign(layer @ result_matrix)  # Batch matrix multiplication
            outputs = result_matrix.reshape(P, 1)  # Flatten results into a 1D array
            E2 = energy_to_bin(np.sum(labels * np.sign(outputs) < 0))
            
            # print(E, E2)
            ln_p_acc = ln_g[E] - ln_g[E2]
            
            if random_number <= ln_p_acc:
                E = E2
                
            ln_g[E] += ln_f
            H[E] += 1  

        
        if isFlat(H, flatness):
            
            print(f'H is flat after {iterations} loops of {num_loops} iterations')
            H[:] = 0
            ln_f *= 0.5
            
            iterations = 0
            
        iterations += 1
        
    g = np.array([1 / np.sum(np.exp(ln_g - ln_g[i])) for i in range(len(ln_g))]) * P / bin_len

    return g