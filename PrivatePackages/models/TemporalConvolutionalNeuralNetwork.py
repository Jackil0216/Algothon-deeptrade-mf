import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torcheval.metrics import MeanSquaredError, R2Score, MulticlassAccuracy, MulticlassF1Score
from torch.utils.data import Dataset, DataLoader



class CustomDataLoader(Dataset):
    def __init__(self, x, y = None):
        self.x = torch.FloatTensor(np.array(x))
        if y is not None:
            if isinstance(y, (np.ndarray, list)):
                self.y = torch.FloatTensor(np.array(y))
            elif isinstance(y, (pd.core.series.Series, pd.core.frame.DataFrame)):
                self.y = y
        self.len = len(x)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if hasattr(self, 'y'):
            if isinstance(self.y, (pd.core.series.Series)):
                y = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
                return self.x[idx], y
            elif isinstance(self.y, (pd.core.frame.DataFrame)):
                y = torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)
                return self.x[idx], y
            else:
                return self.x[idx], self.y[idx]
        else:
            return self.x[idx]



class dense_layer(nn.Module):

    def __init__(self,
        input_dimension,
        output_dimension,
        dropout_prob,
        batch_normalisation,
        activation_function,
        random_state):

        super().__init__()

        torch.manual_seed(random_state)

        self.ACTIVATION_FUNCTIONS_MAP = {'relu': nn.ReLU(),
                                    'sigmoid': nn.Sigmoid(),
                                    'tanh': nn.Tanh(),
                                    'softmax': nn.Softmax(dim=1)}

        if batch_normalisation:
            self.layer = nn.Sequential(
                nn.Linear(input_dimension, output_dimension),
                nn.BatchNorm1d(output_dimension),
                self.ACTIVATION_FUNCTIONS_MAP[activation_function],
                nn.Dropout(dropout_prob)
            )
        else:
            self.layer = nn.Sequential(
                nn.Linear(input_dimension, output_dimension),
                self.ACTIVATION_FUNCTIONS_MAP[activation_function],
                nn.Dropout(dropout_prob)
            )

    def forward(self, x):
        return self.layer(x)


class residual_layer(nn.Module): ## only useable for constant

    def __init__(self,
        dimension,
        dropout_prob,
        batch_normalisation,
        activation_function,
        random_state):

        super().__init__()

        torch.manual_seed(random_state)

        self.ACTIVATION_FUNCTIONS_MAP = {'relu': nn.ReLU(),
                                    'sigmoid': nn.Sigmoid(),
                                    'tanh': nn.Tanh(),
                                    'softmax': nn.Softmax(dim=1)}

        if batch_normalisation:
            self.layer = nn.Sequential(
                nn.Linear(dimension, dimension),
                nn.BatchNorm1d(dimension),
                self.ACTIVATION_FUNCTIONS_MAP[activation_function]
            )

        else:
            self.layer = nn.Sequential(
                nn.Linear(dimension, dimension),
                self.ACTIVATION_FUNCTIONS_MAP[activation_function]
            )

        self.linear = nn.Linear(dimension, dimension)

        self.activation = self.ACTIVATION_FUNCTIONS_MAP[activation_function]
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):
        y = self.layer(x)
        y = self.linear(y)

        y = self.activation(x+y)

        return self.dropout(y)




class Convolutional1DLayer(nn.module):

    def __init__(self,
        input_channel,
        output_channel,
        convolution_kernel_dim,
        convolution_stride,
        activation_function,
        batch_normalisation, 
        pool_type,
        pool_kernel, # TODO: Dilation?
        random_state,
        input_window_size,
        convolution_padding = 'same',
        convolution_padding_mode = 'zeros',
        pooling_padding = True,
        ):

        super().__init__()

        torch.manual_seed(random_state)

        self.ACTIVATION_FUNCTIONS_MAP = {'relu': nn.ReLU(),
                                    'sigmoid': nn.Sigmoid(),
                                    'tanh': nn.Tanh(),
                                    'softmax': nn.Softmax(dim=1)}
        
        self.batch_normalisation = batch_normalisation

        self.POOL_MAP = {'MaxPool': nn.MaxPool1d, 'AvgPool': nn.AvgPool1d}

        self.convolution_layer = nn.Conv1d(in_channels = input_channel,  
                                    out_channel = output_channel,
                                    kernel_size = convolution_kernel_dim,
                                    stride = convolution_stride,
                                    padding = convolution_padding,
                                    padding_mode = convolution_padding_mode)
        
        if batch_normalisation:
            self.batchnorm_layer = nn.BatchNorm1d(output_channel)

        self.activation = self.ACTIVATION_FUNCTIONS_MAP[activation_function]

        if input_window_size <= pool_kernel:
            pooling_padding = 0 # prevent when it can't condense down to 1 value

        self.pooling_layer = self.POOL_MAP[pool_type](kernel_size = pool_kernel, 
                                                      stride = pool_kernel, 
                                                      padding = pooling_padding)



    def forward(self, x):
        y = self.convolution_layer(x)

        if self.batch_normalisation:
            y = self.batchnorm_layer(y)

        y = self.activation(y)

        return self.pooling_layer(y)




class TemporalConvolutionalNeuralNetworkRegressor(nn.Module):

    def __init__(self,
            n_features,
            lookback_window_size,
            cnn_n_hidden_layers,
            output_channels_per_input_channel,
            convolution_kernel_dim,
            convolution_stride,
            activation_function,
            pool_type,
            pool_kernel,
            dense_n_hidden_layers,
            dense_hidden_layer_embed_dim,
            dropout_prob,
            output_size,
            batch_normalisation,
            random_state,
            dense_layer_type = 'Dense'):

        super(TemporalConvolutionalNeuralNetworkRegressor, self).__init__()

        torch.manual_seed(random_state)

        self.cnn_n_hidden_layers = cnn_n_hidden_layers

        self.convolutional1dlayers = nn.ModuleList()

        # first convolutional layer
        self.convolutional1dlayers.append(Convolutional1DLayer(input_channel = n_features, 
                                    output_channel = output_channels_per_input_channel * n_features,
                                    convolution_kernel_dim=convolution_kernel_dim,
                                    convolution_stride = convolution_stride, 
                                    activation_function = activation_function,
                                    batch_normalisation = batch_normalisation,
                                    pool_type = pool_type,
                                    pool_kernel = pool_kernel,
                                    random_state=random_state,
                                    input_window_size = lookback_window_size))

        lookback_window_size = np.ceil(lookback_window_size/pool_kernel)

        for i in range(1, self.cnn_n_hidden_layers):
            
            self.convolutional1dlayers.append(Convolutional1DLayer(input_channel = n_features* output_channels_per_input_channel * 2**(i-1), 
                                            output_channel = n_features * output_channels_per_input_channel* 2**(i),
                                            convolution_kernel_dim=convolution_kernel_dim, 
                                            convolution_stride = convolution_stride, 
                                            activation_function = activation_function,
                                            batch_normalisation = batch_normalisation,
                                            pool_type = pool_type,
                                            pool_kernel = pool_kernel,
                                            andom_state=random_state,
                                            input_window_size = lookback_window_size))
            
            lookback_window_size = np.ceil(lookback_window_size/pool_kernel)

            if lookback_window_size == 1:
                print('WARNING: remaining n_values per channel == 1, consider reducing the number of hidden layers or increasing the initial lookback window. Further hidden layers with unchanging dimensions serve no benefit.')

        self.dense_n_hidden_layers = dense_n_hidden_layers
        self.batch_normalisation = batch_normalisation
        self.dense_layer_type = dense_layer_type

        self.layers = nn.ModuleList()

        # this is for non dilation only
        input_embed_dim = (2**(self.cnn_n_hidden_layers-1)) * \
                    (n_features * output_channels_per_input_channel) *  lookback_window_size

        actual_neuron_list = [input_embed_dim] + \
              [dense_hidden_layer_embed_dim for _ in range(self.dense_n_hidden_layers)] + \
                [output_size]

        if self.dense_layer_type == 'Dense':

            # define layers
            for i in range(dense_n_hidden_layers):
                self.layers.append(dense_layer(actual_neuron_list[i], actual_neuron_list[i+1], dropout_prob, batch_normalisation, activation_function, random_state))

            
        elif self.dense_layer_type == 'Residual': 

            # define layers
            self.input_layer = dense_layer(actual_neuron_list[0], actual_neuron_list[1], dropout_prob, batch_normalisation, activation_function, random_state)
            for i in range(dense_n_hidden_layers):

                if i == 0: # previously counted input layer as first layer, now get extra input layer to get hidden layer to right size before residual, so add make sure 1st layer in this loop has correct input size
                    self.layers.append(residual_layer(actual_neuron_list[i+1], dropout_prob, batch_normalisation, activation_function, random_state))
                else:
                    self.layers.append(residual_layer(actual_neuron_list[i], dropout_prob, batch_normalisation, activation_function, random_state))

        # final layers
        self.final_dense_layer = nn.Linear(actual_neuron_list[-2], actual_neuron_list[-1])


    def forward(self, x, training=True):
        
        # get vector of matrix into the right shape (each channel with each other)
        x = torch.tensor(x, dtype=torch.float32).transpose(1, 2)

        for i in range(self.cnn_n_hidden_layers):
            x = self.convolutional2dlayers[i](x)
        
        x = x.squeeze(-1).view(2, -1)

        if self.dense_layer_type == 'Residual': # only for dense neural network
            x = self.input_layer(x)
        
        for i in range(self.dense_n_hidden_layers):
            x = self.layers[i](x)
        
        out = self.final_dense_layer(x)

        return out