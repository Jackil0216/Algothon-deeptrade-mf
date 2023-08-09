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




class Convolutional2DLayer(nn.module):

    def __init__(self,
        input_channel,
        output_channel,
        convolution_kernel_dim,
        convolution_stride,
        activation_function,
        pool_type,
        pool_kernel, # TODO: Dilation?
        ):

        super().__init__()

        torch.manual_seed(random_state)

        self.ACTIVATION_FUNCTIONS_MAP = {'relu': nn.ReLU(),
                                    'sigmoid': nn.Sigmoid(),
                                    'tanh': nn.Tanh(),
                                    'softmax': nn.Softmax(dim=1)}

        self.POOL_MAP = {'MaxPool': nn.MaxPool, 'AvgPool': AvgPool}

        self.convolution_layer = nn.Conv2d(in_channels = in_channel, 
                                    out_channel = out_channel,
                                    kernel_size = convolution_kernel_dim,
                                    stride = convolution_stride)

        self.activation = self.ACTIVATION_FUNCTIONS_MAP[activation_function]

        self.pooling_layer = self.POOL_MAP[pool_type](kernel_size = pool_kernel, stride = pool_kernel)



    def forward(self, x):
        y = self.convolution_layer(x)
        y = self.activation(y)

        return self.pooling_layer(y)




class TS_CNNRegressor_AllFeatures(nn.Module):

    def __init__(self,
            cnn_n_layers,
            output_channels_per_convolutional_layer,
            convolution_kernel_dim,
            convolution_stride,
            activation_function,
            pool_type,
            pool_kernel,
            n_hidden_layers,
            dense_hidden_layer_embed_dim,
            activation_function,
            dropout_prob,
            output_size,
            batch_normalisation,
            random_state,
            dense_layer_type = 'Dense'):

        super(LSTMRegressor, self).__init__()

        torch.manual_seed(random_state)

        self.cnn_n_layers = cnn_n_layers

        self.convolutional2dlayers = nn.ModuleList()

        for i in range(self.cnn_n_layers): ##TODO: 算一下

            self.convolutional2dlayers.append(TS_CNNRegressor_AllFeatures(input_channel = 1 if i == 1 else output_channels_per_convolutional_layer*i, 
                        output_channel =output_channels_per_convolutional_layer,
                        convolution_kernel_dim=convolutional_kernel_dim, # TODO: 输入时count ndim
                        stride = convolution_stride, 
                        activation_function = activation_function,
                        pool_type = pool_type,
                        pool_kernel = pool_kernel))


        self.n_hidden_layers = n_hidden_layers
        self.batch_normalisation = batch_normalisation
        self.dense_layer_type = dense_layer_type

        self.layers = nn.ModuleList()

        input_embed_dim = recurrent_hidden_layer_embed_dim # 改
        if bidirectional:
            input_embed_dim *= 2
        if self.attention_num_heads:
            input_embed_dim *= 2

        actual_neuron_list = [input_embed_dim] + \
              [dense_hidden_layer_embed_dim for _ in range(self.n_hidden_layers)] + \
                [output_size]

        if self.dense_layer_type == 'Dense':

            # define layers
            for i in range(n_hidden_layers):
                self.layers.append(dense_layer(actual_neuron_list[i], actual_neuron_list[i+1], dropout_prob, batch_normalisation, activation_function, random_state))

            
        elif self.dense_layer_type == 'Residual': 

            # define layers
            self.input_layer = dense_layer(actual_neuron_list[0], actual_neuron_list[1], dropout_prob, batch_normalisation, activation_function, random_state)
            for i in range(n_hidden_layers):

                if i == 0: # previously counted input layer as first layer, now get extra input layer to get hidden layer to right size before residual, so add make sure 1st layer in this loop has correct input size
                    self.layers.append(residual_layer(actual_neuron_list[i+1], dropout_prob, batch_normalisation, activation_function, random_state))
                else:
                    self.layers.append(residual_layer(actual_neuron_list[i], dropout_prob, batch_normalisation, activation_function, random_state))

        # final layers
        self.final_dense_layer = nn.Linear(actual_neuron_list[-2], actual_neuron_list[-1])


    def forward(self, x, training=True):

        x, (h, c) = self.lstm(x)

        if self.attention_num_heads:
            attention_output, _ = self.temporal_h_attention(x, x, x)
            x = torch.cat((x, attention_output), dim=-1)

        x = x[:, -1, :] # get last output of lstm

        if self.dense_layer_type == 'Residual': # only for dense neural network
            x = self.input_layer(x)
        
        for i in range(self.n_hidden_layers):
            x = self.layers[i](x)
        
        out = self.final_dense_layer(x)

        return out