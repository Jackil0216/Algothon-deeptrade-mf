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




class Convolutional1DLayer(nn.Module):

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
                                    out_channels = output_channel,
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




class T_CNNR(nn.Module):

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

        super(T_CNNR, self).__init__()

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
                                            random_state=random_state,
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
    



class TemporalConvolutionalNeuralNetworkRegressor_pt:

    def __init__(self,
                 lookback_window_size,
                 cnn_n_hidden_layers,
                 output_channels_per_input_channel,
                 convolution_kernel_dim,
                 convolution_stride,
                 activation,
                 pool_type,
                 pool_kernel,
                 dense_n_hidden_layers,
                 dense_hidden_layer_embed_dim,
                 dropout_prob,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 random_state,
                 batch_normalisation = False,
                 verbose = False,
                 loss_function='MSE',
                 data_loader = CustomDataLoader,
                 dense_layer_type = 'Dense',
                 grad_clip = False,
                 eval_metric = 'R2',
                 **kwargs):
        
        self.lookback_window_size = lookback_window_size
        self.cnn_n_hidden_layers = cnn_n_hidden_layers
        self.output_channels_per_input_channel = output_channels_per_input_channel
        self.convolution_kernel_dim = convolution_kernel_dim
        self.convolution_stride = convolution_stride
        self.activation = activation
        self.pool_type = pool_type
        self.pool_kernel = pool_kernel
        self.dense_n_hidden_layers = dense_n_hidden_layers
        self.dense_hidden_layer_embed_dim = dense_hidden_layer_embed_dim
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.batch_normalisation = batch_normalisation
        self.verbose = verbose
        self.loss_function = loss_function
        self.dense_layer_type = dense_layer_type
        self.data_loader = data_loader
        self.grad_clip = grad_clip
        self.eval_metric = eval_metric

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.LOSS_FUNCTIONS_MAP = {'MSE': nn.MSELoss(),
                                    'MAE': nn.L1Loss(),
                                    'Huber': nn.SmoothL1Loss()}

        self.EVAL_METRICS_MAP = {'R2': R2Score, 'MeanSquaredError': MeanSquaredError}

        torch.manual_seed(self.random_state)



    def fit(self, train_x, train_y, initial_model = None, val_x = None, val_y = None):


        if type(train_y) == pd.core.frame.DataFrame:
            self.output_size = train_y.shape[1]
        else:
            self.output_size = 1

        if type(train_x) == pd.core.frame.DataFrame:
            self.input_size = train_x.shape[1]
        else:
            self.input_size = train_x[0].shape[1]


        # Create the model
        self.model = T_CNNR(n_features=self.input_size,
                    lookback_window_size = self.lookback_window_size,
                    cnn_n_hidden_layers = self.cnn_n_hidden_layers,
                    output_channels_per_input_channel = self.output_channels_per_input_channel,
                    convolution_kernel_dim = self.convolution_kernel_dim,
                    convolution_stride = self.convolution_stride,
                    activation_function = self.activation,
                    pool_type =   self.pool_type,
                    pool_kernel =   self.pool_kernel,
                    dense_n_hidden_layers = self.dense_n_hidden_layers,
                    dense_hidden_layer_embed_dim = self.dense_hidden_layer_embed_dim,
                    dropout_prob = self.dropout_prob,
                    output_size = self.output_size,
                    batch_normalisation = self.batch_normalisation,
                    random_state = self.random_state,
                    dense_layer_type = self.dense_layer_type)

        if initial_model is not None:
            self.model.load_state_dict(initial_model.model.state_dict())

        self.model.to(self.device)

        self.model.train()

        # Define the loss function and optimizer
        if type(self.loss_function) == str:
            self.criterion = self.LOSS_FUNCTIONS_MAP[self.loss_function]
        else:
            self.criterion = self.loss_function

        self.criterion.to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create the custom datasets
        train_dataset = self.data_loader(train_x, train_y)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if type(self.eval_metric) == str:
            eval_metric_function = self.EVAL_METRICS_MAP[self.eval_metric]
        else:
            eval_metric_function = self.eval_metric


        # Training loop
        for epoch in range(self.num_epochs):

            n_instance_observed = 0

            total_loss = 0

            if (epoch+1)%100 == 0:
                predictions = torch.tensor([])
                labels = torch.tensor([])

            for batch_idx, (batch_train_x, batch_train_y) in enumerate(train_loader):


                batch_train_x, batch_train_y = batch_train_x.to(self.device), batch_train_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_train_x)
                target = batch_train_y.view(-1, 1)  # Reshape target tensor to match the size of the output
                loss = self.criterion(outputs, target)

                total_loss += loss.item()*len(batch_train_x)

                n_instance_observed += len(batch_train_x)

                if (epoch+1)%100 == 0:

                    predictions = torch.cat([predictions, outputs.to(torch.device('cpu'))])
                    labels = torch.cat([labels, target.to(torch.device('cpu'))])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

            # Print the progress
            if self.verbose:

                if (epoch + 1) % 100 == 0:

                    predictions = predictions.detach().numpy()
                    labels = labels.detach().numpy()

                    if type(self.eval_metric) == str:
                        metric = eval_metric_function()
                        metric.update(labels, predictions)
                        metric_value = metric.compute()
                    else:
                        metric_value = self.eval_metric(labels, predictions)

                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Avg Loss: {total_loss/n_instance_observed:.4f}', f'Train {self.eval_metric} (Metric)', np.round(float(metric_value), 6))

                    if val_x is not None and val_y is not None:
                        self.eval(val_x, val_y, self.eval_metric)



    def predict(self, x):

        self.model.eval()

        predictions = []

        predict_dataset = self.data_loader(x)

        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():

            for batch_idx, batch_predict_x in enumerate(predict_loader):
 
                batch_prediction = self.model(batch_predict_x, training=False).cpu()
                batch_prediction_numpy = batch_prediction.numpy().flatten()
                predictions.extend(batch_prediction_numpy)

        return predictions
    

    def save(self, address):
        
        torch.save(self.model.state_dict(), f'{address}.pt')
    

    def load(self, address):

        self.model.load_state_dict(torch.load(f'{address}.pt'), map_location=self.device)


    
    def eval(self, val_x, val_y, eval_metric = None):

        eval_metric = None or self.eval_metric
        
        # Define the loss function and optimizer
        if type(eval_metric) == str:
            eval_metric_function = self.EVAL_METRICS_MAP[eval_metric]
        else:
            eval_metric_function = eval_metric

        self.model.eval()

        # Create the custom datasets
        val_dataset = self.data_loader(val_x, val_y)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        n_instance_observed = 0

        total_loss = 0

        with torch.no_grad():

            predictions = torch.tensor([])
            labels = torch.tensor([])

            for batch_idx, (batch_val_x, batch_val_y) in enumerate(val_loader):


                batch_val_x, batch_val_y = batch_val_x.to(self.device), batch_val_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_val_x)
                target = batch_val_y.view(-1, self.num_labels)  # Reshape target tensor to match the size of the output

                predictions = torch.cat([predictions, outputs.to(torch.device('cpu'))])
                labels = torch.cat([labels, target.to(torch.device('cpu'))])

                loss = self.criterion(outputs, target)

                total_loss += loss.item()*len(batch_val_x)

                n_instance_observed += len(batch_val_x)

        predictions = predictions.detach().numpy()
        labels = labels.detach().numpy()

        if type(eval_metric) == str:
            metric = eval_metric_function()
            metric.update(labels, predictions)
            metric_value = metric.compute()
        else:
            metric_value = eval_metric(labels, predictions)


        print(f'\t\tVal Avg Loss: {total_loss/n_instance_observed:.4f},', f'Val {eval_metric} (Metric)', np.round(float(metric_value), 6))


        self.model.train()