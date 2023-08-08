import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader





class CustomDataLoader(Dataset):
    def __init__(self, x, y = None):
        self.x = x
        if y is not None:
            self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = torch.tensor(self.x.iloc[idx].values, dtype=torch.float32)

        
        try: 
            if type(self.y) == pd.core.series.Series:
                y = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
                return x, y
            elif type(self.y) == pd.core.frame.DataFrame:
                y = torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)
                return x, y
        except:
            return x
        



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



class DenseNeuralNetworkRegressor(nn.Module):

    def __init__(self,
            n_hidden_layers,
            hidden_layer_n_neurons,
            activation_function,
            dropout_prob,
            input_size,
            output_size,
            batch_normalisation,
            random_state,
            dense_layer_type = 'Dense'):

        super(DenseNeuralNetworkRegressor, self).__init__()

        torch.manual_seed(random_state)

        self.n_hidden_layers = n_hidden_layers
        self.batch_normalisation = batch_normalisation
        self.dense_layer_type = dense_layer_type

        self.layers = nn.ModuleList()

        actual_neuron_list = [input_size] + hidden_layer_n_neurons + [output_size]

        if self.dense_layer_type == 'Dense':

            # define layers
            for i in range(n_hidden_layers):
                self.layers.append(dense_layer(actual_neuron_list[i], actual_neuron_list[i+1], dropout_prob, batch_normalisation, activation_function, random_state))

            
        
        elif self.dense_layer_type == 'Residual': # we will never use residual for shrinking networks

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


        if self.dense_layer_type == 'Residual': # only for dense neural network
            x = self.input_layer(x)
        
        for i in range(self.n_hidden_layers):
            x = self.layers[i](x)
        
        out = self.final_dense_layer(x)

        return out




class DenseNeuralNetworkClassifier(nn.Module):

    def __init__(self,
            n_hidden_layers,
            hidden_layer_n_neurons,
            activation_function,
            dropout_prob,
            input_size,
            output_size,
            batch_normalisation,
            random_state,
            dense_layer_type = 'Dense'):

        super(DenseNeuralNetworkClassifier, self).__init__()

        torch.manual_seed(random_state)

        self.n_hidden_layers = n_hidden_layers
        self.batch_normalisation = batch_normalisation
        self.dense_layer_type = dense_layer_type

        self.layers = nn.ModuleList()

        actual_neuron_list = [input_size] + hidden_layer_n_neurons + [output_size]

        if self.dense_layer_type == 'Dense':

            # define layers
            for i in range(n_hidden_layers):
                self.layers.append(dense_layer(actual_neuron_list[i], actual_neuron_list[i+1], dropout_prob, batch_normalisation, activation_function, random_state))

            
        
        elif self.dense_layer_type == 'Residual': # we will never use residual for shrinking networks

            # define layers
            self.input_layer = dense_layer(actual_neuron_list[0], actual_neuron_list[1], dropout_prob, batch_normalisation, activation_function, random_state)
            for i in range(n_hidden_layers):

                if i == 0: # previously counted input layer as first layer, now get extra input layer to get hidden layer to right size before residual, so add make sure 1st layer in this loop has correct input size
                    self.layers.append(residual_layer(actual_neuron_list[i+1], dropout_prob, batch_normalisation, activation_function, random_state))
                else:
                    self.layers.append(residual_layer(actual_neuron_list[i], dropout_prob, batch_normalisation, activation_function, random_state))

        # final layers
        self.final_dense_layer = nn.Linear(actual_neuron_list[-2], actual_neuron_list[-1])

        self.softmax = nn.Softmax()


    def forward(self, x, training=True):


        if self.dense_layer_type == 'Residual': # only for dense neural network
            x = self.input_layer(x)
        
        for i in range(self.n_hidden_layers):
            x = self.layers[i](x)
        
        out = self.final_dense_layer(x)

        softmax_out = self.softmax(out)

        return softmax_out





class DNNC_const_pt:

    def __init__(self,
                 n_hidden_layers,
                 activation,
                 lambda_lasso,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 random_state,
                 dropout_prob,
                 hidden_layer_n_neuron,
                 batch_normalisation = False,
                 verbose = False,
                 loss_function='MSE',
                 data_loader = CustomDataLoader,
                 grad_clip = False,
                 dense_layer_type = 'Dense',
                 **kwargs):

        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.lambda_lasso = lambda_lasso
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.hidden_layer_n_neuron = hidden_layer_n_neuron
        self.dropout_prob = dropout_prob
        self.verbose = verbose
        self.loss_function = loss_function
        self.batch_normalisation = batch_normalisation
        self.data_loader = data_loader
        self.dense_layer_type = dense_layer_type
        self.grad_clip = grad_clip

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.LOSS_FUNCTIONS_MAP = {'CrossEntropy': nn.CrossEntropyLoss(),
                                    'KLDiv': nn.KLDivLoss(),
                                    'Hinge': nn.HingeEmbeddingLoss()}

        torch.manual_seed(self.random_state) 


    def fit(self, train_x, train_y, initial_model = None):

        self.hidden_layer_sizes = [self.hidden_layer_n_neuron for i in range(self.n_hidden_layers)]

        # Currently only allow univariate labels

        labels = list(set(train_y))
        labels.sort()
        self.num_labels = len(labels)
        self.label_encoder = {labels[i]:i for i in range(self.num_labels)}
        self.label_decoder = {val:key for key, val in self.label_encoder.items()}

        one_hot_train_y = pd.DataFrame([np.eye(self.num_labels)[self.label_encoder[train_y[i]]] for i in range(len(train_y))])

        if type(train_x) == pd.core.frame.DataFrame:
            self.input_size = train_x.shape[1]
        else:
            self.input_size = train_x[0].shape[1]


        # Create the model
        self.model = DenseNeuralNetworkClassifier(self.n_hidden_layers,
                                        self.hidden_layer_sizes,
                                        self.activation,
                                        self.dropout_prob,
                                        self.input_size,
                                        self.num_labels,
                                        self.batch_normalisation,
                                        self.random_state,
                                        self.dense_layer_type)

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
        train_dataset = self.data_loader(train_x, one_hot_train_y)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.num_epochs):

            n_instance_observed = 0

            total_loss = 0

            for batch_idx, (batch_train_x, batch_train_y) in enumerate(train_loader):

                batch_train_x, batch_train_y = batch_train_x.to(self.device), batch_train_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_train_x)
                target = batch_train_y.view(-1, self.num_labels)  # Reshape target tensor to match the size of the output
                loss = self.criterion(outputs, target)

                # Lasso regularization
                l1_regularization = torch.tensor(0.).to(self.device)

                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, p=1)
                
                loss += self.lambda_lasso * l1_regularization

                total_loss += loss.item()*len(batch_train_x)

                n_instance_observed += len(batch_train_x)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

            # Print the progress
            if self.verbose:
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/n_instance_observed:.4f}')



    def predict(self, x):
    
        self.model.eval()

        predictions = []

        predict_dataset = self.data_loader(x)

        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():

            for batch_idx, batch_predict_x in enumerate(predict_loader):
 
                batch_prediction = self.model(batch_predict_x, training=False).cpu()
                batch_prediction_numpy = batch_prediction.numpy().reshape(-1, self.num_labels)
                predictions.extend(batch_prediction_numpy)
        
        predictions_decoded = [self.label_decoder[np.argmax(predictions[i])] for i in range(len(x))]

        return predictions_decoded
    


    def save(self, address):
        
        torch.save(self.model.state_dict(), f'{address}.pt')
    


    def load(self, address):

        self.model.load_state_dict(torch.load(f'{address}.pt'), map_location=self.device)




class DNNC_shrink_pt:

    def __init__(self,
                 n_hidden_layers,
                 activation,
                 lambda_lasso,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 random_state,
                 dropout_prob,
                 batch_normalisation = False,
                 verbose = False,
                 loss_function='MSE',
                 data_loader = CustomDataLoader,
                 grad_clip = False,
                 **kwargs):

        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.lambda_lasso = lambda_lasso
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.dropout_prob = dropout_prob
        self.verbose = verbose
        self.loss_function = loss_function
        self.batch_normalisation = batch_normalisation
        self.data_loader = data_loader
        self.grad_clip = grad_clip

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.LOSS_FUNCTIONS_MAP = {'CrossEntropy': nn.CrossEntropyLoss(),
                                    'KLDiv': nn.KLDivLoss(),
                                    'Hinge': nn.HingeEmbeddingLoss()}

        torch.manual_seed(self.random_state)



    def fit(self, train_x, train_y, initial_model=None):

        labels = list(set(train_y))
        labels.sort()
        self.num_labels = len(labels)
        self.label_encoder = {labels[i]:i for i in range(self.num_labels)}
        self.label_decoder = {val:key for key, val in self.label_encoder.items()}

        one_hot_train_y = pd.DataFrame([np.eye(self.num_labels)[self.label_encoder[train_y[i]]] for i in range(len(train_y))])

        if type(train_x) == pd.core.frame.DataFrame:
            self.input_size = train_x.shape[1]
        else:
            self.input_size = train_x[0].shape[1]

        gap = (self.input_size - self.num_labels)//(self.n_hidden_layers+1)
        self.hidden_layer_sizes = [self.input_size - i * gap for i in range(self.n_hidden_layers)]

        # Create the model
        self.model = DenseNeuralNetworkClassifier(self.n_hidden_layers,
                                        self.hidden_layer_sizes,
                                        self.activation,
                                        self.dropout_prob,
                                        self.input_size,
                                        self.num_labels,
                                        self.batch_normalisation,
                                        self.random_state,
                                        'Dense')

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
        train_dataset = self.data_loader(train_x, one_hot_train_y)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.num_epochs):

            n_instance_observed = 0

            total_loss = 0

            for batch_idx, (batch_train_x, batch_train_y) in enumerate(train_loader):

                batch_train_x, batch_train_y = batch_train_x.to(self.device), batch_train_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_train_x)
                target = batch_train_y.view(-1, self.num_labels)  # Reshape target tensor to match the size of the output
                loss = self.criterion(outputs, target)

                # Lasso regularization
                l1_regularization = torch.tensor(0.).to(self.device)

                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, p=1)
                loss += self.lambda_lasso * l1_regularization

                total_loss += loss.item()*len(batch_train_x)

                n_instance_observed += len(batch_train_x)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

            # Print the progress
            if self.verbose:
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/n_instance_observed:.4f}')



    def predict(self, x):
    
        self.model.eval()

        predictions = []

        predict_dataset = self.data_loader(x)

        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():

            for batch_idx, batch_predict_x in enumerate(predict_loader):
 
                batch_prediction = self.model(batch_predict_x, training=False).cpu()
                batch_prediction_numpy = batch_prediction.numpy().reshape(-1, self.num_labels)
                predictions.extend(batch_prediction_numpy)
        
        predictions_decoded = [self.label_decoder[np.argmax(predictions[i])] for i in range(len(x))]

        return predictions_decoded
    


    def save(self, address):
        
        torch.save(self.model.state_dict(), f'{address}.pt')
    


    def load(self, address):

        self.model.load_state_dict(torch.load(f'{address}.pt'), map_location=self.device)





class DNNR_const_pt:

    def __init__(self,
                 n_hidden_layers,
                 activation,
                 lambda_lasso,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 random_state,
                 dropout_prob,
                 hidden_layer_n_neuron,
                 batch_normalisation = False,
                 verbose = False,
                 loss_function='MSE',
                 data_loader = CustomDataLoader,
                 grad_clip = False,
                 dense_layer_type = 'Dense',
                 **kwargs):

        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.lambda_lasso = lambda_lasso
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.hidden_layer_n_neuron = hidden_layer_n_neuron
        self.dropout_prob = dropout_prob
        self.verbose = verbose
        self.loss_function = loss_function
        self.batch_normalisation = batch_normalisation
        self.data_loader = data_loader
        self.dense_layer_type = dense_layer_type
        self.grad_clip = grad_clip

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.LOSS_FUNCTIONS_MAP = {'MSE': nn.MSELoss(),
                                    'MAE': nn.L1Loss(),
                                    'Huber': nn.SmoothL1Loss()}

        torch.manual_seed(self.random_state) 


    def fit(self, train_x, train_y, initial_model = None):

        self.hidden_layer_sizes = [self.hidden_layer_n_neuron for i in range(self.n_hidden_layers)]

        if type(train_y) == pd.core.frame.DataFrame:
            self.output_size = train_y.shape[1]
        else:
            self.output_size = 1

        if type(train_x) == pd.core.frame.DataFrame:
            self.input_size = train_x.shape[1]
        else:
            self.input_size = train_x[0].shape[1]


        # Create the model
        self.model = DenseNeuralNetworkRegressor(self.n_hidden_layers,
                                        self.hidden_layer_sizes,
                                        self.activation,
                                        self.dropout_prob,
                                        self.input_size,
                                        self.output_size,
                                        self.batch_normalisation,
                                        self.random_state,
                                        self.dense_layer_type)

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

        # Training loop
        for epoch in range(self.num_epochs):

            n_instance_observed = 0

            total_loss = 0

            for batch_idx, (batch_train_x, batch_train_y) in enumerate(train_loader):


                batch_train_x, batch_train_y = batch_train_x.to(self.device), batch_train_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_train_x)
                target = batch_train_y.view(-1, 1)  # Reshape target tensor to match the size of the output
                loss = self.criterion(outputs, target)

                # Lasso regularization
                l1_regularization = torch.tensor(0.).to(self.device)

                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, p=1)
                
                loss += self.lambda_lasso * l1_regularization

                total_loss += loss.item()*len(batch_train_x)

                n_instance_observed += len(batch_train_x)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

            # Print the progress
            if self.verbose:
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/n_instance_observed:.4f}')



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





class DNNR_shrink_pt:

    def __init__(self,
                 n_hidden_layers,
                 activation,
                 lambda_lasso,
                 batch_size,
                 learning_rate,
                 num_epochs,
                 random_state,
                 dropout_prob,
                 batch_normalisation = False,
                 verbose = False,
                 loss_function='MSE',
                 data_loader = CustomDataLoader,
                 grad_clip = False,
                 **kwargs):

        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.lambda_lasso = lambda_lasso
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.dropout_prob = dropout_prob
        self.verbose = verbose
        self.loss_function = loss_function
        self.batch_normalisation = batch_normalisation
        self.data_loader = data_loader
        self.grad_clip = grad_clip

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.LOSS_FUNCTIONS_MAP = {'MSE': nn.MSELoss(),
                                    'MAE': nn.L1Loss(),
                                    'Huber': nn.SmoothL1Loss()}

        torch.manual_seed(self.random_state)



    def fit(self, train_x, train_y, initial_model=None):

        if type(train_y) == pd.core.frame.DataFrame:
            self.output_size = train_y.shape[1]
        else:
            self.output_size = 1

        if type(train_x) == pd.core.frame.DataFrame:
            self.input_size = train_x.shape[1]
        else:
            self.input_size = train_x[0].shape[1]

        gap = (self.input_size - self.output_size)//(self.n_hidden_layers+1)
        self.hidden_layer_sizes = [self.input_size - i * gap for i in range(self.n_hidden_layers)]

        # Create the model
        self.model = DenseNeuralNetworkRegressor(self.n_hidden_layers,
                                        self.hidden_layer_sizes,
                                        self.activation,
                                        self.dropout_prob,
                                        self.input_size,
                                        self.output_size,
                                        self.batch_normalisation,
                                        self.random_state,
                                        'Dense')

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

        # Training loop
        for epoch in range(self.num_epochs):

            n_instance_observed = 0

            total_loss = 0

            for batch_idx, (batch_train_x, batch_train_y) in enumerate(train_loader):

                batch_train_x, batch_train_y = batch_train_x.to(self.device), batch_train_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_train_x)
                target = batch_train_y.view(-1, 1)  # Reshape target tensor to match the size of the output
                loss = self.criterion(outputs, target)

                # Lasso regularization
                l1_regularization = torch.tensor(0.).to(self.device)

                for param in self.model.parameters():
                    l1_regularization += torch.norm(param, p=1)
                loss += self.lambda_lasso * l1_regularization

                total_loss += loss.item()*len(batch_train_x)

                n_instance_observed += len(batch_train_x)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

            # Print the progress
            if self.verbose:
                if (epoch + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/n_instance_observed:.4f}')



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