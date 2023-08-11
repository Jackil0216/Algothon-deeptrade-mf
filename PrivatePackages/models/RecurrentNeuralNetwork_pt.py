import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from torcheval.metrics import MeanSquaredError, R2Score, MulticlassAccuracy, MulticlassF1Score
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from torchmetrics import Metric


class Corrcoef(Metric):
    def __init__(self):
        super().__init__()

    def update(self, preds, target):
        self.preds, self.target = preds, target
        assert preds.shape == target.shape

    def compute(self):
        return torch.corrcoef(torch.stack([self.preds, self.target]).squeeze(-1))[0, 1]



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



class LSTMClassifier(nn.Module):

    def __init__(self,
            recurrent_hidden_layer_embed_dim,
            recurrent_n_hidden_layers,
            bidirectional,
            dense_n_hidden_layers,
            dense_hidden_layer_embed_dim,
            activation_function,
            dropout_prob,
            input_size,
            output_size,
            batch_normalisation,
            attention_num_heads,
            random_state,
            attention_embed_dim = 0,
            dense_layer_type = 'Dense'):

        super(LSTMClassifier, self).__init__()

        self.attention_num_heads = attention_num_heads

        torch.manual_seed(random_state)

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=recurrent_hidden_layer_embed_dim, 
                            num_layers=recurrent_n_hidden_layers, 
                            bidirectional=bidirectional, 
                            batch_first=True, 
                            dropout = dropout_prob)  

        if self.attention_num_heads:
            assert attention_embed_dim != 0, 'Attention Embedding Dimension cannot be 0'
            self.temporal_h_attention = nn.MultiheadAttention(embed_dim = (attention_embed_dim * 2 if bidirectional else attention_embed_dim),
                                                        num_heads = self.attention_num_heads,
                                                        dropout = dropout_prob,
                                                        batch_first = True)

        self.dense_n_hidden_layers = dense_n_hidden_layers
        self.batch_normalisation = batch_normalisation
        self.dense_layer_type = dense_layer_type

        self.layers = nn.ModuleList()

        input_embed_dim = recurrent_hidden_layer_embed_dim
        if bidirectional:
            input_embed_dim *= 2
        if self.attention_num_heads:
            input_embed_dim *= 2

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

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, training=True):

        x, (h, c) = self.lstm(x)

        if self.attention_num_heads:
            attention_output, _ = self.temporal_h_attention(x, x, x)
            x = torch.cat((x, attention_output), dim=-1)

        x = x[:, -1, :] # get last output of lstm

        if self.dense_layer_type == 'Residual': # only for dense neural network
            x = self.input_layer(x)
        
        for i in range(self.dense_n_hidden_layers):
            x = self.layers[i](x)
        
        out = self.final_dense_layer(x)

        softmax_out = self.softmax(out)

        return softmax_out





class LongShortTermMemoryClassifier_pt:

    def __init__(self,
                 recurrent_hidden_layer_embed_dim,
                 recurrent_n_hidden_layers,
                 bidirectional,
                 dense_n_hidden_layers,
                 batch_size,
                 learning_rate,
                 dense_hidden_layer_embed_dim,
                 activation,
                 num_epochs,
                 random_state,
                 dropout_prob,
                 batch_normalisation = False,
                 verbose = False,
                 loss_function='CrossEntropy',
                 data_loader = CustomDataLoader,
                 dense_layer_type = 'Dense',
                 grad_clip = False,
                 attention_num_heads = 0,
                 attention_embed_dim = 0,
                 eval_metric = 'Accuracy',
                 **kwargs):
        
        self.recurrent_hidden_layer_embed_dim = recurrent_hidden_layer_embed_dim
        self.recurrent_n_hidden_layers = recurrent_n_hidden_layers
        self.bidirectional = bidirectional
        self.dense_n_hidden_layers = dense_n_hidden_layers
        self.dense_hidden_layer_embed_dim = dense_hidden_layer_embed_dim
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.batch_normalisation = batch_normalisation
        self.verbose = verbose
        self.dense_layer_type = dense_layer_type
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.loss_function = loss_function
        self.data_loader = data_loader
        self.grad_clip = grad_clip
        self.attention_num_heads = attention_num_heads
        self.attention_embed_dim = attention_embed_dim
        self.eval_metric = eval_metric

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.LOSS_FUNCTIONS_MAP = {'CrossEntropy': nn.CrossEntropyLoss(),
                                    'KLDiv': nn.KLDivLoss(),
                                    'Hinge': nn.HingeEmbeddingLoss()}

        self.EVAL_METRICS_MAP = {'Accuracy': MulticlassAccuracy, 'F1': MulticlassF1Score}

        torch.manual_seed(self.random_state)



    def fit(self, train_x, train_y, initial_model = None, val_x = None, val_y = None):


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
        self.model = LSTMClassifier(recurrent_hidden_layer_embed_dim = self.recurrent_hidden_layer_embed_dim,
                            recurrent_n_hidden_layers = self.recurrent_n_hidden_layers,
                            bidirectional = self.bidirectional,
                            dense_n_hidden_layers = self.dense_n_hidden_layers,
                            dense_hidden_layer_embed_dim = self.dense_hidden_layer_embed_dim,
                            activation_function = self.activation,
                            dropout_prob = self.dropout_prob,
                            input_size = self.input_size,
                            output_size = self.num_labels,
                            batch_normalisation = self.batch_normalisation,
                            dense_layer_type = self.dense_layer_type,
                            attention_num_heads=self.attention_num_heads,
                            attention_embed_dim = self.recurrent_hidden_layer_embed_dim,
                            random_state = self.random_state
                            )

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

        if type(self.eval_metric) == str:
            eval_metric_function = self.EVAL_METRICS_MAP[self.eval_metric]
        else:
            eval_metric_function = self.eval_metric

        # Training loop
        for epoch in range(self.num_epochs):

            n_instance_observed = 0

            total_loss = 0

            if self.verbose:
                print('Epoch:', epoch+1)
                predictions = torch.tensor([]).to(self.device)
                labels = torch.tensor([]).to(self.device)

            for batch_idx, (batch_train_x, batch_train_y) in tqdm(enumerate(train_loader)):


                batch_train_x, batch_train_y = batch_train_x.to(self.device), batch_train_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_train_x)
                target = batch_train_y.view(-1, self.num_labels)  # Reshape target tensor to match the size of the output
                loss = self.criterion(outputs, target)

                total_loss += loss.item()*len(batch_train_x)

                n_instance_observed += len(batch_train_x)

                if self.verbose:
                    outputs_decoded = torch.tensor([np.argmax(outputs[i].detach()) for i in range(len(batch_train_x))]) # todo: nn.argmax
                    targets_decoded = torch.tensor([np.argmax(target[i].detach()) for i in range(len(batch_train_x))])

                    predictions = torch.cat([predictions, outputs_decoded])
                    labels = torch.cat([labels, targets_decoded])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

            # Print the progress
            if self.verbose:
 
                if type(self.eval_metric) == str:
                    metric = eval_metric_function().to(self.device)
                    metric.update(labels, predictions)
                    metric_value = metric.compute()
                else:
                    metric_value = self.eval_metric(labels.to(torch.device('cpu')), predictions.to(torch.device('cpu')))

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

                batch_predict_x = batch_predict_x.to(self.device)
 
                batch_prediction = self.model(batch_predict_x, training=False).to(torch.device('cpu'))
                batch_prediction_numpy = batch_prediction.numpy().reshape(-1, self.num_labels)
                predictions.extend(batch_prediction_numpy)
        
        predictions_decoded = [self.label_decoder[np.argmax(predictions[i])] for i in range(len(x))]

        return predictions_decoded
    

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

        one_hot_val_y = pd.DataFrame([np.eye(self.num_labels)[self.label_encoder[val_y[i]]] for i in range(len(val_y))])

        self.model.eval()

        # Create the custom datasets
        val_dataset = self.data_loader(val_x, one_hot_val_y)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        n_instance_observed = 0

        total_loss = 0

        with torch.no_grad():

            predictions = torch.tensor([]).to(self.device)
            labels = torch.tensor([]).to(self.device)

            for batch_idx, (batch_val_x, batch_val_y) in enumerate(val_loader):


                batch_val_x, batch_val_y = batch_val_x.to(self.device), batch_val_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_val_x)
                target = batch_val_y.view(-1, self.num_labels)  # Reshape target tensor to match the size of the output
                
                outputs_decoded = torch.tensor([np.argmax(outputs[i].detach()) for i in range(len(batch_val_x))])
                targets_decoded = torch.tensor([np.argmax(target[i].detach()) for i in range(len(batch_val_x))])

                predictions = torch.cat([predictions, outputs_decoded])
                labels = torch.cat([labels, targets_decoded])

                loss = self.criterion(outputs, target)

                total_loss += loss.item()*len(batch_val_x)

                n_instance_observed += len(batch_val_x)

        if type(eval_metric) == str:
            metric = eval_metric_function().to(self.device)
            metric.update(labels, predictions)
            metric_value = metric.compute()
        else:
            metric_value = eval_metric(labels.to(torch.device('cpu')), predictions.to(torch.device('cpu')))


        print(f'\t\tVal Avg Loss: {total_loss/n_instance_observed:.4f},', f'Val {eval_metric} (Metric)', np.round(float(metric_value), 6))


        self.model.train()



class LSTMRegressor(nn.Module):

    def __init__(self,
            recurrent_hidden_layer_embed_dim,
            recurrent_n_hidden_layers,
            bidirectional,
            dense_n_hidden_layers,
            dense_hidden_layer_embed_dim,
            activation_function,
            dropout_prob,
            input_size,
            output_size,
            batch_normalisation,
            attention_num_heads,
            random_state,
            attention_embed_dim = 0,
            dense_layer_type = 'Dense'):

        super(LSTMRegressor, self).__init__()

        torch.manual_seed(random_state)

        self.attention_num_heads = attention_num_heads

        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=recurrent_hidden_layer_embed_dim, 
                            num_layers=recurrent_n_hidden_layers, 
                            bidirectional=bidirectional, 
                            batch_first=True, 
                            dropout = dropout_prob)  

        if self.attention_num_heads:
            assert attention_embed_dim != 0, 'Attention Embedding Dimension cannot be 0'
            self.temporal_h_attention = nn.MultiheadAttention(embed_dim = (attention_embed_dim * 2 if bidirectional else attention_embed_dim),
                                                        num_heads = self.attention_num_heads,
                                                        dropout = dropout_prob,
                                                        batch_first = True)


        self.dense_n_hidden_layers = dense_n_hidden_layers
        self.batch_normalisation = batch_normalisation
        self.dense_layer_type = dense_layer_type

        self.layers = nn.ModuleList()

        input_embed_dim = recurrent_hidden_layer_embed_dim
        if bidirectional:
            input_embed_dim *= 2
        if self.attention_num_heads:
            input_embed_dim *= 2

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

        x, (h, c) = self.lstm(x)

        if self.attention_num_heads:
            attention_output, _ = self.temporal_h_attention(x, x, x)
            x = torch.cat((x, attention_output), dim=-1)

        x = x[:, -1, :] # get last output of lstm

        if self.dense_layer_type == 'Residual': # only for dense neural network
            x = self.input_layer(x)
        
        for i in range(self.dense_n_hidden_layers):
            x = self.layers[i](x)
        
        out = self.final_dense_layer(x)

        return out





class LongShortTermMemoryRegressor_pt:

    def __init__(self,
                 recurrent_hidden_layer_embed_dim,
                 recurrent_n_hidden_layers,
                 bidirectional,
                 dense_n_hidden_layers,
                 batch_size,
                 learning_rate,
                 dense_hidden_layer_embed_dim,
                 activation,
                 num_epochs,
                 random_state,
                 dropout_prob,
                 batch_normalisation = False,
                 verbose = False,
                 loss_function='MSE',
                 data_loader = CustomDataLoader,
                 dense_layer_type = 'Dense',
                 grad_clip = False,
                 attention_num_heads = 0,
                 attention_embed_dim = 0,
                 eval_metric = 'R2',
                 **kwargs):
        
        self.recurrent_hidden_layer_embed_dim = recurrent_hidden_layer_embed_dim
        self.recurrent_n_hidden_layers = recurrent_n_hidden_layers
        self.bidirectional = bidirectional
        self.dense_n_hidden_layers = dense_n_hidden_layers
        self.dense_hidden_layer_embed_dim = dense_hidden_layer_embed_dim
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.batch_normalisation = batch_normalisation
        self.verbose = verbose
        self.dense_layer_type = dense_layer_type
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.loss_function = loss_function
        self.data_loader = data_loader
        self.grad_clip = grad_clip
        self.attention_num_heads = attention_num_heads
        self.attention_embed_dim = attention_embed_dim
        self.eval_metric = eval_metric

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.LOSS_FUNCTIONS_MAP = {'MSE': nn.MSELoss(),
                                    'MAE': nn.L1Loss(),
                                    'Huber': nn.SmoothL1Loss()}


        self.EVAL_METRICS_MAP = {'R2': R2Score, 'MeanSquaredError': MeanSquaredError, 'Correlation': Corrcoef}

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
        self.model = LSTMRegressor(recurrent_hidden_layer_embed_dim = self.recurrent_hidden_layer_embed_dim,
                            recurrent_n_hidden_layers = self.recurrent_n_hidden_layers,
                            bidirectional = self.bidirectional,
                            dense_n_hidden_layers = self.dense_n_hidden_layers,
                            dense_hidden_layer_embed_dim = self.dense_hidden_layer_embed_dim,
                            activation_function = self.activation,
                            dropout_prob = self.dropout_prob,
                            input_size = self.input_size,
                            output_size = self.output_size,
                            batch_normalisation = self.batch_normalisation,
                            dense_layer_type = self.dense_layer_type,
                            attention_num_heads=self.attention_num_heads,
                            attention_embed_dim = self.recurrent_hidden_layer_embed_dim,
                            random_state = self.random_state)

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

            if self.verbose:
                predictions = torch.tensor([]).to(self.device)
                labels = torch.tensor([]).to(self.device)

            for batch_idx, (batch_train_x, batch_train_y) in tqdm(enumerate(train_loader)):


                batch_train_x, batch_train_y = batch_train_x.to(self.device), batch_train_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_train_x)
                target = batch_train_y.view(-1, 1)  # Reshape target tensor to match the size of the output
                loss = self.criterion(outputs, target)

                total_loss += loss.item()*len(batch_train_x)

                n_instance_observed += len(batch_train_x)

                if self.verbose:

                    predictions = torch.cat([predictions, outputs.detach()])
                    labels = torch.cat([labels, target.detach()])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

            # Print the progress
            if self.verbose:

                if type(self.eval_metric) == str:
                    metric = eval_metric_function().to(self.device)
                    metric.update(labels, predictions)
                    metric_value = metric.compute()
                else:
                    metric_value = self.eval_metric(labels.to(torch.device('cpu')), predictions.to(torch.device('cpu'))) # todo: easy for crashes when converting back to cpu

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

                batch_predict_x = batch_predict_x.to(self.device)
 
                batch_prediction = self.model(batch_predict_x, training=False).to(torch.device('cpu'))
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

            predictions = torch.tensor([]).to(self.device)
            labels = torch.tensor([]).to(self.device)

            for batch_idx, (batch_val_x, batch_val_y) in enumerate(val_loader):


                batch_val_x, batch_val_y = batch_val_x.to(self.device), batch_val_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_val_x)
                target = batch_val_y.view(-1, self.num_labels)  # Reshape target tensor to match the size of the output

                predictions = torch.cat([predictions, outputs.detach()])
                labels = torch.cat([labels, target.detach()])

                loss = self.criterion(outputs, target)

                total_loss += loss.item()*len(batch_val_x)

                n_instance_observed += len(batch_val_x)

        if type(eval_metric) == str:
            metric = eval_metric_function().to(self.device)
            metric.update(labels, predictions)
            metric_value = metric.compute()
        else:
            metric_value = eval_metric(labels.to(torch.device('cpu')), predictions.to(torch.device('cpu')))


        print(f'\t\tVal Avg Loss: {total_loss/n_instance_observed:.4f},', f'Val {eval_metric} (Metric)', np.round(float(metric_value), 6))


        self.model.train()