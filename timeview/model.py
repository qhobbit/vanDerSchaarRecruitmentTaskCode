import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from timeview.basis import BSplineBasis
from .config import Config

def is_dynamic_bias_enabled(config):
    if hasattr(config, 'dynamic_bias'):
        return config.dynamic_bias
    else:
        return False

class Encoder(torch.nn.Module):

    def __init__(self,config):
        """
        Args:
            config: an instance of the Config class
        """
        super().__init__()
        self.config = config
        self.n_features = config.n_features
        self.n_basis = config.n_basis
        self.hidden_sizes =  config.encoder.hidden_sizes
        self.dropout_p = config.encoder.dropout_p

        assert len(self.hidden_sizes) > 0

        self.layers = []
        activation = torch.nn.ReLU()

        self.layers.append(torch.nn.Linear(self.n_features,self.hidden_sizes[0]))
        self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[0]))
        self.layers.append(activation)
        self.layers.append(torch.nn.Dropout(self.dropout_p))

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(torch.nn.Linear(self.hidden_sizes[i],self.hidden_sizes[i+1]))
            self.layers.append(torch.nn.BatchNorm1d(self.hidden_sizes[i+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(self.dropout_p))
        
        latent_size = self.n_basis

        if is_dynamic_bias_enabled(config):
            latent_size += 1
        
        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1],latent_size))

        self.nn = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn(x)
    
class DynamicEncoder(torch.nn.Module):

    def __init__(self, config):
        """
        Args:
            config: an instance of the Config class containing the encoder configuration
        """
        super().__init__()
        self.config = config
        self.n_dynamic_features = config.n_features_dynamic * 1 
        self.hidden_size = config.dynamic_encoder.hidden_size  
        self.rnn_layers = config.dynamic_encoder.rnn_layers  
        self.dropout_p = config.dynamic_encoder.dropout_p  
        self.rnn_type = config.dynamic_encoder.rnn_type  

        # self.position_embedding = nn.Embedding(60, self.n_dynamic_features)

        # Choose RNN type (LSTM or GRU)
        if self.rnn_type == 'lstm':
            self.rnn = torch.nn.LSTM(
                input_size=self.n_dynamic_features,  
                hidden_size=self.hidden_size,  
                num_layers=self.rnn_layers,  
                batch_first=True,  
                dropout=self.dropout_p if self.rnn_layers > 1 else 0.0 
            )
        elif self.rnn_type == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=self.n_dynamic_features,
                hidden_size=self.hidden_size,
                num_layers=self.rnn_layers,
                batch_first=True,
                dropout=self.dropout_p if self.rnn_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")


        latent_size = self.config.n_basis  

        if is_dynamic_bias_enabled(config):
            latent_size += 1 


        self.fc = torch.nn.Linear(self.hidden_size, latent_size)

        self.batch_norm = torch.nn.BatchNorm1d(latent_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout_p)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, n_dynamic_features)
        Returns:
            Latent representation of dynamic features
        """
        # Pass the input through the RNN (LSTM or GRU)
        x = x.permute(0,2,1)
        # batch_size, time_steps, _ = x.size()
        # positions = torch.arange(0, time_steps, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        # position_embeds = self.position_embedding(positions)
        # x = torch.cat((x, position_embeds), dim=2) 
        # print(x.shape)
        # batch_size, n_dynamic_features, n_intervals, n_derivatives = x.shape
        # x = x.view(batch_size, n_intervals, n_dynamic_features * n_derivatives) 
        rnn_output, (h_n, c_n) = self.rnn(x) if self.rnn_type == 'lstm' else self.rnn(x)

        last_hidden_state = h_n[-1] 

        latent = self.fc(last_hidden_state)

        latent = self.batch_norm(latent)
        latent = self.activation(latent)
        latent = self.dropout(latent)

        return latent


class TTS(torch.nn.Module):

    def __init__(self,config):
        """
        Args:
            config: an instance of the Config class
        """
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.encoder = Encoder(self.config)
        if not is_dynamic_bias_enabled(self.config):
            self.bias = torch.nn.Parameter(torch.zeros(1))
        
    
    def forward(self, X, Phis):
        """
        Args:
            X: a tensor of shape (D,M) where D is the number of sample and M is the number of static features
            Phi:
                if dataloader_type = 'tensor': a tensor of shape (D,N_max,B) where D is the number of sample, N_max is the maximum number of time steps and B is the number of basis functions
                if dataloader_type = 'iterative': a list of D tensors of shape (N_d,B) where N_d is the number of time steps and B is the number of basis functions
        """
        h = self.encoder(X)
        if is_dynamic_bias_enabled(self.config):
            self.bias = h[:,-1]
            h = h[:,:-1]
        
        if self.config.dataloader_type == "iterative":
            if is_dynamic_bias_enabled(self.config):
                return [torch.matmul(Phi,h[d,:]) + self.bias[d] for d, Phi in enumerate(Phis)]
            else:
                return [torch.matmul(Phi,h[d,:]) + self.bias for d, Phi in enumerate(Phis)]
        elif self.config.dataloader_type == "tensor":
            if is_dynamic_bias_enabled(self.config):
                return torch.matmul(Phis,torch.unsqueeze(h,-1)).squeeze(-1) + torch.unsqueeze(self.bias,-1)
            else:
                return torch.matmul(Phis,torch.unsqueeze(h,-1)).squeeze(-1) + self.bias
        
    def predict_latent_variables(self,X):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of sample and M is the number of static features
        Returns:
            a numpy array of shape (D,B) where D is the number of sample and B is the number of basis functions
        """
        device = self.encoder.layers[0].bias.device
        X = torch.from_numpy(X).float().to(device)
        self.encoder.eval()
        if is_dynamic_bias_enabled(self.config):
            with torch.no_grad():
                return self.encoder(X)[:,:-1].cpu().numpy()
        else:
            with torch.no_grad():
                return self.encoder(X).cpu().numpy()        

    def forecast_trajectory(self,x,t):
        """
        Args:
            x: a numpy array of shape (M,) where M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (N,) where N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        x = torch.unsqueeze(torch.from_numpy(x),0).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0,self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(x)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[0,-1]
                h = h[:,:-1]
            return (torch.matmul(Phi,h[0,:]) + self.bias).cpu().numpy()

    def forecast_trajectories(self,X,t):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of sample and M is the number of static features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (D,N) where D is the number of sample and N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        X = torch.from_numpy(X).float().to(device)
        bspline = BSplineBasis(self.config.n_basis, (0,self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device) # shape (N,B)
        self.encoder.eval()
        with torch.no_grad():
            h = self.encoder(X) # shape (D,B)
            if is_dynamic_bias_enabled(self.config):
                self.bias = h[:,-1]
                h = h[:,:-1]
            return (torch.matmul(h,Phi.T)+self.bias).cpu().numpy() # shape (D,N), broadcasting will take care of the bias
        
class TTSDynamic(torch.nn.Module):

    def __init__(self,config):
        """
        Args:
            config: an instance of the Config class
        """
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.encoder = Encoder(self.config)
        self.dynamic_encoder = DynamicEncoder(self.config)
        if not is_dynamic_bias_enabled(self.config):
            self.bias = torch.nn.Parameter(torch.zeros(1))
        
    
    def forward(self, X, X_dynamic, Phis, contrastive=True):
        """
        Args:
            X: a tensor of shape (D,M) where D is the number of sample and M is the number of static features
            Phi:
                if dataloader_type = 'tensor': a tensor of shape (D,N_max,B) where D is the number of sample, N_max is the maximum number of time steps and B is the number of basis functions
                if dataloader_type = 'iterative': a list of D tensors of shape (N_d,B) where N_d is the number of time steps and B is the number of basis functions
        """
        h = self.encoder(X)
        h_dynamic = self.dynamic_encoder(X_dynamic)

        h_combined = h + h_dynamic

        if is_dynamic_bias_enabled(self.config):
            self.bias = h_combined[:,-1]
            h_combined = h_combined[:,:-1]

        if contrastive:
            contrastive_loss = self.compute_contrastive_loss(h, h_dynamic) + self.compute_kl_divergence_loss(h, h_dynamic)
        
        if self.config.dataloader_type == "iterative":
            if is_dynamic_bias_enabled(self.config):
                output = [torch.matmul(Phi,h_combined[d,:]) + self.bias[d] for d, Phi in enumerate(Phis)]
            else:
                output = [torch.matmul(Phi,h_combined[d,:]) + self.bias for d, Phi in enumerate(Phis)]
        elif self.config.dataloader_type == "tensor":
            if is_dynamic_bias_enabled(self.config):
                output = torch.matmul(Phis,torch.unsqueeze(h_combined,-1)).squeeze(-1) + torch.unsqueeze(self.bias,-1)
            else:
                output = torch.matmul(Phis,torch.unsqueeze(h_combined,-1)).squeeze(-1) + self.bias
        
        if contrastive:
            return output, contrastive_loss
        else:
            return output
            
    def compute_contrastive_loss(self, h_static, h_dynamic, temperature=0.07):

        h_static_norm = torch.nn.functional.normalize(h_static, dim=-1)
        h_dynamic_norm = torch.nn.functional.normalize(h_dynamic, dim=-1)

        similarity_matrix = torch.matmul(h_static_norm, h_dynamic_norm.T) / temperature

        batch_size = h_static.size(0)
        labels = torch.arange(batch_size).to(similarity_matrix.device)

        contrastive_loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)

        return contrastive_loss
    
    def compute_kl_divergence_loss(self, h_static, h_dynamic):

        h_static_norm = torch.nn.functional.normalize(h_static, dim=-1)
        h_dynamic_norm = torch.nn.functional.normalize(h_dynamic, dim=-1)

        p_static = torch.nn.functional.softmax(h_static_norm, dim=-1)
        p_dynamic = torch.nn.functional.softmax(h_dynamic_norm, dim=-1)

        kl_div = torch.nn.functional.kl_div(p_static.log(), p_dynamic, reduction='batchmean')

        return kl_div
        
    def predict_latent_variables(self, X, X_dynamic=None, return_aligned=False):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of samples and M is the number of static features
            X_dynamic: (optional) a numpy array of shape (D, time_steps, n_dynamic_features) for dynamic features
            return_aligned: Boolean flag to return the aligned latent vector (from static + dynamic features)
        Returns:
            A numpy array of shape (D,B) where D is the number of samples and B is the number of basis functions (latent dimensions)
        """
        device = self.encoder.layers[0].bias.device
        X = torch.from_numpy(X).float().to(device)
        
        # Set the encoder to evaluation mode
        self.encoder.eval()

        # Compute static latent variables
        with torch.no_grad():
            h_static = self.encoder(X)

            if is_dynamic_bias_enabled(self.config):
                h_static = h_static[:, :-1]  # Exclude the bias if enabled

        # If no dynamic input is provided or not requesting aligned, return static latent variables
        if X_dynamic is None or not return_aligned:
            return h_static.cpu().numpy()

        # If dynamic input is provided and return_aligned=True, compute the aligned latent vector
        X_dynamic = torch.from_numpy(X_dynamic).float().to(device)
        self.dynamic_encoder.eval()

        with torch.no_grad():
            h_dynamic = self.dynamic_encoder(X_dynamic)  # Compute dynamic latent variables

            # Create aligned vector from static and dynamic representations
            aligned_vector = h_static + h_dynamic

            return aligned_vector.cpu().numpy()   

    def forecast_trajectory(self, x, x_dynamic, t):
        """
        Args:
            x: a numpy array of shape (M,) where M is the number of static features
            x_dynamic: a numpy array of dynamic features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (N,) where N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        
        # Convert static features (x) to a torch tensor
        x = torch.unsqueeze(torch.from_numpy(x), 0).float().to(device)
        
        # Convert dynamic features (x_dynamic) to a torch tensor
        x_dynamic = torch.unsqueeze(torch.from_numpy(x_dynamic), 0).float().to(device)
        
        # Create the B-spline basis matrix for time steps t
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)
        
        # Set both encoders to evaluation mode
        self.encoder.eval()
        self.dynamic_encoder.eval()
        
        with torch.no_grad():
            # Compute static latent variables
            h_static = self.encoder(x)
            
            # Compute dynamic latent variables
            h_dynamic = self.dynamic_encoder(x_dynamic)
            
            # Combine static and dynamic latent vectors
            h_combined = h_static + h_dynamic
            
            # Handle bias if dynamic bias is enabled
            if is_dynamic_bias_enabled(self.config):
                self.bias = h_combined[0, -1]  # Extract the bias
                h_combined = h_combined[:, :-1]  # Remove bias from latent vector
                
            # Forecast the trajectory using the combined latent vector
            return (torch.matmul(Phi, h_combined[0, :]) + self.bias).cpu().numpy()


    def forecast_trajectories(self, X, X_dynamic, t):
        """
        Args:
            X: a numpy array of shape (D,M) where D is the number of samples and M is the number of static features
            X_dynamic: a numpy array of shape (D, time_steps, n_dynamic_features) for dynamic features
            t: a numpy array of shape (N,) where N is the number of time steps
        Returns:
            a numpy array of shape (D,N) where D is the number of samples and N is the number of time steps
        """
        device = self.encoder.layers[0].bias.device
        
        # Convert static features (X) and dynamic features (X_dynamic) to torch tensors
        X = torch.from_numpy(X).float().to(device)
        X_dynamic = torch.from_numpy(X_dynamic).float().to(device)
        
        # Create the B-spline basis matrix for time steps t
        bspline = BSplineBasis(self.config.n_basis, (0, self.config.T), internal_knots=self.config.internal_knots)
        Phi = torch.from_numpy(bspline.get_matrix(t)).float().to(device)  # Shape: (N, B)

        # Set both encoders to evaluation mode
        self.encoder.eval()
        self.dynamic_encoder.eval()
        
        with torch.no_grad():
            # Compute static latent variables for all samples
            h_static = self.encoder(X)  # Shape: (D, latent_size)

            # Compute dynamic latent variables for all samples
            h_dynamic = self.dynamic_encoder(X_dynamic)  # Shape: (D, latent_size)

            # Combine static and dynamic latent vectors
            h_combined = h_static + h_dynamic  # Shape: (D, latent_size)

            # Handle dynamic bias if enabled
            if is_dynamic_bias_enabled(self.config):
                self.bias = h_combined[:, -1]  # Extract the bias from the last dimension
                h_combined = h_combined[:, :-1]  # Remove the bias from latent vectors
            
            # Forecast trajectories for all samples using the combined latent vectors
            return (torch.matmul(h_combined, Phi.T) + self.bias).cpu().numpy()  # Shape: (D, N)
