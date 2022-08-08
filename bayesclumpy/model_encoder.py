import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, hyperparameters):
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters
        
        self.input_size = hyperparameters['input_size']
        self.hidden_size = hyperparameters['hidden_size']
        self.n_hidden_layers = hyperparameters['n_hidden_layers']
        self.output_size = hyperparameters['output_size']
        self.activation = hyperparameters['activation']

        if (self.activation == 'relu'):
            act = nn.ReLU()
        if (self.activation == 'elu'):
            act = nn.ELU()
        if (self.activation == 'leakyrelu'):
            act = nn.LeakyReLU(0.2)

        # Define the layers of the network. We use a ModuleList here
        self.layers = nn.ModuleList([])
        
        # Start appending all layers
        # 1D Convolutional Auto-Encoder

        """
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1), # 1D 16*kernels of length=36
            act,
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 1D 32*kernels of length=18
            act,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 1D 64*kernels of length=9
            act,
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=9, stride=1), # 1D 32*kernels of length=1
            act,
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, stride=1) # NiN 16 channels of 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
            act,
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=9, stride=1),
            act,
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            act,
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            act,
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1)
        )
        """
               
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1), # 1D 16*kernels of length=53
            act,
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 1D 32*kernels of length=27
            act,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 1D 64*kernels of length=14
            act,
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=14, stride=1), # 1D 32*kernels of length=1
            act,
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1) # NiN 16 channels of 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            act,
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=14, stride=1), 
            act,
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            act,
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),
            act,
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1)
        )
        

        #for i in range(self.n_hidden_layers):
        #    self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        #    self.layers.append(act)

        #self.layers.append(nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        """
        Evaluate the network

        Parameters
        ----------
        x : tensor
            Input tensor

        Returns
        -------
        tensor
            Output tensor
        """       
        """
        
        for layer in self.layers:
            x = layer(x)

        return x
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        