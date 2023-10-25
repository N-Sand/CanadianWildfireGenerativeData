import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

# credit/inspired by: https://avandekleut.github.io/vae/
# with some modifications particular to our problem

class Decoder(nn.Module):

    def __init__(self, latent_dims, outdim, hidden_width = 512):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dims, hidden_width)
        self.linear2 = nn.Linear(hidden_width, outdim)

    def forward(self, z):

        z = F.selu(self.linear1(z))
        z = torch.tanh(self.linear2(z))
        return z
    
class VariationalEncoder(nn.Module):

    def __init__(self, input_dim, latent_dims, width_1 = 512):
        super(VariationalEncoder, self).__init__()

        self.linear1 = nn.Linear(input_dim, width_1)
        self.linear2 = nn.Linear(width_1, latent_dims)
        self.linear3 = nn.Linear(width_1, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):

        x = torch.flatten(x, start_dim=1)
        x = F.selu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class VariationalAutoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_width_1 = 512, hidden_width_2 = 512):
        super(VariationalAutoencoder, self).__init__()

        self.encoder = VariationalEncoder(input_dim, latent_dim, hidden_width_1)
        self.decoder = Decoder(latent_dim, input_dim, hidden_width_2)

    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)