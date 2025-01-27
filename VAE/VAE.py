import torch
import torch.nn as nn

def calculate_bottleneck_size(input_size, encoder, input_channels, device):
    dum = torch.randn(32, input_channels, input_size, input_size).to(device)

    for layer in encoder:
        dum = layer(dum)
    return dum.shape[1]*dum.shape[2]*dum.shape[3]



class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=2, padding=1, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=padding),
            
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
      #      nn.Dropout2d(p=dropout)
        )
        nn.init.xavier_uniform_(self.conv[0].weight)
        nn.init.zeros_(self.conv[0].bias)
    
    def forward(self, x):
        return self.conv(x)

class DeConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=2, padding=1, final_layer=False, dropout=0.2):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
     #       nn.Dropout2d(p=dropout)
        ]
        if final_layer:
            layers = [
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride ,padding=padding),
                nn.Tanh()
            ]
        self.conv = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.conv[0].weight)
        nn.init.zeros_(self.conv[0].bias)

    def forward(self, x):
        return self.conv(x)



class VAE(nn.Module):

    def __init__(self, input_size, in_channel, latent_space_dim, device, features=[16, 32, 64, 128, 256], padding=1, kernel_size = 4, stride=2):
        
        super().__init__()
        H=input_size
        W = input_size
        self.device = device
        self.latent_space_dim = latent_space_dim
        self.features = features
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_channels = in_channel
        
       
        self.encoder = nn.ModuleList().to(self.device)
        self.decoder = nn.ModuleList().to(self.device)
        
      
        for f in features:
            self.encoder.append(
               nn.Sequential(
                      Conv(in_channel=in_channel, out_channel=f, stride=self.stride, kernel_size=self.kernel_size, padding=self.padding),
                   #    Conv(in_channel=f, out_channel=f, stride=self.stride, kernel_size=self.kernel_size, padding=self.padding)  
                ).to(device))
            
            in_channel = f
        
        
        self.flatten = nn.Flatten()

        bottleneck_size = calculate_bottleneck_size(input_size=input_size, encoder=self.encoder, input_channels=self.input_channels, device=self.device)
        #bottleneck_size = self.features[-1]

     
        print(f"Bottleneck size: {bottleneck_size}")

        self.mu_layer = nn.Linear(bottleneck_size, self.latent_space_dim[0]).to(device)
        self.log_var_layer = nn.Linear(bottleneck_size, self.latent_space_dim[0]).to(device)
        self.latent_to_decoder_layer = nn.Linear(self.latent_space_dim[0], bottleneck_size).to(device)

        nn.init.xavier_uniform_(self.mu_layer.weight)
        nn.init.zeros_(self.mu_layer.bias)
        nn.init.xavier_uniform_(self.log_var_layer.weight)
        nn.init.zeros_(self.log_var_layer.bias)
        nn.init.xavier_uniform_(self.latent_to_decoder_layer.weight)
        nn.init.zeros_(self.latent_to_decoder_layer.bias)

        
        
        for i, f in enumerate(features[::-1]):
            out_channel=features[::-1][i+1] if i+1 != len(features) else 3
            final_layer = True if i+1==len(features) else False
            self.decoder.append(
                nn.Sequential(DeConv(in_channel=f, out_channel=out_channel, kernel_size=self.kernel_size, final_layer=final_layer),# DeConv(in_channel=out_channel, out_channel=out_channel, kernel_size=self.kernel_size, final_layer=final_layer)
                              )
                              ).to(device)

        
        self.to(self.device)

    def encode(self, x):
        
      #  print(f"Initial shape of input: {x.shape}")
        x = x.to(self.device)

        for i, layer in enumerate(self.encoder):
            x = layer(x)
        #    print(f"Shape at layer {i} of encoder: {x.shape}")
        
        self.shape_before_bottleneck = x.shape
      #  print(f"Shape before flattening: {self.shape_before_bottleneck}\n\n")

       
        x = self.flatten(x)

     #   print(f"Shape after flattening: {x.shape}\n")

        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x) + 1e-8

        return mu, log_var
    
    def _reparameterize(self, mu, log_var):
        if self.training:
            epsilon = torch.randn_like(mu)
            return mu + torch.exp(0.5 * log_var) * epsilon
        else:
            return mu
    
    def decode(self, z):
        
        z = z.to(self.device)

        x = self.latent_to_decoder_layer(z)
     #   print(f"Output shape (analogous to flattened data): {x.shape}\n")
        x = x.view(-1, self.shape_before_bottleneck[1], self.shape_before_bottleneck[2], self.shape_before_bottleneck[3])
     #   print(f"Output shape (analogous to before flattening): {x.shape} \n")

        for i, layer in enumerate(self.decoder):
            x = layer(x)
    #        print(f"Shape at layer {i} of decoder: {x.shape}")
    #    print(f"Output shape (should be equal to initial input shape): {x.shape}")

        return x
    
    def forward(self, x):
       
        mu, log_var = self.encode(x)
        z = self._reparameterize(mu, log_var)
        reconstructed = self.decode(z)

        if torch.isnan(reconstructed).any():
          print("NaN detected in reconstructed output, replacing with random constant")
          random_constant = torch.randn_like(reconstructed) * 0.1 
          reconstructed = torch.nan_to_num(reconstructed, nan=random_constant)
        
        return reconstructed