import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.ModelParameters import ModelParameters
from dataclasses import dataclass
from data import GenreateTopologies as Top
import torch
import paths
import numpy as np
import utils_public as up
import model.train_util as train_util
from torchsummary import summary


# 64x64 image has 64+64+62+62+1 pixels of interest
# Construct: take all edges and lay them out + 1 for center color
# Reconstruct: make zeroes and fill with center color. fill out edges

def flatten_constraint_image(constraints_tensor):
    constraints_size = 64
    end = constraints_size-1
    # constraint tensor is n x 1 x 64 x 64
    edge_t = constraints_tensor[:, :, 0, :]
    edge_r = constraints_tensor[:, :, 1:end, end]
    edge_b = constraints_tensor[:, :, end, :]
    edge_l = constraints_tensor[:, :, 1:end, 0]
    color = constraints_tensor[:, :, constraints_size//2, constraints_size//2]

    # Concatenate -> n x 4*63+1
    flattened = torch.cat((torch.squeeze(edge_t, 1),
                                  torch.squeeze(edge_r, 1),
                                  torch.squeeze(edge_b, 1),
                                  torch.squeeze(edge_l, 1),
                                  color), 1)

    return flattened

def unflatten_constraint_image(flattened):
    constraints_size = 64
    end = constraints_size-1
    num_images = flattened.shape[0]

    edge_t = flattened[:, 0:64]
    edge_r = flattened[:, 64:126]
    edge_b = flattened[:, 126:190]
    edge_l = flattened[:, 190:252]
    color = torch.squeeze(flattened[:, 252]).cpu()

    # Create base
    ones_index = torch.arange(num_images)[color == 1]
    ones_index.type(torch.int64)
    constraints_tensor = torch.full(size=(num_images, 64, 64), fill_value=0).type(torch.float32)
    constraints_tensor = constraints_tensor.index_fill_(dim=0, index=ones_index, value=1)

    # Fill in edges
    constraints_tensor[:, 0, :] = edge_t
    constraints_tensor[:, 1:end, end] = edge_r
    constraints_tensor[:, end, :] = edge_b
    constraints_tensor[:, 1:end, 0] = edge_l

    constraints_tensor = torch.unsqueeze(constraints_tensor, 1)
    return constraints_tensor.to(paths.device)


def test_flattening(constraints_tensor):
    print("orig sum: ", str(constraints_tensor.sum().item()))

    flattened = flatten_constraint_image(constraints_tensor)
    print(flattened.shape)
    print("flat sum: ", str(flattened.sum().item()))

    unflattened = unflatten_constraint_image(flattened)
    print(unflattened.shape)
    print("flat sum: ", str(unflattened.sum().item()))

    print(torch.equal(unflattened, constraints_tensor))

    length = constraints_tensor.shape[0]
    topology = []
    for i in range(length):
        topology.append(torch.squeeze(constraints_tensor[i, :, :, :].cpu()))
        topology.append(torch.squeeze(unflattened[i, :, :, :].cpu()))

    # up.plot_n_topologies(topology)



@dataclass
class EdgeAutoEncoderParameters(ModelParameters):
    num_layers: int = 3
    latent_dim: int = 5

    def getName(self):
        return (self.__class__.__name__ +
                str(self.num_layers) + "_" +
                str(self.latent_dim) + "_")

    def instantiate_new_model(self):
        return EdgeAutoEncoder(num_layers=self.num_layers,
                                 latent_dim=self.latent_dim)



class EdgeAutoEncoder(nn.Module): #Create AE class inheriting from pytorch nn Module class
    def __init__(self, num_layers, latent_dim):
        super(EdgeAutoEncoder, self).__init__()
        input_size = 4*(64-1) + 1

        # Create encoder model
        self.encoder = Encoder(input_size, num_layers, latent_dim)

        #Create decoder after calculating input size for decoder
        self.decoder = Decoder(input_size, num_layers, latent_dim)

    def forward(self, x_in):
        #Pass through encoder, reparameterize using mu and logvar as given by the encoder, then pass through decoder
        z = self.encoder(x_in)
        x_recon = self.decoder(z)
        return x_recon

    def set_to_evaluation_mode(self, mode):
        self.encoder.set_to_evaluation_mode(mode)
        self.decoder.set_to_evaluation_mode(mode)

class Encoder(nn.Module): #Encoder model of VAE
    def __init__(self, input_size, num_layers, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        scale = np.power(latent_dim * 1.0 / input_size, 1.0/(num_layers+1))
        self.evaluation_mode = True

        layers = []
        last_layer_dim = input_size
        for _ in range(num_layers): # Loop over layers, adding conv2d, layernorm, and relu.
            layer_dim = int(scale * last_layer_dim)
            layers.append(
                nn.Sequential(
                    nn.Linear(last_layer_dim, layer_dim),
                    nn.ReLU()
                )
            )
            last_layer_dim = layer_dim

        layers.append(
            nn.Sequential(
                nn.Linear(last_layer_dim, latent_dim),
                nn.ReLU()
            )
        )
        self.layers = nn.ModuleList(layers)

    # When evaluating, input the image
    # When training, input the flattened image
    def set_to_evaluation_mode(self, mode):
        self.evaluation_mode = mode

    # Input is the image
    def forward(self, x): #Forward call for encoder
        if self.evaluation_mode:
            x = flatten_constraint_image(x)
        for layer in self.layers:
            x = layer(x)
        return x  # Final sigmoid layer

class Decoder(nn.Module):  #Decoder model of VAE
    def __init__(self, output_size, num_layers, latent_dim):
        super(Decoder, self).__init__()
        self.decoder_input_size = output_size
        self.evaluation_mode = True

        scale = np.power(output_size * 1.0 / latent_dim, 1.0 / (num_layers + 1))

        layers = []
        last_layer_dim = latent_dim
        for _ in range(num_layers):  # Loop over layers, adding conv2d, layernorm, and relu.
            layer_dim = int(scale * last_layer_dim)
            layers.append(
                nn.Sequential(
                    nn.Linear(last_layer_dim, layer_dim),
                    nn.ReLU()
                )
            )
            last_layer_dim = layer_dim
        layers.append(
            nn.Sequential(
                nn.Linear(last_layer_dim, output_size),
                nn.ReLU()
            )
        )
        self.layers = nn.ModuleList(layers)

    # When evaluating, output the image
    # When training, output the flattened image
    def set_to_evaluation_mode(self, mode):
        self.evaluation_mode = mode

    # Output is unflattened image
    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        z = torch.sigmoid(z) #Final sigmoid layer
        if self.evaluation_mode:
            z = unflatten_constraint_image(z)
        return z

def loss_function(recon_x, x_out):
    # VAE loss is a sum of KL Divergence regularizing the latent space and reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x_out, reduction='sum') # Reconstruction loss from Binary Cross Entropy
    return BCE

def train(model, optimizer, batch_size, epoch, data_in_tensor, data_out_tensor): #Train function for one epoch of training
    model.train()
    train_loss = 0
    num_batches = len(data_in_tensor) // batch_size

    #Tqdm progress bar object contains a list of the batch indices to train over
    progress_bar = tqdm(range(num_batches), desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)

    for batch_idx in progress_bar: #Loop over batch indices
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        data_in = data_in_tensor[start_idx:end_idx] #Gather corresponding data
        data_out = data_out_tensor[start_idx:end_idx] #Gather corresponding data

        optimizer.zero_grad()  # Set up optimizer
        recon_batch = model(data_in) #Call model
        loss = loss_function(recon_batch, data_out) #Call loss function
        loss.backward() #Get gradients of loss
        train_loss += loss.item() #Append to total loss
        optimizer.step() #Update weights using optimizeer

        # Updating the progress bar
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

    average_train_loss = train_loss / len(data_in_tensor) #Calculate average train loss
    tqdm.write('Epoch: {} \tTraining Loss: {:.3f}'.format(epoch, average_train_loss))

if __name__ == "__main__":
    constraint_type = Top.Constraint.V_LOAD

    # Construction should just reconstruct from original
    masked_data_sources = [Top.Source.TRAIN]
    masked_topology_sets = Top.get_topology_sets_from_sources(sources=masked_data_sources)

    out_data_sources = [Top.Source.TRAIN]
    out_topology_sets = Top.get_topology_sets_from_sources(sources=out_data_sources)

    indices = list(range(out_topology_sets.length))
    print(out_topology_sets.length)
    indices.extend(indices)

    print("setting up inputs")
    data_in_tensor = masked_topology_sets.get_constraints_tensor(indices=indices, as_tensor=True, constraint_type=constraint_type)
    print("setting up outputs")
    data_out_tensor = out_topology_sets.get_constraints_tensor(indices=indices, as_tensor=True, constraint_type=constraint_type)

    print("flattening")
    flattened_data_in = flatten_constraint_image(data_in_tensor)
    flattened_data_out = flatten_constraint_image(data_out_tensor)

    num_epochs = 100
    batch_size = 16

    model_parameters = EdgeAutoEncoderParameters(num_layers=2,
                                                 latent_dim=5)

    print("setting up model")
    model = model_parameters.instantiate_new_model().to(paths.device)
    model.set_to_evaluation_mode(False)
    # print(summary(model, (256, 253)))
    lr = 1e-3
    wd = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    print("begin training")
    for epoch in range(1, num_epochs + 1):
        train(model=model, optimizer=optimizer, batch_size=batch_size, epoch=epoch, data_in_tensor=flattened_data_in,
              data_out_tensor=flattened_data_out)

    save_name = model_parameters.getName() + "OPT_" + str(lr) + "_" + str(batch_size) + "CT_" + constraint_type.name + ".pickle"
    train_util.save_model_and_hp(model, model_parameters, batch_size=batch_size, name=save_name, constraint=constraint_type)