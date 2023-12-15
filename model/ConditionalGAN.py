import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import data.GenreateTopologies as Top
import paths
from model.ModelParameters import ModelParameters
from dataclasses import dataclass
import model.train_util as train_util
from model.VFEncoder import VFEncoder
from model.EdgeAutoEncoder import EdgeAutoEncoderParameters, EdgeAutoEncoder, flatten_constraint_image, unflatten_constraint_image
from model.NormalVAE import SimpleVAE, SimpleVAEParameters

@dataclass
class CGanParameters(ModelParameters):
    latent_top_dim: int = 1
    latent_const_dim: int = 5
    n_layers: int = 1
    layer_size: int = 5
    relu_leak: float = 0.2

    def getName(self):
        return (self.__class__.__name__ +
                str(self.latent_top_dim) + "_" +
                str(self.latent_const_dim) + "_" +
                str(self.n_layers) + "_" +
                str(self.layer_size) + "_" +
                str(self.relu_leak) + "_")
    def instantiate_new_model(self):
        return (Generator(latent_top_dim=self.latent_top_dim,
                          latent_const_dim=self.latent_const_dim,
                          n_layers=self.n_layers,
                          layer_size=self.layer_size),
                Discriminator(latent_top_dim=self.latent_top_dim,
                              latent_const_dim=self.latent_const_dim,
                              n_layers=self.n_layers,
                              layer_size=self.layer_size,
                              relu_leak=self.relu_leak))

class Generator(nn.Module): #Encoder model of VAE
    def __init__(self, latent_top_dim, latent_const_dim, n_layers, layer_size):
        super(Generator, self).__init__()
        self.latent_top_dim = latent_top_dim

        layers = []
        input_size = latent_top_dim + latent_const_dim
        layers.append(
            nn.Sequential(
                nn.Linear(input_size, layer_size),
                nn.ReLU()
            )
        )
        for _ in range(n_layers): # Loop over layers, adding conv2d, layernorm, and relu.
            layers.append(
                nn.Sequential(
                    nn.Linear(layer_size, layer_size),
                    nn.ReLU()
                )
            )
        layers.append(
            nn.Sequential(
                nn.Linear(layer_size, latent_top_dim),
                nn.ReLU()
            )
        )
        self.layers = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #Forward call for encoder
        for layer in self.layers: #Call conv layers sequentially
            x = layer(x)
        return self.sigmoid(x)

class Discriminator(nn.Module): #Encoder model of VAE
    def __init__(self, latent_top_dim, latent_const_dim, n_layers, layer_size, relu_leak):
        super(Discriminator, self).__init__()

        layers = []
        input_size = latent_top_dim + latent_const_dim
        layers.append(
            nn.Sequential(
                nn.Linear(input_size, layer_size),
                nn.LeakyReLU(relu_leak)
            )
        )
        for _ in range(n_layers): # Loop over layers, adding conv2d, layernorm, and relu.
            layers.append(
                nn.Sequential(
                    nn.Linear(layer_size, layer_size),
                    nn.LeakyReLU(relu_leak)
                )
            )
        layers.append(
            nn.Sequential(
                nn.Linear(layer_size, 1),
                nn.LeakyReLU(relu_leak)
            )
        )
        self.layers = nn.ModuleList(layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z): #Forward call for encoder
        for layer in self.layers: #Call conv layers sequentially
            z = layer(z)
        return self.sigmoid(z)


def get_generator_loss(fake_inputs, fake_topology_latent, transformed_fake_topology_latent):
    n_fake = len(fake_inputs)
    fake_target = torch.full((n_fake,), 1.0, dtype=torch.float, device=paths.device)
    fake_loss = nn.functional.binary_cross_entropy(fake_inputs, fake_target, reduction='sum')

    deviation_loss = nn.functional.binary_cross_entropy(fake_topology_latent, transformed_fake_topology_latent, reduction='sum')

    return fake_loss + deviation_loss


def get_discriminator_loss(fake_inputs, real_inputs):
    n_fake = len(fake_inputs)
    n_real = len(real_inputs)
    fake_target = torch.full((n_fake,), 0.0, dtype=torch.float, device=paths.device)
    real_target = torch.full((n_real,), 1.0, dtype=torch.float, device=paths.device)

    fake_loss = nn.functional.binary_cross_entropy(fake_inputs, fake_target, reduction='sum')
    real_loss = nn.functional.binary_cross_entropy(real_inputs, real_target, reduction='sum')

    return fake_loss + real_loss

def train(generator_model: Generator, discriminator_model: Discriminator,
          generator_optimizer, discriminator_optimizer,
          batch_size, epoch,
          real_topology_tensor, topology_encoder,
          constraint_tensor, constraint_encoder): #Train function for one epoch of training
    generator_model.train()
    discriminator_model.train()

    latent_top_dim = generator_model.latent_top_dim

    train_loss = 0
    num_batches = len(real_topology_tensor) // batch_size

    #Tqdm progress bar object contains a list of the batch indices to train over
    progress_bar = tqdm(range(num_batches), desc='Epoch {:03d}'.format(epoch), leave=False, disable=False)

    train_gan = False

    for batch_idx in progress_bar: #Loop over batch indices
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        # Get constraints
        constraints = constraint_tensor[start_idx:end_idx]
        constraints_latent = constraint_encoder(constraints)

        # Generate real images
        real_topology = real_topology_tensor[start_idx:end_idx] #Gather corresponding data
        mu, logvar = topology_encoder(real_topology)
        real_topology_latent = topology_encoder.reparameterize(mu, logvar)
        # Concatenate
        real_input = torch.cat((real_topology_latent, constraints_latent), 1)

        # Generate fake images
        fake_topology_latent = torch.rand(batch_size, latent_top_dim).to(paths.device)
        # fake_topology_latent = topology_encoder(fake_topology)
        # Concatenate
        fake_input = torch.cat((fake_topology_latent, constraints_latent), 1)

        # Transform fake images via generator
        generator_optimizer.zero_grad()  # Set up optimizer
        transformed_fake_topology_latent = generator_model(fake_input)
        transformed_fake_input = torch.cat((transformed_fake_topology_latent, constraints_latent), 1)

        # Pass through discriminator to calculate loss
        fake_evaluations = torch.squeeze(discriminator_model(transformed_fake_input))

        # Calculate loss
        # Training alternately resolves the "trying to backwards twice" issue
        if train_gan:
            generator_loss = get_generator_loss(fake_evaluations, fake_topology_latent, transformed_fake_topology_latent)
            generator_loss.backward()
            generator_optimizer.step()
            loss = generator_loss.item()
        else:
            discriminator_optimizer.zero_grad()
            real_evaluations = torch.squeeze(discriminator_model(real_input))
            discriminator_loss = get_discriminator_loss(fake_evaluations, real_evaluations)
            discriminator_loss.backward()
            discriminator_optimizer.step()
            loss = discriminator_loss.item()

        train_loss += loss
        train_gan = not train_gan

        # Updating the progress bar
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss)})

    average_train_loss = train_loss / len(real_topology_tensor) #Calculate average train loss
    tqdm.write('Epoch: {} \tTraining Loss: {:.3f}'.format(epoch, average_train_loss))

if __name__ == "__main__":
    # Only train for
    constraint_type = Top.Constraint.VOL_FRAC

    if constraint_type is not Top.Constraint.VOL_FRAC:
        constraints_param_map = {
            Top.Constraint.H_BC: "ae_hbc",
            Top.Constraint.V_BC: "ae_vbc"
        }
        constraint_model, _, _ = train_util.load_model_from_file(paths.get_favorite_model_path(constraints_param_map[constraint_type]))
        constraint_encoder = constraint_model.encoder
    else:
        constraint_encoder = VFEncoder()

    topology_model, _ = train_util.load_model_from_file(paths.get_favorite_model_path("vae_topology"))
    topology_encoder = topology_model.encoder

    # Construction should just reconstruct from original
    data_sources = [Top.Source.TRAIN]
    topology_sets = Top.get_topology_sets_from_sources(sources=data_sources)

    indices = list(range(topology_sets.length))
    print(topology_sets.length)
    indices.extend(indices)

    print("setting up inputs")
    real_topology_tensor = topology_sets.get_topology_tensor(indices=indices, as_tensor=True)
    constraints_tensor = topology_sets.get_constraints_tensor(indices=indices, as_tensor=True, constraint_type=constraint_type)

    print("setting up model")
    model_parameters = CGanParameters(latent_top_dim=topology_encoder.latent_dim,
                                      latent_const_dim=constraint_encoder.latent_dim,
                                      n_layers = 3,
                                      layer_size = 64,
                                      relu_leak=0.2)
    generator_model, discriminator_model = model_parameters.instantiate_new_model()
    generator_model = generator_model.to(paths.device)
    discriminator_model = discriminator_model.to(paths.device)

    # print(summary(model, (256, 253)))
    num_epochs = 100
    batch_size = 64
    lr = 1e-6
    wd = 1e-8
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=lr, weight_decay=wd)
    discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=lr, weight_decay=wd)

    print("begin training")
    for epoch in range(1, num_epochs + 1):
        train(generator_model=generator_model, discriminator_model=discriminator_model,
              generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
              batch_size=batch_size, epoch=epoch,
              real_topology_tensor=real_topology_tensor, topology_encoder=topology_encoder,
              constraint_tensor=constraints_tensor, constraint_encoder=constraint_encoder)

    save_name = model_parameters.getName() + "OPT_" + str(lr) + "_" + str(batch_size) + "CT_" + constraint_type.name + ".pickle"
    train_util.save_gan_model_and_hp(generator=generator_model, discriminator=discriminator_model,
                                     hyperparam=model_parameters, batch_size=batch_size,
                                     name=save_name, constraint=constraint_type)