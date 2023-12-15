import matplotlib.pyplot as plt
import torch
import numpy as np
import paths
from model import train_util as util
from data import GenreateTopologies as Top
import torch.nn as nn
from model.AutoEncoder import AutoEncoderParameters, SimpleAutoEncoder
from model.EdgeAutoEncoder import EdgeAutoEncoderParameters, EdgeAutoEncoder, flatten_constraint_image, unflatten_constraint_image
from model.VFEncoder import VFEncoder
from model.NormalVAE import SimpleVAE, SimpleVAEParameters
import model.train_util as train_util
from model.ConditionalGAN import CGanParameters

class TransformingDemasker():
    def __init__(self):
        topology_model, _ = train_util.load_model_from_file(paths.get_favorite_model_path("vae_topology"))
        self.topology_encoder = topology_model.encoder
        self.topology_decoder = topology_model.decoder

        constraints_param_map = {
            Top.Constraint.H_BC: ("ae_hbc", "gan_hbc"),
            Top.Constraint.V_BC: ("ae_vbc", "gan_vbc"),
            Top.Constraint.VOL_FRAC: ("", "gan_vf")
        }

        self.constraint_encoders = {}
        self.constraint_generators = {}
        for constraint_type in [Top.Constraint.H_BC, Top.Constraint.V_BC, Top.Constraint.VOL_FRAC]:
            ae_name, gan_name = constraints_param_map[constraint_type]
            if constraint_type is not Top.Constraint.VOL_FRAC:
                constraint_model, _, _ = train_util.load_model_from_file(paths.get_favorite_model_path(ae_name))
                constraint_encoder = constraint_model.encoder
            else:
                constraint_encoder = VFEncoder()
            constraint_generator, _, _ = train_util.load_gan_model_from_file(paths.get_favorite_model_path(gan_name))

            self.constraint_encoders[constraint_type] = constraint_encoder
            self.constraint_generators[constraint_type] = constraint_generator

    # Image must be 1x1x64x64
    def demask(self, topology_sets: Top.Topologies, index):
        with torch.no_grad():
            topology_tensor = topology_sets.get_topology_tensor([index], as_tensor=True)
            constraints = topology_sets.get_data(index).constraints


            # Feed through topology encoder
            mu, logvar = self.topology_encoder(topology_tensor)
            topology_latent = self.topology_encoder.reparameterize(mu, logvar)

            # Check if constraints are available
            if constraints[Top.Constraint.H_BC] is not None:
                topology_latent = self.do_latent_transformation_for_constraint(topology_latent=topology_latent,
                                                                               constraint_type=Top.Constraint.H_BC,
                                                                               topology_sets=topology_sets,
                                                                               index=index)
            if constraints[Top.Constraint.V_BC] is not None:
                topology_latent = self.do_latent_transformation_for_constraint(topology_latent=topology_latent,
                                                                               constraint_type=Top.Constraint.V_BC,
                                                                               topology_sets=topology_sets,
                                                                               index=index)
            if constraints[Top.Constraint.VOL_FRAC] is not None:
                topology_latent = self.do_latent_transformation_for_constraint(topology_latent=topology_latent,
                                                                               constraint_type=Top.Constraint.VOL_FRAC,
                                                                               topology_sets=topology_sets,
                                                                               index=index)
            topology_reconstructed = self.topology_decoder(topology_latent)

            topology_reconstructed = topology_reconstructed[:, 0, :, :].to('cpu').numpy()
            topology_reconstructed = np.round(topology_reconstructed)
        return topology_reconstructed

    def do_latent_transformation_for_constraint(self, topology_latent, constraint_type, topology_sets, index):
        constraints_tensor = topology_sets.get_constraints_tensor(indices=[index], as_tensor=True, constraint_type=constraint_type)
        constraints_latent = self.constraint_encoders[constraint_type](constraints_tensor)

        input = torch.cat((topology_latent, constraints_latent), 1)
        transformed_topology_latent = self.constraint_generators[constraint_type](input)
        return transformed_topology_latent


class VaeDemasker:
    def __init__(self):
        self.topology_model, _ = train_util.load_model_from_file(paths.get_favorite_model_path("vae_topology"))

    def demask(self, topology_sets: Top.Topologies, index):
        with torch.no_grad():
            topology_tensor = topology_sets.get_topology_tensor([index], as_tensor=True)
            samples = self.topology_model(topology_tensor)[0][:, 0, :, :].to('cpu').numpy()
            samples = np.round(samples)
        return samples

def plot_reconstruction(originals, masked, reconstructions1, reconstructions2):
    # Function to plot reconstructed city grids alongside originals
    n = len(originals)
    fig, axes = plt.subplots(nrows=n, ncols=5, figsize=(9, 2 * n))
    for i in range(n):  # Loop over the grids
        axes[i, 0].imshow(masked[i], cmap="gray")  # Plot masked on the left
        axes[i, 1].imshow(reconstructions1[i], cmap="gray")  # Plot reconstruction on the left
        axes[i, 2].imshow(originals[i], cmap="gray")  # Plot originals on the right
        axes[i, 3].imshow(reconstructions2[i], cmap="gray")  # Plot reconstruction on the left
        axes[i, 4].imshow(originals[i] - reconstructions1[i], cmap="RdBu", vmin=-1,
                          vmax=1)  # Plot error on the right
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    save = True
    transforming_demasker = TransformingDemasker()
    vae_demasker = VaeDemasker()

    if save:
        masked_data_sources = [Top.Source.SUBMISSION_MASKED]
        masked_topology_sets = Top.get_topology_sets_from_sources(sources=masked_data_sources)

        reconstructions_submission = []
        for i in range(masked_topology_sets.length):
            # reconstructions_submission.append(transforming_demasker.demask(topology_sets=masked_topology_sets, index=i)[0])
            reconstructions_submission.append(vae_demasker.demask(topology_sets=masked_topology_sets, index=i)[0])

        # reconstructions_submission = np.array(reconstructions_submission)
        reconstructions_submission = np.round(reconstructions_submission).astype(bool)
        print(reconstructions_submission.shape)
        print(reconstructions_submission.dtype)
        assert reconstructions_submission.shape == (1200, 64, 64)
        assert reconstructions_submission.dtype == bool
        np.save("CP3_final_submission.npy", reconstructions_submission)
    else:
        masked_data_sources = [Top.Source.TEST_MASKED]
        masked_topology_sets = Top.get_topology_sets_from_sources(sources=masked_data_sources)

        num_masked_sources = len(masked_data_sources)
        out_data_sources = [Top.Source.TEST]
        for i in range(num_masked_sources - 1):
            out_data_sources.append(Top.Source.TEST)
        out_topology_sets = Top.get_topology_sets_from_sources(sources=out_data_sources)

        indices = np.random.choice(np.arange(out_topology_sets.length), size=7, replace=False)  # Select 5 random indices
        vae_reconstructions = []
        transform_reconstructions = []
        for i in indices:
            vae_reconstructions.append(vae_demasker.demask(topology_sets=masked_topology_sets, index=i)[0])
            transform_reconstructions.append(transforming_demasker.demask(topology_sets=masked_topology_sets, index=i)[0])

        plot_reconstruction(out_topology_sets.get_topology_tensor(indices=indices),
                            masked_topology_sets.get_topology_tensor(indices=indices),
                            np.array(vae_reconstructions),
                            np.array(transform_reconstructions))  # Compare
