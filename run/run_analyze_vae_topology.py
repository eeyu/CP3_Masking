import matplotlib.pyplot as plt
import torch
import numpy as np
import paths
from model import train_util as util
from data import GenreateTopologies as Top
import torch.nn as nn
from model.NormalVAE import SimpleVAE, SimpleVAEParameters

def reconstruct_from_vae(model, masked_topologies, device='cpu'):
    with torch.no_grad():
        # data_in = torch.from_numpy(masked_topologies).float()
        # data_in = data_in.unsqueeze(1).to(device)
        data_in = masked_topologies
        output = model(data_in)
        samples = output[0][:, 0, :, :].to('cpu').numpy()
        samples = np.round(samples)
    return samples


def plot_reconstruction(originals, masked, reconstructions):
    # Function to plot reconstructed city grids alongside originals
    n = len(originals)
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(9, 2 * n))
    for i in range(n):  # Loop over the grids
        axes[i, 0].imshow(masked[i], cmap="gray")  # Plot masked on the left
        axes[i, 1].imshow(reconstructions[i], cmap="gray")  # Plot reconstruction on the left
        axes[i, 2].imshow(originals[i], cmap="gray")  # Plot originals on the right
        axes[i, 3].imshow(originals[i] - reconstructions[i], cmap="RdBu", vmin=-1,
                          vmax=1)  # Plot error on the right
    fig.tight_layout()
    plt.show()

def calculate_loss(x_out, recon_x):
    return nn.functional.binary_cross_entropy(input=recon_x, target=x_out, reduction='sum').item()  # Reconstruction loss from Binary Cross Entropy

if __name__ == "__main__":
    # orig: 984405
    # 0: 787744
    # 0O: 677253
    # 012: 737537
    # 012O: 668291

    file = paths.select_file(paths.PARAM_PATH, choose_file=True)
    print(file)
    model, batch_size = util.load_model_from_file(file)
    masked_data_sources = [Top.Source.TEST_MASKED]
    masked_topology_sets = Top.get_topology_sets_from_sources(sources=masked_data_sources)

    num_masked_sources = len(masked_data_sources)
    out_data_sources = [Top.Source.TEST]
    for i in range(num_masked_sources - 1):
        out_data_sources.append(Top.Source.TEST)
    out_topology_sets = Top.get_topology_sets_from_sources(sources=out_data_sources)

    # Calculate loss
    with torch.no_grad():
        data_in = masked_topology_sets.get_topology_tensor(as_tensor=True)
        recon_batch, mu, logvar = model(data_in)
        data_out = out_topology_sets.get_topology_tensor(as_tensor=True)
        loss = calculate_loss(x_out=data_out, recon_x=recon_batch)
    print("loss: ", str(loss))

    # Plot
    originals = np.random.choice(np.arange(out_topology_sets.length), size=5, replace=False)  # Select 5 random indices
    reconstructions = reconstruct_from_vae(model, masked_topology_sets.get_topology_tensor(indices=originals, mask=False, as_tensor=True), paths.device)  # Reconstruct
    plot_reconstruction(out_topology_sets.get_topology_tensor(indices=originals),
                        masked_topology_sets.get_topology_tensor(indices=originals),
                        reconstructions)  # Compare