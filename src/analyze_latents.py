from itertools import pairwise

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.manifold import TSNE
import os
import json
from scipy.spatial.distance import pdist, squareform
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import torchvision
from torch.distributions import Normal




def analyze_psnr_values(psnr_values, plots_dir=None, out_name_prefix="", show_figures=False):
    """
    This function takes a list of PSNR values and plots them.

    Args:
    psnr_vals (list): List of PSNR values.
    """

    # Calculate mean and variance
    mean_psnr = np.mean(psnr_values)
    variance_psnr = np.var(psnr_values)
    std_psnr = np.std(psnr_values)

    # Plot histogram
    plt.hist(psnr_values, bins=10, edgecolor='black', alpha=0.7)

    # Add mean and variance to the plot
    plt.axvline(mean_psnr, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_psnr, plt.ylim()[1] * 0.9, f' Mean: {mean_psnr:.2f}', color='red')
    plt.text(mean_psnr, plt.ylim()[1] * 0.8, f' std: {std_psnr:.2f}', color='red')

    # Add labels and title
    plt.title('Histogram of PSNR Values')
    plt.xlabel('PSNR')
    plt.ylabel('Frequency')

    if plots_dir:
        plt.savefig(os.path.join(plots_dir, f'{out_name_prefix}psnr_histogram.pdf'))

    if show_figures:
        plt.show()

    # close all plot windows
    plt.close('all')


def tensor_to_nll(z):
    standard_normal = Normal(0, 1)
    likelihood = standard_normal.log_prob(z.detach().cpu()).sum()
    return -1 * likelihood.item()


def analyze_nll_values(nll_values, plots_dir=None, out_name_prefix="", show_figures=False):
    """
    This function takes a list of NLL values and plots them.

    Args:
    nll_vals (list): List of NLL values.
    """

    # Calculate mean and variance
    mean_nll = np.mean(nll_values)
    # variance_nll = np.var(nll_values)
    std_nll = np.std(nll_values)

    # Plot histogram
    plt.hist(nll_values, bins=10, edgecolor='black', alpha=0.7)

    # Add mean and variance to the plot
    plt.axvline(mean_nll, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_nll, plt.ylim()[1] * 0.9, f' Mean: {mean_nll:.2f}', color='red')
    plt.text(mean_nll, plt.ylim()[1] * 0.8, f' std: {std_nll:.2f}', color='red')

    # Add labels and title
    plt.title('Histogram of NLL Values')
    plt.xlabel('NLL')
    plt.ylabel('Frequency')

    if plots_dir:
        plt.savefig(os.path.join(plots_dir, f'{out_name_prefix}z_T_nll_histogram.png'))

    if show_figures:
        plt.show()

    # close all plot windows
    plt.close('all')



def calculate_emd(mean1, var1, mean2, var2):
    # Compute EMD for scalar values
    emd = (mean1 - mean2)**2 + var1 + var2 - 2 * np.sqrt(var1 * var2)
    return emd

def analyze_goal_source_normal_nll_values(nll_values_1, nll_values_2, label_1="Distribution 1",
                         label_2="Distribution 2", label_3=None, plots_dir=None,
                         out_name_prefix="", show_figures=False):
    """
    This function takes two lists of NLL values and plots them on the same graph, each with a different color.

    Args:
    nll_values_1 (list): List of NLL values for the first distribution.
    nll_values_2 (list): List of NLL values for the second distribution.
    add_random_nll (bool): If True, add a list of NLL values of random samples to the plot (default is False).
    label_1 (str): Label for the first distribution in the legend.
    label_2 (str): Label for the second distribution in the legend.
    label_3 (str): Label for the random distribution in the legend (optional).
    plots_dir (str): Directory to save the plot (optional).
    out_name_prefix (str): Prefix for the saved plot filename (optional).
    """

    # Calculate mean and variance for both distributions
    mean_nll_1 = np.mean(nll_values_1)
    std_nll_1 = np.std(nll_values_1)
    var_nll_1 = np.var(nll_values_1)

    mean_nll_2 = np.mean(nll_values_2)
    std_nll_2 = np.std(nll_values_2)
    var_nll_2 = np.var(nll_values_2)

    # random_samples = [torch.randn(1, 4, 64, 64).to('cuda') for r in range(len(nll_values_1))]
    random_samples = [torch.randn(1, 4, 64, 64).to('cuda') for r in range(50)]
    nll_values_3 = [tensor_to_nll(z) for z in random_samples]
    mean_nll_3 = np.mean(nll_values_3)
    std_nll_3 = np.std(nll_values_3)
    var_nll_3 = np.var(nll_values_3)

    # Calculate common bin edges
    all_values = np.concatenate([nll_values_1, nll_values_2] + ([nll_values_3]))
    bin_edges = np.histogram_bin_edges(all_values, bins=50+len(nll_values_1)+len(nll_values_2))

    # Plot histograms for both distributions with common bin edges
    plt.hist(nll_values_1, bins=bin_edges, alpha=0.7, label=label_1, color='cyan', density=True)
    plt.hist(nll_values_2, bins=bin_edges, alpha=0.7, label=label_2, color='orange', density=True)
    plt.hist(nll_values_3, bins=bin_edges, alpha=0.7, label=label_3, color='limegreen', density=True)

    # Add mean and variance lines for both distributions
    plt.axvline(mean_nll_1, color='blue', linestyle='dashed', linewidth=1)
    plt.text(mean_nll_1, plt.ylim()[1] * 0.9, f' Mean: {mean_nll_1:.2f}', color='blue')
    plt.text(mean_nll_1, plt.ylim()[1] * 0.8, f'std: {std_nll_1:.2f}', color='blue')

    plt.axvline(mean_nll_2, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_nll_2, plt.ylim()[1] * 0.7, f' Mean: {mean_nll_2:.2f}', color='red')
    plt.text(mean_nll_2, plt.ylim()[1] * 0.6, f' std: {std_nll_2:.2f}', color='red')

    plt.axvline(mean_nll_3, color='darkgreen', linestyle='dashed', linewidth=1)
    plt.text(mean_nll_3, plt.ylim()[1] * 0.5, f' Mean: {mean_nll_3:.2f}', color='darkgreen')
    plt.text(mean_nll_3, plt.ylim()[1] * 0.4, f' std: {std_nll_3:.2f}', color='darkgreen')

    # Add labels, title, and legend
    plt.title('Histogram of NLL Values for Two Distributions')
    plt.xlabel('NLL')
    plt.ylabel('Frequency')
    plt.legend()

    # Save the plot if a directory is provided
    if plots_dir:
        plt.savefig(os.path.join(plots_dir, f'{out_name_prefix}z_T_nll_both_histograms.png'))

    if show_figures:
        plt.show()

    # Close all plot windows
    plt.close('all')

    emd_scores = {}
    emd_scores['goal_and_normal_emd_score'] = calculate_emd(mean_nll_1, var_nll_1, mean_nll_3, var_nll_3)
    emd_scores['goal_and_source_emd_score'] = calculate_emd(mean_nll_1, var_nll_1, mean_nll_2, var_nll_2)
    emd_scores['source_and_normal_emd_score'] = calculate_emd(mean_nll_2, var_nll_2, mean_nll_3, var_nll_3)
    return emd_scores


# Helper function to compute SSIM between two images
def ssim(img1, img2, window_size=11, size_average=True, val_range=None):
    # Get the number of channels
    if len(img1.shape) == 2:  # Grayscale case
        img1 = img1.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif len(img1.shape) == 3:  # Single-channel batch
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Use torchvision's SSIM for each pair of images
    return torchvision.metrics.functional.ssim(
        img1, img2, data_range=val_range or img1.max() - img1.min(),
        window_size=window_size, size_average=size_average
    )


def plot_latent_distances(latents, metric='euclidean', plots_dir=None, index=None, show_figures=False):
    """
    This function computes the pairwise distance matrix for a list of latent tensors,
    visualizes it as a heatmap with a colorbar, and shows the average distance and variance numerically.
    Diagonal elements are ignored for the distance calculation.

    Args:
    latents (list): List of latent tensors.
    metric (str): Distance metric to use ('euclidean', 'cosine', 'ssim').
    plots_dir (str): Directory to save the plot.
    index (int): Optional index for saving the file.
    """

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    # Flatten the tensors if not using SSIM (SSIM needs 2D tensors)
    if metric == 'ssim':
        latents_flat = [latent.cpu() for latent in latents]  # Keep the latents as 2D tensors for SSIM
    else:
        latents_flat = [latent.view(-1).cpu().numpy() for latent in latents]

    # Compute pairwise distances based on the metric
    if metric == 'euclidean':
        distances = squareform(pdist(latents_flat, metric='euclidean'))
    elif metric == 'cosine':
        # Cosine similarity needs to be transformed into a distance matrix (1 - similarity)
        similarities = cosine_similarity(latents_flat)
        distances = 1 - similarities
    elif metric == 'ssim':
        # Compute SSIM for each pair of images (requires 2D arrays)
        num_latents = len(latents_flat)
        distances = np.zeros((num_latents, num_latents))
        for i in range(num_latents):
            for j in range(i + 1, num_latents):
                ssim_value = ssim(latents_flat[i], latents_flat[j], val_range=1.0)
                distances[i, j] = distances[j, i] = 1 - ssim_value.item()  # SSIM similarity to distance
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Set diagonal values to NaN to ignore them
    np.fill_diagonal(distances, np.nan)

    # Compute statistics (mean and variance, ignoring NaN)
    mean_distance = np.nanmean(distances)
    variance_distance = np.nanvar(distances)
    std_distance = np.nanstd(distances)

    # Plot the distance matrix as a heatmap
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(distances, cmap='viridis')
    ax.set_title(f'Pairwise {metric.capitalize()} Distance Matrix')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Sample Index')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{metric.capitalize()} Distance')

    # Annotate mean and variance above the title
    ax.text(0.5, 1.1, f'Mean: {mean_distance:.4f}, std: {std_distance:.4f}',
            transform=ax.transAxes, fontsize=12, va='center', ha='center')

    plt.tight_layout()
    if plots_dir:
        # Save the figure with the metric name in the filename
        filename = f'latent_distances_{metric}'
        if index:
            filename += f'_{index}'
        plt.savefig(os.path.join(plots_dir, f'{filename}.pdf'))

    if show_figures:
        plt.show()

    # close all plot windows
    plt.close('all')

    distances_to_target = distances[-1, :-1]
    pairwise_distances = distances[:-1, :-1]
    # return mean_distance, std_distance, np.mean(distances_to_target), np.std(distances_to_target), pairwise_distances, distances_to_target
    return np.nanmean(pairwise_distances), np.nanstd(pairwise_distances), np.mean(distances_to_target), np.std(distances_to_target)