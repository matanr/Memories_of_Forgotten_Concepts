import argparse
import os
import yaml
import torch
from easydict import EasyDict
import os.path as osp
from vae_inversion_tools import (
    vae_inversion_start_from_arbitrary_latent,
    vae_inversion_start_from_encoder_latent,
    get_latent_from_encoder,
)
from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel
from PIL import Image
from concept_datasets import CocoCaptions17Paths, CaptionsPaths
import json
from nti import ddim_inversion, null_text_inversion, reconstruct
from analyze_latents import (
    plot_latent_distances,
    analyze_psnr_values,
    analyze_nll_values,
    analyze_goal_reference_normal_nll_values,
)
from utils import tensor_to_nll, calculate_psnr
from detect_concepts import detect_concept_post_analysis
from analysis_utils import clip_similarity
import numpy as np
from renoise.main import run as renoise_invert
from renoise.eunms import Model_Type, Scheduler_Type
from renoise.utils.enums_utils import get_pipes
from renoise.config import RunConfig


def validate_and_get_args():
    """
    Parses and validates command-line arguments for generating unlikely images.
    Returns:
        EasyDict: A dictionary-like object containing the parsed arguments.
    Command-line Arguments:
        --model_base (str): Base SD model. Default is "CompVis/stable-diffusion-v1-4".
        --out_dir (str): Output directory.
        --seed (int): Random seed. Default is 42.
        --image_indices (int): Indices of images to perform NTI on. Default is [0].
        --num_diffusion_inversion_steps (int): Number of optimization steps for NTI. Default is 50.
        --reference_dataset_root (str): Path to the reference root directory. Default is "./mscoco17".
        --dataset_root (str): Path to root directory of the images from the erased concept. Default is "./nudity".
        --ablated_model (str): Path to the ablated diffusion model. Default is None.
        --ablated_text_encoder (str): Path to the ablated text encoder. Default is None.
        --ablated_concept_name (str): Name of the ablated concept. Default is None.
        --diffusion_inversion_method (str): Diffusion inversion method. Valid options are 'nti' and 'renoise'. Default is "renoise".
        --num_vae_inversion_steps (int): Number of VAE inversion steps. Default is 1000.
        --num_ref_images (int): Number of reference images. Default is 10.
        --show_figures (bool): Show figures. Default is False.
        --analyze_only (bool): Analyze only, do not find latents. Default is False.
    """
    parser = argparse.ArgumentParser(
        description="Create many memories for an erased image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_base",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base SD model",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--image_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="indices of images. This index is the relative index of the image in the dataset directory",
    )

    parser.add_argument(
        "--num_diffusion_inversion_steps",
        type=int,
        default=50,
        help="Number of optimization steps for NTI",
    )

    parser.add_argument(
        "--reference_dataset_root",
        type=str,
        default="./mscoco17",
        help="Path to the reference root directory (default is MSCOCO17): https://cocodataset.org/#download",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./nudity",
        help="Path to root directory of the images from the erased concept",
    )
    parser.add_argument(
        "--ablated_model",
        type=str,
        default=None,
        help="Path to the ablated diffusion model",
    )
    parser.add_argument(
        "--ablated_text_encoder",
        type=str,
        default=None,
        help="Path to the ablated text encoder",
    )

    parser.add_argument(
        "--ablated_concept_name",
        type=str,
        default=None,
        help="Name of the ablated concept. For example: nudity, vangogh, parachute, garbage_truck, tench, church",
    )
    parser.add_argument(
        "--diffusion_inversion_method",
        type=str,
        default="renoise",
        help="Diffusion inversion method. Valid options are 'nti' and 'renoise'",
    )
    parser.add_argument(
        "--num_vae_inversion_steps",
        type=int,
        default=3000,
        help="Number of VAE inversion steps",
    )
    parser.add_argument(
        "--num_ref_images", type=int, default=10, help="Number of reference images"
    )
    parser.add_argument(
        "--show_figures", default=False, action="store_true", help="Show figures"
    )
    parser.add_argument(
        "--analyze_only",
        default=False,
        action="store_true",
        help="Analyze only, do not find latents",
    )

    args = parser.parse_args()
    args = EasyDict(vars(args))

    return args


def start_from_latents_from_other_imgs_vae_inversion(
    pipe, args, imgs_list, test_out_dir
):
    """
    Perform VAE inversion starting from latents of other images.
    This function performs VAE inversion on a list of images, starting from the latent
    representation of each image in the list. The target image is the last image in the
    list, and the reference images are all the other images in the list. The function saves
    the latent representations of the target image and the reference images, as well as the
    latent representations of the target image starting from the latent of each reference image.
    Args:
        pipe: The pipeline object containing the VAE encoder and decoder.
        args: The arguments object containing the number of VAE inversion steps.
        imgs_list: A list of images to perform VAE inversion on.
        test_out_dir: The directory to save the output latent representations.
    Returns:
        None
    """
    target_latent = vae_inversion_start_from_encoder_latent(
        encoder=pipe.vae.encode,
        decoder=pipe.vae.decode,
        image=imgs_list[-1],
        num_steps=args.num_vae_inversion_steps,
        out_dir=osp.join(test_out_dir, "vae_inversion_target_image"),
        scale_reconstruction=1,
        scale_l2=0,
        scale_gaussian=0,
    )
    torch.save(target_latent, os.path.join(test_out_dir, "z_0_target.pth"))

    for img_index, starting_img in enumerate(imgs_list[:-1]):
        starting_latent = vae_inversion_start_from_encoder_latent(
            encoder=pipe.vae.encode,
            decoder=pipe.vae.decode,
            image=starting_img,
            num_steps=args.num_vae_inversion_steps,
            out_dir=osp.join(test_out_dir, f"vae_inversion_reference_{img_index}"),
            scale_reconstruction=1,
            scale_l2=0,
            scale_gaussian=0,
        )
        target_latent_start_from_latent = vae_inversion_start_from_arbitrary_latent(
            encoder=pipe.vae.encode,
            decoder=pipe.vae.decode,
            image=imgs_list[-1],
            num_steps=args.num_vae_inversion_steps,
            latent_init=starting_latent.detach().clone(),
            epsilon=None,
            out_dir=osp.join(
                test_out_dir, f"vae_inversion_target_start_from_reference_{img_index}"
            ),
            scale_reconstruction=1,
            scale_l2=0,
            scale_gaussian=0,
        )

        torch.save(
            starting_latent, os.path.join(test_out_dir, f"z_0_reference_{img_index}.pth")
        )
        torch.save(
            target_latent_start_from_latent,
            os.path.join(test_out_dir, f"z_0_target_start_from_reference_{img_index}.pth"),
        )


def sample_latents_around_target(pipe, args, imgs_list, test_out_dir):
    """
    Samples latent vectors around a target image and saves them to disk.
    Args:
        pipe (object): The pipeline object containing the VAE encoder and decoder.
        args (object): An object containing the number of VAE inversion steps.
        imgs_list (list): A list of images, where the last image is the target image.
        test_out_dir (str): The directory where the output latent vectors will be saved.
    Saves:
        - The target latent vector as "z_0_target.pth" in the test_out_dir.
        - The starting latent vector for each reference image as "z_0_reference_{img_index}.pth" in the test_out_dir.
        - The latent vector obtained after VAE inversion as "z_0_target_start_from_reference_{img_index}.pth" in the test_out_dir.
    """

    target_latent = get_latent_from_encoder(pipe.vae.encode, imgs_list[-1])
    torch.save(target_latent, os.path.join(test_out_dir, "z_0_target.pth"))

    for img_index in range(len(imgs_list[:-1])):
        starting_latent = torch.randn(target_latent.shape).cuda()
        target_latent_start_from_latent = vae_inversion_start_from_arbitrary_latent(
            encoder=pipe.vae.encode,
            decoder=pipe.vae.decode,
            image=imgs_list[-1],
            num_steps=args.num_vae_inversion_steps,
            latent_init=starting_latent.detach().clone(),
            epsilon=None,
            out_dir=osp.join(
                test_out_dir, f"vae_inversion_target_start_from_reference_{img_index}"
            ),
            scale_reconstruction=1,
            scale_l2=0,
            scale_gaussian=0,
        )

        torch.save(
            starting_latent, os.path.join(test_out_dir, f"z_0_reference_{img_index}.pth")
        )
        torch.save(
            target_latent_start_from_latent,
            os.path.join(test_out_dir, f"z_0_target_start_from_reference_{img_index}.pth"),
        )


def nti_invert(
    pipe,
    z_0,
    prompt,
    num_diffusion_inversion_steps: int = 10,
    out_dir: str = "test_out",
    is_return_both=False,
):
    """
    Perform neural text inversion (NTI) on a given latent vector and prompt.
    Args:
        pipe: The pipeline object containing the VAE, scheduler, UNet, and other components.
        z_0: The initial latent vector.
        prompt: The text prompt to guide the inversion process.
        num_diffusion_inversion_steps (int, optional): The number of diffusion inversion steps to perform. Default is 10.
        out_dir (str, optional): The directory to save the output images. Default is "test_out".
        is_return_both (bool, optional): If True, return both the final latent vector and the initial latent vector. Default is False.
    Returns:
        The final latent vector after NTI. If is_return_both is True, returns a tuple of (final latent vector, initial latent vector).
    """

    latent = z_0 * pipe.vae.scaling_factor
    text_embeddings = pipe.encode_prompt(prompt, device, 1, False, None)[0]

    pipe.scheduler.set_timesteps(num_diffusion_inversion_steps, device=device)
    all_latents = ddim_inversion(latent, text_embeddings, pipe.scheduler, pipe.unet)

    z_T, all_null_texts = null_text_inversion(
        pipe,
        all_latents,
        prompt,
        num_opt_steps=num_diffusion_inversion_steps,
        device=device,
    )
    recon_img = reconstruct(
        pipe, z_T, prompt, all_null_texts, guidance_scale=1, device=device
    )
    Image.fromarray((recon_img[0] * 255).astype(np.uint8)).save(
        f"{out_dir}/{num_diffusion_inversion_steps}_inversion_steps.png"
    )

    reconstruction = pipe.vae.decode(z_0).sample.cpu().detach()
    pp_image = pipe.image_processor.postprocess(reconstruction)
    pp_image[0].save(f"{out_dir}/input.png")

    if is_return_both:
        return z_T, z_0
    return z_T


def start_from_latents_from_other_imgs_diffusion_inversion(
    pipe,
    out_dir,
    target_ds,
    ref_img_indices,
    target_img_index,
    num_diffusion_inversion_steps=20,
    inversion_method="nti",
):
    """
    Perform diffusion inversion starting from latents of other images.
    Args:
        pipe (Pipeline): The diffusion pipeline used for inversion.
        out_dir (str): The output directory to save results.
        target_ds (Dataset): The target dataset containing images and prompts.
        ref_img_indices (list): List of reference image indices to start from.
        target_img_index (int): Index of the target image in the dataset.
        num_diffusion_inversion_steps (int, optional): Number of diffusion inversion steps. Defaults to 20.
        inversion_method (str, optional): Method of inversion, either "nti" or "renoise". Defaults to "nti".
    Returns:
        None
    """

    image_index_to_latent = {}
    for ref_index, ref_img_name in enumerate(ref_img_indices):
        image_index_to_latent[ref_img_name] = os.path.join(
            out_dir, f"z_0_target_start_from_reference_{ref_index}.pth"
        )
    image_index_to_latent["target"] = os.path.join(out_dir, "z_0_target.pth")

    _, constant_prompt = target_ds[target_img_index]

    if inversion_method == "renoise":
        pipe_inversion, pipe_inference = get_pipes(
            Model_Type.SD14, Scheduler_Type.DDIM, device=device
        )
        pipe_inversion.unet = pipe.unet
        pipe_inference.unet = pipe.unet
        pipe_inversion.text_encoder = pipe.text_encoder
        pipe_inference.text_encoder = pipe.text_encoder
        config = RunConfig(
            model_type=Model_Type.SD14,
            num_inference_steps=num_diffusion_inversion_steps,
            num_inversion_steps=num_diffusion_inversion_steps,
            num_renoise_steps=5,
            scheduler_type=Scheduler_Type.DDIM,
            perform_noise_correction=False,
            seed=42,
        )

    for image_idx, image_name in enumerate(ref_img_indices):
        z_0 = torch.load(image_index_to_latent[image_name])
        out_dir_invert_from_z0_to_zT = os.path.join(
            out_dir, f"diffusion_inversion_target_start_from_reference_{image_idx}"
        )
        os.makedirs(out_dir_invert_from_z0_to_zT, exist_ok=True)
        if inversion_method == "nti":
            z_T = nti_invert(
                pipe,
                z_0,
                constant_prompt,
                num_diffusion_inversion_steps=num_diffusion_inversion_steps,
                out_dir=out_dir_invert_from_z0_to_zT,
                is_return_both=False,
            )
        elif inversion_method == "renoise":
            reconstruction = pipe.vae.decode(z_0).sample.cpu().detach()

            pp_image = pipe.image_processor.postprocess(reconstruction)
            pp_image[0].save(f"{out_dir_invert_from_z0_to_zT}/input.png")

            img, z_T, _, _ = renoise_invert(
                init_image=pp_image,
                prompt=constant_prompt,
                cfg=config,
                pipe_inversion=pipe_inversion,
                pipe_inference=pipe_inference,
                do_reconstruction=True,
            )
            # save the reconstructed image
            img.save(
                osp.join(
                    out_dir_invert_from_z0_to_zT,
                    f"{num_diffusion_inversion_steps}_inversion_steps.png",
                )
            )

        torch.save(
            z_T, os.path.join(out_dir, f"z_T_target_start_from_reference_{image_idx}.pth")
        )

    z_0 = torch.load(image_index_to_latent["target"])
    out_dir_invert_from_z0_to_zT = os.path.join(out_dir, "diffusion_inversion_target")
    os.makedirs(out_dir_invert_from_z0_to_zT, exist_ok=True)
    if inversion_method == "nti":
        z_T = nti_invert(
            pipe,
            z_0,
            constant_prompt,
            num_diffusion_inversion_steps=num_diffusion_inversion_steps,
            out_dir=out_dir_invert_from_z0_to_zT,
            is_return_both=False,
        )
    elif inversion_method == "renoise":
        reconstruction = pipe.vae.decode(z_0).sample.cpu().detach()

        pp_image = pipe.image_processor.postprocess(reconstruction)
        pp_image[0].save(f"{out_dir_invert_from_z0_to_zT}/input.png")

        img, z_T, _, _ = renoise_invert(
            init_image=pp_image,
            prompt=constant_prompt,
            cfg=config,
            pipe_inversion=pipe_inversion,
            pipe_inference=pipe_inference,
            do_reconstruction=True,
        )
        # save the reconstructed image
        img.save(
            osp.join(
                out_dir_invert_from_z0_to_zT,
                f"{num_diffusion_inversion_steps}_inversion_steps.png",
            )
        )
    torch.save(z_T, os.path.join(out_dir, "z_T_target.pth"))


def store_z_T_of_ref_imgs_from_encoder(
    pipe,
    out_dir,
    ref_ds,
    ref_img_indices,
    num_diffusion_inversion_steps=20,
    inversion_method="nti",
):
    """
    Stores the latent representations (z_T) of reference images from an encoder.
    Args:
        pipe (Pipeline): The pipeline object containing the model and other components.
        out_dir (str): The output directory where the latent representations will be saved.
        ref_ds (Dataset): The dataset containing the reference images.
        ref_img_indices (list): A list of indices or names of the reference images to process.
        num_diffusion_inversion_steps (int, optional): The number of diffusion inversion steps. Default is 20.
        inversion_method (str, optional): The method to use for inversion. Options are "nti" or "renoise". Default is "nti".
    Returns:
        None
    """

    image_index_to_latent = {}
    for ref_index, ref_img_name in enumerate(ref_img_indices):
        image_index_to_latent[ref_img_name] = os.path.join(
            out_dir, f"z_0_reference_{ref_index}.pth"
        )

    if inversion_method == "renoise":
        pipe_inversion, pipe_inference = get_pipes(
            Model_Type.SD14, Scheduler_Type.DDIM, device=device
        )
        pipe_inversion.unet = pipe.unet
        pipe_inference.unet = pipe.unet
        pipe_inversion.text_encoder = pipe.text_encoder
        pipe_inference.text_encoder = pipe.text_encoder
        config = RunConfig(
            model_type=Model_Type.SD14,
            num_inference_steps=num_diffusion_inversion_steps,
            num_inversion_steps=num_diffusion_inversion_steps,
            num_renoise_steps=5,
            scheduler_type=Scheduler_Type.DDIM,
            perform_noise_correction=False,
            seed=42,
        )

    for image_idx, image_name in enumerate(ref_img_indices):
        z_0 = torch.load(image_index_to_latent[image_name])
        out_dir_invert_from_z0_to_zT = os.path.join(
            out_dir, f"diffusion_inversion_reference_{image_idx}"
        )
        os.makedirs(out_dir_invert_from_z0_to_zT, exist_ok=True)
        _, prompt = ref_ds[image_name]

        if inversion_method == "nti":
            z_T = nti_invert(
                pipe,
                z_0,
                prompt,
                num_diffusion_inversion_steps=num_diffusion_inversion_steps,
                out_dir=out_dir_invert_from_z0_to_zT,
                is_return_both=False,
            )
        elif inversion_method == "renoise":
            reconstruction = pipe.vae.decode(z_0).sample.cpu().detach()

            pp_image = pipe.image_processor.postprocess(reconstruction)
            pp_image[0].save(f"{out_dir_invert_from_z0_to_zT}/input.png")

            img, z_T, _, _ = renoise_invert(
                init_image=pp_image,
                prompt=prompt,
                cfg=config,
                pipe_inversion=pipe_inversion,
                pipe_inference=pipe_inference,
                do_reconstruction=True,
            )
            # save the reconstructed image
            img.save(
                osp.join(
                    out_dir_invert_from_z0_to_zT,
                    f"{num_diffusion_inversion_steps}_inversion_steps.png",
                )
            )

        torch.save(z_T, os.path.join(out_dir, f"z_T_reference_{image_idx}.pth"))


def create_collage(
    image_paths,
    out_dir,
    collage_size=(10, 10),
    image_size=(100, 100),
    out_name_prefix=None,
):
    """
    Create a 10x10 collage from a list of image paths.

    Args:
        image_paths (list): List of image file paths.
        collage_size (tuple): The number of images per row and column (default is 10x10).
        image_size (tuple): The size to resize each image to (default is 100x100 pixels).
        output_path (str): Path to save the final collage image (default is 'collage.jpg').

    """
    rows, cols = collage_size

    # Create a blank canvas for the collage
    collage_width = cols * image_size[0]
    collage_height = rows * image_size[1]
    collage_image = Image.new("RGB", (collage_width, collage_height))

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img = img.resize(image_size)

        x = (i % cols) * image_size[0]
        y = (i // cols) * image_size[1]

        collage_image.paste(img, (x, y))

    collage_image.save(osp.join(out_dir, f"{out_name_prefix}_collage.jpg"))


def create_all_collages(
    out_dir, plots_dir, num_vae_inversion_steps=2000, num_diffusion_inversion_steps=50
):
    """
    Generates and saves collages of images from VAE and diffusion inversion processes.
    Parameters:
    out_dir (str): The directory containing the output images from the inversion processes.
    plots_dir (str): The directory where the generated collages will be saved.
    num_vae_inversion_steps (int, optional): The number of steps used in the VAE inversion process. Default is 2000.
    num_diffusion_inversion_steps (int, optional): The number of steps used in the diffusion inversion process. Default is 50.
    The function performs the following steps:
    1. Collects paths to VAE inversion images and creates a collage.
    2. Collects paths to diffusion inversion images and creates a collage.
    3. Collects paths to reference images and creates a collage.
    Each collage is saved in the specified plots directory with a prefix indicating the type of images it contains.
    """

    vae_paths = [
        osp.join(
            out_dir,
            x,
            "epsilon_none_scale_rec_1_scale_l2_0_scale_gaussian_0",
            f"reconstructed_step_{num_vae_inversion_steps - 1:03d}.png",
        )
        for x in os.listdir(out_dir)
        if x.startswith("vae_inversion_target_start_from_reference_")
        and osp.isdir(
            osp.join(out_dir, x, "epsilon_none_scale_rec_1_scale_l2_0_scale_gaussian_0")
        )
    ]
    create_collage(
        vae_paths, out_dir=plots_dir, out_name_prefix="vae", collage_size=(2, 5)
    )

    diff_paths = [
        osp.join(out_dir, x, f"{num_diffusion_inversion_steps}_inversion_steps.png")
        for x in os.listdir(out_dir)
        if x.startswith("diffusion_inversion_target_start_from_reference_")
        and osp.isdir(osp.join(out_dir, x))
    ]
    create_collage(
        diff_paths, out_dir=plots_dir, out_name_prefix="diffusion", collage_size=(2, 5)
    )

    reference_paths = [
        osp.join(
            out_dir,
            x,
            "epsilon_none_scale_rec_1_scale_l2_0_scale_gaussian_0",
            "reconstructed_step_000.png",
        )
        for x in os.listdir(out_dir)
        if x.startswith("vae_inversion_target_start_from_reference_")
        and osp.isdir(
            osp.join(out_dir, x, "epsilon_none_scale_rec_1_scale_l2_0_scale_gaussian_0")
        )
    ]
    create_collage(
        reference_paths, out_dir=plots_dir, out_name_prefix="reference", collage_size=(2, 5)
    )


def analyze_z_T(
    args,
    ref_img_indices,
    target_img_index,
    out_dir,
    plots_dir,
    original_images_dir,
    control_group=False,
    target_prompt=None,
):
    """
    Analyzes the latent representations and reconstruction quality of images.
    Args:
        args: Argument parser object containing various parameters.
        ref_img_indices (list): List of reference image indices.
        target_img_index (int): Index of the target image.
        out_dir (str): Directory where output files are stored.
        plots_dir (str): Directory where plots are saved.
        original_images_dir (str): Directory containing original images.
        control_group (bool, optional): Flag indicating if the analysis is for the control group. Defaults to False.
        target_prompt (str, optional): Target prompt for calculating CLIP similarity. Defaults to None.
    Returns:
        dict: A dictionary containing the following keys:
            - "nll_values" (list): Negative log-likelihood values for the reference images.
            - "psnrs" (list): PSNR values for the reconstructed images.
            - "vae_psnrs" (list): PSNR values for the VAE reconstructed images.
            - "mean_distance_cos" (float): Mean cosine distance between latent representations.
            - "std_distance_cos" (float): Standard deviation of cosine distances.
            - "mean_distance_to_target_cos" (float): Mean cosine distance to the target latent representation.
            - "std_distance_to_target_cos" (float): Standard deviation of cosine distances to the target.
            - "mean_distance_euc" (float): Mean Euclidean distance between latent representations.
            - "std_distance_euc" (float): Standard deviation of Euclidean distances.
            - "mean_distance_to_target_euc" (float): Mean Euclidean distance to the target latent representation.
            - "std_distance_to_target_euc" (float): Standard deviation of Euclidean distances to the target.
            - "clip_similarities" (list): CLIP similarities between reconstructed images and the target prompt.
    """

    prefix = "z_T_target_start_from_reference_" if not control_group else "z_T_reference_"
    ref_image_name_to_latent = {
        f"image{image_idx}": torch.load(
            os.path.join(out_dir, f"{prefix}{image_idx}.pth")
        )
        for image_idx, img_name in enumerate(ref_img_indices)
    }

    nll_values = []
    for k in ref_image_name_to_latent:
        nll_values.append(tensor_to_nll(ref_image_name_to_latent[k]))
    if not control_group:
        analyze_nll_values(
            nll_values, plots_dir=plots_dir, show_figures=args.show_figures
        )
    else:
        analyze_nll_values(
            nll_values,
            plots_dir=plots_dir,
            out_name_prefix="control_group_",
            show_figures=args.show_figures,
        )

    target_latent = torch.load(os.path.join(out_dir, "z_T_target.pth"))
    all_latents = list(ref_image_name_to_latent.values()) + [target_latent]

    if not control_group:
        (
            mean_distance_euc,
            std_distance_euc,
            mean_distance_to_target_euc,
            std_distance_to_target_euc,
        ) = plot_latent_distances(
            all_latents,
            metric="euclidean",
            plots_dir=plots_dir,
            show_figures=args.show_figures,
        )
        (
            mean_distance_cos,
            std_distance_cos,
            mean_distance_to_target_cos,
            std_distance_to_target_cos,
        ) = plot_latent_distances(
            all_latents,
            metric="cosine",
            plots_dir=plots_dir,
            show_figures=args.show_figures,
        )
    else:
        (
            mean_distance_euc,
            std_distance_euc,
            mean_distance_to_target_euc,
            std_distance_to_target_euc,
        ) = plot_latent_distances(
            all_latents,
            metric="euclidean",
            plots_dir=plots_dir,
            index="control_group_",
            show_figures=args.show_figures,
        )
        (
            mean_distance_cos,
            std_distance_cos,
            mean_distance_to_target_cos,
            std_distance_to_target_cos,
        ) = plot_latent_distances(
            all_latents,
            metric="cosine",
            plots_dir=plots_dir,
            index="control_group_",
            show_figures=args.show_figures,
        )

    psnrs = []
    vae_psnrs = []
    clip_similarities = []
    for ref_image_idx, ref_image_name in enumerate(ref_img_indices):
        if not control_group:
            original_image_path = os.path.join(
                original_images_dir, f"target_{target_img_index}.png"
            )
            reconstructed_image_path = os.path.join(
                out_dir,
                f"diffusion_inversion_target_start_from_reference_{ref_image_idx}",
                f"{args.num_diffusion_inversion_steps}_inversion_steps.png",
            )
            vae_reconstructed_image_path = os.path.join(
                out_dir,
                f"vae_inversion_target_start_from_reference_{ref_image_idx}",
                "epsilon_none_scale_rec_1_scale_l2_0_scale_gaussian_0",
                f"reconstructed_step_{args.num_vae_inversion_steps - 1:03d}.png",
            )
            if target_prompt:
                clip_similarities.append(
                    clip_similarity(reconstructed_image_path, target_prompt)
                )
        else:
            original_image_path = osp.join(
                original_images_dir, f"reference_{ref_image_name}.png"
            )
            reconstructed_image_path = os.path.join(
                out_dir,
                f"diffusion_inversion_reference_{ref_image_idx}",
                f"{args.num_diffusion_inversion_steps}_inversion_steps.png",
            )
            vae_reconstructed_image_path = os.path.join(
                out_dir,
                f"vae_inversion_reference_{ref_image_idx}",
                "scale_rec_1_scale_l2_0_scale_gaussian_0",
                f"reconstructed_step_{args.num_vae_inversion_steps - 1:03d}.png",
            )
        original_image = Image.open(original_image_path)
        reconstructed_image = Image.open(reconstructed_image_path)
        vae_reconstructed_image = Image.open(vae_reconstructed_image_path)
        psnr = calculate_psnr(original_image, reconstructed_image)
        vae_psnr = calculate_psnr(original_image, vae_reconstructed_image)
        psnrs.append(psnr)
        vae_psnrs.append(vae_psnr)
    if not control_group:
        analyze_psnr_values(psnrs, plots_dir, show_figures=args.show_figures)
        analyze_psnr_values(
            vae_psnrs, plots_dir, show_figures=args.show_figures, out_name_prefix="vae_"
        )
    else:
        analyze_psnr_values(
            psnrs,
            plots_dir,
            out_name_prefix="control_group_",
            show_figures=args.show_figures,
        )
        analyze_psnr_values(
            vae_psnrs,
            plots_dir,
            out_name_prefix="vae_control_group_",
            show_figures=args.show_figures,
        )

    results = {
        "nll_values": nll_values,
        "psnrs": psnrs,
        "vae_psnrs": vae_psnrs,
        "mean_distance_cos": mean_distance_cos,
        "std_distance_cos": std_distance_cos,
        "mean_distance_to_target_cos": mean_distance_to_target_cos,
        "std_distance_to_target_cos": std_distance_to_target_cos,
        "mean_distance_euc": mean_distance_euc,
        "std_distance_euc": std_distance_euc,
        "mean_distance_to_target_euc": mean_distance_to_target_euc,
        "std_distance_to_target_euc": std_distance_to_target_euc,
        "clip_similarities": clip_similarities,
    }

    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def write_results_into_json(results_target, results_control_group):
    psnr_json = {
        "tsi_goal_psnrs": results_target["psnrs"],
        "vae_goal_psnrs": results_target["vae_psnrs"],
        "tsi_control_group_psnrs": results_control_group["psnrs"],
        "vae_control_group_psnrs": results_control_group["vae_psnrs"],
        "mean_tsi_goal_psnrs": np.mean(results_target["psnrs"]),
        "mean_vae_goal_psnrs": np.mean(results_target["vae_psnrs"]),
        "mean_tsi_control_group_psnrs": np.mean(results_control_group["psnrs"]),
        "mean_vae_control_group_psnrs": np.mean(results_control_group["vae_psnrs"]),
    }

    nll_json = {
        "goal_nlls": results_target["nll_values"],
        "control_group_nlls": results_control_group["nll_values"],
        "mean_goal_nlls": np.mean(results_target["nll_values"]),
        "std_goal_nlls": np.std(results_target["nll_values"]),
        "mean_control_group_nlls": np.mean(results_control_group["nll_values"]),
        "std_control_group_nlls": np.std(results_control_group["nll_values"]),
    }

    distances_json = {
        "goal_pairwise_mean_distance_euc": results_target["mean_distance_euc"],
        "goal_pairwise_std_distance_euc": results_target["std_distance_euc"],
        "goal_to_target_mean_distance_euc": results_target[
            "mean_distance_to_target_euc"
        ],
        "goal_to_target_std_distance_euc": results_target["std_distance_to_target_euc"],
        "goal_pairwise_mean_distance_cos": results_target["mean_distance_cos"],
        "goal_pairwise_std_distance_cos": results_target["std_distance_cos"],
        "goal_to_target_mean_distance_cos": results_target[
            "mean_distance_to_target_cos"
        ],
        "goal_to_target_std_distance_cos": results_target["std_distance_to_target_cos"],
    }

    detection_json = {"success_rate": results_target["success_rate"]}

    emd_json = results_target["emd_scores"]

    clip_json = {
        "clip_similarities": results_target["clip_similarities"],
        "mean_clip_similarity": np.mean(results_target["clip_similarities"]),
    }

    out_json = {}
    out_json["psnr"] = psnr_json
    out_json["nll"] = nll_json
    out_json["distances"] = distances_json
    out_json["detection"] = detection_json
    out_json["emd"] = emd_json
    out_json["clip"] = clip_json

    return out_json


def average_targets_json(out_json):
    avg_json = {}
    all_tsi_goal_psnrs = []
    all_emd_scores = {}
    all_clip_similarities = []
    all_nll_values = []
    all_mean_distances_euc = []
    all_mean_distances_cos = []
    all_mean_distances_to_target_euc = []
    all_mean_distances_to_target_cos = []
    all_detection_success_rates = []

    for target_json in out_json.values():
        all_tsi_goal_psnrs.extend(target_json["psnr"]["tsi_goal_psnrs"])

        all_nll_values.extend(target_json["nll"]["goal_nlls"])

        all_mean_distances_euc.append(
            target_json["distances"]["goal_pairwise_mean_distance_euc"]
        )
        all_mean_distances_cos.append(
            target_json["distances"]["goal_pairwise_mean_distance_cos"]
        )
        all_mean_distances_to_target_euc.append(
            target_json["distances"]["goal_to_target_mean_distance_euc"]
        )
        all_mean_distances_to_target_cos.append(
            target_json["distances"]["goal_to_target_mean_distance_cos"]
        )

        all_detection_success_rates.append(target_json["detection"]["success_rate"])

        # all_emd_scores.append(target_json['emd']['emd_score'])
        for key, value in target_json["emd"].items():
            if key not in all_emd_scores:
                all_emd_scores[key] = []
            all_emd_scores[key].append(value)

        all_clip_similarities.extend(target_json["clip"]["clip_similarities"])

    avg_json["psnr"] = {
        "all_tsi_goal_psnrs": all_tsi_goal_psnrs,
        "avg_tsi_goal_psnrs": np.mean(all_tsi_goal_psnrs),
    }
    avg_json["nll"] = {
        "all_goal_nlls": all_nll_values,
        "avg_goal_nlls": np.mean(all_nll_values),
    }
    avg_json["distances"] = {
        "avg_pairwise_distance_euc": np.mean(all_mean_distances_euc),
        "avg_pairwise_distance_cos": np.mean(all_mean_distances_cos),
        "avg_distance_to_target_euc": np.mean(all_mean_distances_to_target_euc),
        "avg_distance_to_target_cos": np.mean(all_mean_distances_to_target_cos),
    }
    avg_json["detection"] = {"avg_success_rate": np.mean(all_detection_success_rates)}
    avg_json["emd"] = {
        f"avg_{key}": np.mean(value) for key, value in all_emd_scores.items()
    }
    avg_json["clip"] = {"avg_clip_similarity": np.mean(all_clip_similarities)}
    return avg_json


if __name__ == "__main__":
    args = validate_and_get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model_id = args.model_base
    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_ds = CocoCaptions17Paths(args.reference_dataset_root, train=False)
    target_ds = CaptionsPaths(args.dataset_root, train=False)

    with open(osp.join(args.out_dir, "args.json"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    out_json = {}

    all_psnrs = []
    all_vae_psnrs = []

    all_control_group_psnrs = []
    all_control_group_vae_psnrs = []

    for tid, target_index in enumerate(args.image_indices):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, safety_checker=None, scheduler=scheduler
        )

        if args.ablated_model:
            pipe.unet.load_state_dict(torch.load(args.ablated_model))

        if args.ablated_text_encoder:
            assert args.ablated_concept_name, "Ablated concept name must be provided"
            pipe.text_encoder = CLIPTextModel.from_pretrained(
                args.ablated_text_encoder,
                subfolder=f"{args.ablated_concept_name}_unlearned",
            )
        pipe = pipe.to(device)

        ref_img_indices = []
        imgs_list = []

        out_dir = osp.join(args.out_dir, f"target_img_index_{tid}")

        for img_num in range(args.num_ref_images):
            ref_index = tid * args.num_ref_images + img_num
            image_path, prompt = ref_ds[ref_index]
            image = Image.open(image_path)
            image = image.resize((512, 512))
            image_path = osp.join(out_dir, "original", f"reference_{ref_index}.png")
            os.makedirs(osp.dirname(image_path), exist_ok=True)
            image.save(image_path)
            imgs_list.append(image)
            ref_img_indices.append(ref_index)

        image_path, prompt = target_ds[target_index]
        target_image = Image.open(image_path)
        target_image = target_image.resize((512, 512))
        image_path = osp.join(out_dir, "original", f"target_{target_index}.png")
        os.makedirs(osp.dirname(image_path), exist_ok=True)
        target_image.save(image_path)
        imgs_list.append(target_image)

        test_out_dir = os.path.join(
            out_dir,
            f"hard_to_forget_vae_{args.num_vae_inversion_steps}_diffusion_{args.num_diffusion_inversion_steps}",
        )
        os.makedirs(test_out_dir, exist_ok=True)
        plots_dir = osp.join(test_out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        if not args.analyze_only:
            start_from_latents_from_other_imgs_vae_inversion(
                pipe, args, imgs_list, test_out_dir
            )
            # sample_latents_around_target(pipe, args, imgs_list, test_out_dir)
            start_from_latents_from_other_imgs_diffusion_inversion(
                pipe,
                test_out_dir,
                target_ds,
                ref_img_indices,
                target_index,
                num_diffusion_inversion_steps=args.num_diffusion_inversion_steps,
                inversion_method=args.diffusion_inversion_method,
            )
            store_z_T_of_ref_imgs_from_encoder(
                pipe,
                test_out_dir,
                ref_ds,
                ref_img_indices,
                num_diffusion_inversion_steps=args.num_diffusion_inversion_steps,
                inversion_method=args.diffusion_inversion_method,
            )
        results_target = analyze_z_T(
            args,
            ref_img_indices,
            target_index,
            test_out_dir,
            plots_dir,
            osp.join(out_dir, "original"),
            target_prompt=prompt,
        )
        results_control_group = analyze_z_T(
            args,
            ref_img_indices,
            target_index,
            test_out_dir,
            plots_dir,
            osp.join(out_dir, "original"),
            control_group=True,
        )
        create_all_collages(
            test_out_dir,
            plots_dir,
            num_diffusion_inversion_steps=args.num_diffusion_inversion_steps,
            num_vae_inversion_steps=args.num_vae_inversion_steps,
        )
        emd_scores = analyze_goal_reference_normal_nll_values(
            results_target["nll_values"],
            results_control_group["nll_values"],
            label_1=r"NLL of TSI of references $\longrightarrow$ target",
            label_2=r"NLL of TSI of references",
            label_3=r"Random normal samples",
            plots_dir=plots_dir,
        )
        success_rate = detect_concept_post_analysis(out_dir, with_vae=True, concept_name=args.ablated_concept_name)

        results_target["emd_scores"] = emd_scores
        results_target["success_rate"] = success_rate

        all_psnrs.extend(results_target["psnrs"])
        all_vae_psnrs.extend(results_target["vae_psnrs"])
        all_control_group_psnrs.extend(results_control_group["psnrs"])
        all_control_group_vae_psnrs.extend(results_control_group["vae_psnrs"])

        out_json_target_img = write_results_into_json(
            results_target, results_control_group
        )
        out_json_target_img["reference_indices"] = ref_img_indices
        out_json_target_img["target_index"] = target_index

        out_json[target_index] = out_json_target_img

        with open(osp.join(args.out_dir, "results.json"), "w") as f:
            json.dump(out_json, f, cls=NumpyEncoder)

    all_target_imgs_plots_dir = osp.join(args.out_dir, "plots")
    os.makedirs(all_target_imgs_plots_dir, exist_ok=True)
    analyze_psnr_values(
        all_psnrs,
        all_target_imgs_plots_dir,
        out_name_prefix="all_",
        show_figures=args.show_figures,
    )
    analyze_psnr_values(
        all_vae_psnrs,
        all_target_imgs_plots_dir,
        out_name_prefix="vae_all_",
        show_figures=args.show_figures,
    )
    analyze_psnr_values(
        all_control_group_psnrs,
        all_target_imgs_plots_dir,
        out_name_prefix="control_group_all_",
        show_figures=args.show_figures,
    )
    analyze_psnr_values(
        all_control_group_vae_psnrs,
        all_target_imgs_plots_dir,
        out_name_prefix="control_group_vae_all_",
        show_figures=args.show_figures,
    )
    out_json["average"] = average_targets_json(out_json)
    with open(osp.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out_json, f, cls=NumpyEncoder)
