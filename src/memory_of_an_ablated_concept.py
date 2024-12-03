import argparse
import os
import yaml
import torch
from easydict import EasyDict
import os.path as osp
from vae_inversion_tools import get_latent_from_encoder
from utils import tensor_to_nll, calculate_psnr
from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel
from PIL import Image
from concept_datasets import CocoCaptions17Paths, CaptionsPaths
import json
from nti import ddim_inversion, null_text_inversion, reconstruct
from analyze_latents import (
    analyze_psnr_values,
    analyze_nll_values,
    analyze_goal_reference_normal_nll_values,
)
from detect_concepts import detect_concept_post_analysis
from analysis_utils import clip_similarity
import numpy as np
from renoise.main import run as renoise_invert
from renoise.eunms import Model_Type, Scheduler_Type
from renoise.utils.enums_utils import get_pipes
from renoise.config import RunConfig


def validate_and_get_args():
    """
    Validate and parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate unlikely images",
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
        default=None,
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
        "--image_type", type=str, default="coco", help="Type of image to generate"
    )
    parser.add_argument(
        "--diffusion_inversion_method",
        type=str,
        default="renoise",
        help="Diffusion inversion method. Valid options are 'nti' and 'renoise'",
    )

    parser.add_argument(
        "--num_ref_images", type=int, default=None, help="Number of reference images"
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
        num_diffusion_inversion_steps (int, optional): Number of diffusion inversion steps. Default is 10.
        out_dir (str, optional): Directory to save the output images. Default is "test_out".
        is_return_both (bool, optional): If True, return both the final latent vector and the initial latent vector. Default is False.
    Returns:
        The final latent vector after inversion. If is_return_both is True, returns a tuple of (final latent vector, initial latent vector).
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


def diffusion_inversion(
    pipe,
    out_dir,
    target_ds,
    target_imgs_indices,
    num_diffusion_inversion_steps=20,
    inversion_method="nti",
):
    """
    Perform diffusion inversion on a set of target images.
    Args:
        pipe (Pipeline): The diffusion pipeline object containing the model and other components.
        out_dir (str): The output directory where results will be saved.
        target_ds (Dataset): The dataset containing target images and their corresponding prompts.
        target_imgs_indices (list): List of indices or names of target images to be processed.
        num_diffusion_inversion_steps (int, optional): Number of diffusion inversion steps. Default is 20.
        inversion_method (str, optional): The method to use for inversion. Options are "nti" and "renoise". Default is "nti".
    Returns:
        None
    """
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
    for image_idx, image_name in enumerate(target_imgs_indices):
        target_img, constant_prompt = target_ds[image_name]
        target_img = Image.open(target_img)
        target_img = target_img.resize((512, 512))

        z_0 = get_latent_from_encoder(pipe.vae.encode, target_img)
        torch.save(z_0, os.path.join(out_dir, f"z_0_target_{image_idx}.pth"))

        out_dir_invert_from_z0_to_zT = os.path.join(
            out_dir, f"diffusion_inversion_target_{image_idx}"
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
            target_img.save(f"{out_dir_invert_from_z0_to_zT}/input.png")

            img, z_T, _, _ = renoise_invert(
                init_image=target_img,
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

        torch.save(z_T, os.path.join(out_dir, f"z_T_target_{image_idx}.pth"))


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
        pipe (Pipeline): The pipeline object containing the VAE and other components.
        out_dir (str): The output directory where the latent representations and images will be saved.
        ref_ds (Dataset): The dataset containing reference images and their corresponding prompts.
        ref_img_indices (list): List of indices or names of reference images to process.
        num_diffusion_inversion_steps (int, optional): Number of diffusion inversion steps. Default is 20.
        inversion_method (str, optional): The method to use for inversion. Options are "nti" or "renoise". Default is "nti".
    Returns:
        None
    """
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
        ref_img, prompt = ref_ds[image_name]
        ref_img = Image.open(ref_img)
        ref_img = ref_img.resize((512, 512))

        z_0 = get_latent_from_encoder(pipe.vae.encode, ref_img)
        torch.save(z_0, os.path.join(out_dir, f"z_0_ref_{image_idx}.pth"))

        out_dir_invert_from_z0_to_zT = os.path.join(
            out_dir, f"diffusion_inversion_ref_{image_idx}"
        )
        os.makedirs(out_dir_invert_from_z0_to_zT, exist_ok=True)

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
            ref_img.save(f"{out_dir_invert_from_z0_to_zT}/input.png")

            img, z_T, _, _ = renoise_invert(
                init_image=ref_img,
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

        torch.save(z_T, os.path.join(out_dir, f"z_T_ref_{image_idx}.pth"))


def create_collage(
    image_paths,
    out_dir,
    collage_size=(10, 10),
    image_size=(100, 100),
    out_name_prefix=None,
):
    """
    Create a 10x10 collage from a list of image paths and save the collage image.

    Args:
        image_paths (list): List of image file paths.
        collage_size (tuple): The number of images per row and column (default is 10x10).
        image_size (tuple): The size to resize each image to (default is 100x100 pixels).
        output_path (str): Path to save the final collage image (default is 'collage.jpg').

    """
    # Number of rows and columns
    rows, cols = collage_size

    # Create a blank canvas for the collage
    collage_width = cols * image_size[0]
    collage_height = rows * image_size[1]
    collage_image = Image.new("RGB", (collage_width, collage_height))

    # Loop through the list of image paths and place them in the collage
    for i, img_path in enumerate(image_paths):
        # Open the image
        img = Image.open(img_path)
        # Resize the image
        img = img.resize(image_size)

        # Calculate the position to paste the image in the collage
        x = (i % cols) * image_size[0]
        y = (i // cols) * image_size[1]

        # Paste the image onto the collage
        collage_image.paste(img, (x, y))

    # Save the collage image
    collage_image.save(osp.join(out_dir, f"{out_name_prefix}_collage.jpg"))


def create_all_collages(
    out_dir, plots_dir, num_diffusion_inversion_steps=50, max_images_per_collage=10
):
    """
    Creates collages of images generated from diffusion inversion steps and their corresponding input images.
    Parameters:
    out_dir (str): The directory containing the output images from the diffusion inversion process.
    plots_dir (str): The directory where the generated collages will be saved.
    num_diffusion_inversion_steps (int, optional): The number of diffusion inversion steps used to generate the images. Default is 50.
    max_images_per_collage (int, optional): The maximum number of images to include in each collage. Default is 10.
    Returns:
    None
    """

    diff_paths = [
        osp.join(out_dir, x, f"{num_diffusion_inversion_steps}_inversion_steps.png")
        for x in os.listdir(out_dir)
        if x.startswith("diffusion_inversion_target_")
        and osp.isdir(osp.join(out_dir, x))
    ]
    for i in range(0, len(diff_paths), max_images_per_collage):
        collage_data = diff_paths[i : i + max_images_per_collage]
        create_collage(
            collage_data,
            out_dir=plots_dir,
            out_name_prefix=f"diffusion_{i}_",
            collage_size=(2, 5),
        )

    ref_paths = [
        osp.join(out_dir, x, "input.png")
        for x in os.listdir(out_dir)
        if x.startswith("diffusion_inversion_target_")
        and osp.isdir(osp.join(out_dir, x))
    ]
    for i in range(0, len(ref_paths), max_images_per_collage):
        collage_data = ref_paths[i : i + max_images_per_collage]
        create_collage(
            collage_data,
            out_dir=plots_dir,
            out_name_prefix=f"ref_{i}_",
            collage_size=(2, 5),
        )


def analyze_z_T(
    args,
    ref_img_indices,
    target_img_indices,
    out_dir,
    plots_dir,
    original_images_dir,
    control_group=False,
    target_prompts=None,
):
    """
    Analyzes the latent representations and reconstructed images for a set of ref or target images.
    Args:
        args: Arguments containing various settings and configurations.
        ref_img_indices (list): List of indices for ref images.
        target_img_indices (list): List of indices for target images.
        out_dir (str): Directory where output files are stored.
        plots_dir (str): Directory where plots will be saved.
        original_images_dir (str): Directory containing the original images.
        control_group (bool, optional): If True, analyze ref images; otherwise, analyze target images. Default is False.
        target_prompts (list, optional): List of prompts for target images to calculate CLIP similarities. Default is None.
    Returns:
        dict: A dictionary containing the following keys:
            - "nll_values" (list): Negative log-likelihood values for the latent representations.
            - "psnrs" (list): Peak signal-to-noise ratio values for the reconstructed images.
            - "clip_similarities" (list): CLIP similarity scores for the reconstructed images (only for target images).
    """
    prefix = "z_T_target_" if not control_group else "z_T_ref_"
    if not control_group:
        indices = target_img_indices
    else:
        indices = ref_img_indices
    latents = [
        torch.load(os.path.join(out_dir, f"{prefix}{image_idx}.pth"))
        for image_idx in indices
    ]

    nll_values = []
    # need to plot something nicer
    for k in latents:
        nll_values.append(tensor_to_nll(k))
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

    psnrs = []
    clip_similarities = []
    for image_idx, image_name in enumerate(indices):
        if not control_group:
            original_image_path = os.path.join(
                original_images_dir, f"target_{image_idx}.png"
            )
            reconstructed_image_path = os.path.join(
                out_dir,
                f"diffusion_inversion_target_{image_idx}",
                f"{args.num_diffusion_inversion_steps}_inversion_steps.png",
            )
            if target_prompts:
                clip_similarities.append(
                    clip_similarity(reconstructed_image_path, target_prompts[image_idx])
                )
        else:
            original_image_path = osp.join(
                original_images_dir, f"ref_{image_idx}.png"
            )
            reconstructed_image_path = os.path.join(
                out_dir,
                f"diffusion_inversion_ref_{image_idx}",
                f"{args.num_diffusion_inversion_steps}_inversion_steps.png",
            )
        original_image = Image.open(original_image_path)
        reconstructed_image = Image.open(reconstructed_image_path)
        psnr = calculate_psnr(original_image, reconstructed_image)
        psnrs.append(psnr)
    if not control_group:
        analyze_psnr_values(psnrs, plots_dir, show_figures=args.show_figures)
    else:
        analyze_psnr_values(
            psnrs,
            plots_dir,
            out_name_prefix="control_group_",
            show_figures=args.show_figures,
        )

    results = {
        "nll_values": nll_values,
        "psnrs": psnrs,
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
        "memory_goal_psnrs": results_target["psnrs"],
        "memory_control_group_psnrs": results_control_group["psnrs"],
        "mean_memory_goal_psnrs": np.mean(results_target["psnrs"]),
        "mean_memory_control_group_psnrs": np.mean(results_control_group["psnrs"]),
    }

    nll_json = {
        "goal_nlls": results_target["nll_values"],
        "control_group_nlls": results_control_group["nll_values"],
        "mean_goal_nlls": np.mean(results_target["nll_values"]),
        "std_goal_nlls": np.std(results_target["nll_values"]),
        "mean_control_group_nlls": np.mean(results_control_group["nll_values"]),
        "std_control_group_nlls": np.std(results_control_group["nll_values"]),
    }

    detection_json = results_target["success_rate"]
    emd_json = results_target["emd_scores"]

    clip_json = {
        "clip_similarities": results_target["clip_similarities"],
        "mean_clip_similarity": np.mean(results_target["clip_similarities"]),
    }

    out_json = {}
    out_json["psnr"] = psnr_json
    out_json["nll"] = nll_json
    out_json["detection"] = detection_json
    out_json["emd"] = emd_json
    out_json["clip"] = clip_json

    return out_json


if __name__ == "__main__":
    # Load the pre-trained Stable Diffusion model
    args = validate_and_get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model_id = args.model_base
    scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_ds = CocoCaptions17Paths(args.reference_dataset_root, train=False)
    target_ds = CaptionsPaths(args.dataset_root, train=False)

    if args.image_indices is None:
        args.image_indices = list(range(len(target_ds)))
    if args.num_ref_images is None:
        args.num_ref_images = len(target_ds)
    
    with open(osp.join(args.out_dir, "args.json"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    out_json = {}
    all_psnrs = []
    all_control_group_psnrs = []

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
    ref_imgs_list = []
    target_imgs_list = []
    target_prompts = []
    out_dir = args.out_dir

    for img_num in range(args.num_ref_images):
        ref_index = img_num
        image_path, prompt = ref_ds[ref_index]
        image = Image.open(image_path)
        image = image.resize((512, 512))
        image_path = osp.join(out_dir, "original", f"ref_{ref_index}.png")
        os.makedirs(osp.dirname(image_path), exist_ok=True)
        image.save(image_path)
        ref_imgs_list.append(image)
        ref_img_indices.append(ref_index)

    for target_index in args.image_indices:
        image_path, prompt = target_ds[target_index]
        image = Image.open(image_path)
        image = image.resize((512, 512))
        image_path = osp.join(out_dir, "original", f"target_{target_index}.png")
        os.makedirs(osp.dirname(image_path), exist_ok=True)
        image.save(image_path)
        target_imgs_list.append(image)
        target_prompts.append(prompt)

    test_out_dir = os.path.join(
        out_dir, f"hard_to_forget_diffusion_{args.num_diffusion_inversion_steps}"
    )
    os.makedirs(test_out_dir, exist_ok=True)
    plots_dir = osp.join(test_out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    if not args.analyze_only:
        diffusion_inversion(
            pipe,
            test_out_dir,
            target_ds,
            args.image_indices,
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
        args.image_indices,
        test_out_dir,
        plots_dir,
        osp.join(out_dir, "original"),
        target_prompts=target_prompts,
    )
    results_control_group = analyze_z_T(
        args,
        ref_img_indices,
        args.image_indices,
        test_out_dir,
        plots_dir,
        osp.join(out_dir, "original"),
        control_group=True,
    )
    create_all_collages(
        test_out_dir,
        plots_dir,
        num_diffusion_inversion_steps=args.num_diffusion_inversion_steps,
    )
    emd_scores = analyze_goal_reference_normal_nll_values(
        results_target["nll_values"],
        results_control_group["nll_values"],
        label_1=r"NLL of diffusion inversion of targets",
        label_2=r"NLL of diffusion inversion of references",
        label_3=r"Random normal samples",
        plots_dir=plots_dir,
    )
    post_analysis_success_rate = detect_concept_post_analysis(out_dir, with_vae=False, concept_name=args.ablated_concept_name)
    results_target["success_rate"] = {
        "post_analysis_success_rate": post_analysis_success_rate,
    }
    results_target["emd_scores"] = emd_scores
    all_psnrs.extend(results_target["psnrs"])
    all_control_group_psnrs.extend(results_control_group["psnrs"])

    out_json = write_results_into_json(results_target, results_control_group)
    out_json["ref_indices"] = ref_img_indices
    out_json["target_indices"] = args.image_indices

    with open(osp.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out_json, f, cls=NumpyEncoder)
