import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os.path as osp
import torch
import os
from src.utils import preprocess_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_latent_from_encoder(encoder, img):
    image = img.resize((512, 512))
    pp_image = preprocess_image(image)
    z_0_from_encoder = encoder(pp_image.to(device)).latent_dist
    return z_0_from_encoder.mean.detach().clone()

def sample_latent_from_encoder(encoder, img):
    image = img.resize((512, 512))
    pp_image = preprocess_image(image)
    z_0_from_encoder = encoder(pp_image.to(device)).latent_dist
    # close:
    # return z_0_from_encoder.sample().detach().clone()
    # far:
    latent = z_0_from_encoder.mean.detach().clone()
    noise = torch.randn(latent.shape).cuda()  # Gaussian noise
    norm = torch.norm(noise, dim=1, keepdim=True)  # Compute L2 norm for each sample
    scaled_noise = (noise / norm) * 10  # Scale to the desired distance
    return latent + scaled_noise

def vae_inversion_start_from_encoder_latent(encoder, decoder, image: Image, num_steps: int = 1000, out_dir: str = 'test_out',
                  scale_reconstruction=10, scale_l2=0.1, scale_gaussian=0.001) -> torch.tensor:
    image = image.resize((512, 512))
    pp_image = preprocess_image(image)
    out_dir = osp.join(out_dir, f"scale_rec_{scale_reconstruction}_scale_l2_{scale_l2}_scale_gaussian_{scale_gaussian}")
    os.makedirs(out_dir, exist_ok=True)
    # validate that the preprocessing went well:
    result = decoder(encoder(pp_image.to(device)).latent_dist.mean).sample
    result_rescaled = (result / 2 + 0.5).clamp(0, 1)
    image_to_show = result_rescaled.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    Image.fromarray((image_to_show[0]*255).astype(np.uint8)).save(f'{out_dir}/decode_the_mean_of_encoder_out.png')

    z_0_from_encoder = encoder(pp_image.to(device)).latent_dist
    target_mean = z_0_from_encoder.mean.detach()
    target_var = z_0_from_encoder.var.detach()
    target_image_scaled_to_01 = torch.from_numpy(np.array(image) * 1.0 / 255.0).to(device)
    reconstruction_loss = nn.MSELoss()

    z = torch.zeros([1, 4, 64, 64], requires_grad=True, device=device)
    z.data = target_mean

    optimizer = optim.Adam([z], lr=1e-2)
    losses = []
    psnrs = []
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.1) 
    from tqdm import tqdm
    pbar = tqdm(range(num_steps))
    for step in pbar:
        optimizer.zero_grad()

        # Decode the latent variable to get the reconstructed image
        reconstructed_image = decoder(z).sample
        reconstructed_image_rescaled = (reconstructed_image / 2 + 0.5).type(torch.float32)
        
        # Compute the loss between the original image and the reconstructed image
        reconstruction_loss_value = reconstruction_loss(reconstructed_image_rescaled, target_image_scaled_to_01.permute(2, 0, 1).unsqueeze(0).type(torch.float32))
        loss = (
            scale_reconstruction * reconstruction_loss_value +
            scale_l2 * ((z - target_mean)**2).mean() +
            scale_gaussian * nn.GaussianNLLLoss()(z, target_mean, target_var)
        )

        # Backpropagation
        loss.backward()
        losses.append(loss.item())
        psnr = 10 * torch.log10(1 / torch.mean((reconstructed_image_rescaled - target_image_scaled_to_01.permute(2, 0, 1).unsqueeze(0).type(torch.float32)) ** 2)).item()
        psnrs.append(psnr)
        pbar.set_description(f"Loss: {loss.item():.6f} | PSNR: {psnr:.4f}")

        # Update the latent variable
        optimizer.step()

        # Update the learning rate using the scheduler
        # scheduler.step()

        # Optionally print the loss
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item()}")
            print(
                f"reconstruction loss: {reconstruction_loss_value.item():.3f}, "
                f"L2: {((z - target_mean)**2).mean().item():.3f} , "
                f"Gaussian NLL: {nn.GaussianNLLLoss()(z, target_mean, target_var):.3f}"
            )
            rec2 = (
                reconstructed_image_rescaled[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                * 255
            ).astype(np.uint8)
            # save the reconstruction
            Image.fromarray(rec2).save(
                f"{out_dir}/reconstructed_step_{step:03d}.png"
            )
    with open(f"{out_dir}/vae_inversion_stats.json", "w") as f:
        json.dump({"LOSS": losses, "PSNR": psnrs}, f, indent=4)
    rec2 = (
                reconstructed_image_rescaled[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                * 255
            ).astype(np.uint8)
    # save the reconstruction
    Image.fromarray(rec2).save(
                f"{out_dir}/reconstructed_step_{step:03d}.png"
            )
    
    return z.detach()



def vae_inversion_start_from_arbitrary_latent(encoder, decoder, image: Image, latent_init, epsilon=None, num_steps: int = 1000, out_dir: str = 'test_out',
                                              scale_reconstruction=10, scale_l2=0.1, scale_gaussian=0.001) -> torch.tensor:
    image = image.resize((512, 512))
    pp_image = preprocess_image(image)
    out_dir = osp.join(out_dir, f"epsilon_{epsilon if epsilon else 'none'}_scale_rec_{scale_reconstruction}_scale_l2_{scale_l2}_scale_gaussian_{scale_gaussian}")
    os.makedirs(out_dir, exist_ok=True)
    # validate that the preprocessing went well:
    result = decoder(encoder(pp_image.to(device)).latent_dist.mean).sample
    result_rescaled = (result / 2 + 0.5).clamp(0, 1)
    image_to_show = result_rescaled.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    Image.fromarray((image_to_show[0]*255).astype(np.uint8)).save(f'{out_dir}/decode_the_mean_of_encoder_out.png')

    target_image_scaled_to_01 = torch.from_numpy(np.array(image) * 1.0 / 255.0).to(device)
    reconstruction_loss = nn.MSELoss()


    z = torch.zeros([1, 4, 64, 64], requires_grad=True, device=device)
    z.data = latent_init.to(device)
    latent_init = latent_init.to(device)

    optimizer = optim.Adam([z], lr=1e-2)
    losses = []
    psnrs = []
    from tqdm import tqdm
    pbar = tqdm(range(num_steps))
    for step in pbar:
        optimizer.zero_grad()
        # Decode the latent variable to get the reconstructed image
        reconstructed_image = decoder(z).sample
        reconstructed_image_rescaled = (reconstructed_image / 2 + 0.5).type(torch.float32)
        
        # Compute the loss between the original image and the reconstructed image
        reconstruction_loss_value = reconstruction_loss(reconstructed_image_rescaled, target_image_scaled_to_01.permute(2, 0, 1).unsqueeze(0).type(torch.float32))
        loss = (
            scale_reconstruction * reconstruction_loss_value
            # + scale_l2 * ((z - target_mean)**2).mean()
            # + scale_gaussian * nn.GaussianNLLLoss()(z, target_mean, target_var)
        )

        # Backpropagation
        loss.backward()
        losses.append(loss.item())
        psnr = 10 * torch.log10(1 / torch.mean((reconstructed_image_rescaled - target_image_scaled_to_01.permute(2, 0, 1).unsqueeze(0).type(torch.float32)) ** 2)).item()
        psnrs.append(psnr)
        pbar.set_description(f"Loss: {loss.item():.6f} | PSNR: {psnr:.4f}")

        # Update the latent variable
        optimizer.step()

        # Optionally print the loss
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item()}")
            print(
                f"reconstruction loss: {reconstruction_loss_value.item():.3f}, "
            )
            rec2 = (
                reconstructed_image_rescaled[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                * 255
            ).astype(np.uint8)
            # save the reconstruction
            Image.fromarray(rec2).save(
                f"{out_dir}/reconstructed_step_{step:03d}.png"
            )
    with open(f"{out_dir}/vae_inversion_stats.json", "w") as f:
        json.dump({"LOSS": losses, "PSNR": psnrs}, f, indent=4)
    rec2 = (
                reconstructed_image_rescaled[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
                * 255
            ).astype(np.uint8)
    # save the reconstruction
    Image.fromarray(rec2).save(
                f"{out_dir}/reconstructed_step_{step:03d}.png"
            )
    return z.detach()