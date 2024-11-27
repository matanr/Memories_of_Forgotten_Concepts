import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def ddim_inversion(latents, encoder_hidden_states, noise_scheduler, unet):
    next_latents = latents
    all_latents = [latents.detach().cpu()]

    # since we are adding noise to the image, we reverse the timesteps list to start at t=0
    reverse_timestep_list = reversed(noise_scheduler.timesteps)

    for i in range(len(reverse_timestep_list) - 1):
        timestep = reverse_timestep_list[i]
        next_timestep = reverse_timestep_list[i + 1]
        latent_model_input = noise_scheduler.scale_model_input(next_latents, timestep)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states).sample

        alpha_prod_t = noise_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_next = noise_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        f = (next_latents - beta_prod_t**0.5 * noise_pred) / (alpha_prod_t**0.5)
        next_latents = alpha_prod_t_next**0.5 * f + beta_prod_t_next**0.5 * noise_pred
        all_latents.append(next_latents.detach().cpu())

    return all_latents


def null_text_inversion(
    pipe,
    all_latents,
    prompt,
    num_opt_steps=15,
    lr=0.01,
    tol=1e-5,
    guidance_scale=7.5,
    eta: float = 0.0,
    generator=None,
    device=None,
):
    # initialise null text embeddings
    null_text_prompt = ""
    null_text_input = pipe.tokenizer(
        null_text_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncaton=True,
        return_tensors="pt",
    )

    # prepare for optimising
    null_text_embeddings = torch.nn.Parameter(
        pipe.text_encoder(null_text_input.input_ids.to(pipe.device))[0],
        requires_grad=True,
    )
    null_text_embeddings = null_text_embeddings.detach()
    null_text_embeddings.requires_grad_(True)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        [null_text_embeddings],  # only optimize the embeddings
        lr=lr,
    )

    # step_ratio = pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    text_embeddings = pipe.encode_prompt(prompt, device, 1, False, None)[0].detach()
    # input_embeddings = torch.cat([null_text_embeddings, text_embeddings], dim=0)
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    all_null_texts = []
    latents = all_latents[-1]
    latents = latents.to(pipe.device)
    for timestep, prev_latents in pipe.progress_bar(
        zip(pipe.scheduler.timesteps, reversed(all_latents[:-1]))
    ):
        prev_latents = prev_latents.to(pipe.device).detach()

        # expand the latents if we are doing classifier free guidance
        latent_model_input = pipe.scheduler.scale_model_input(
            latents, timestep
        ).detach()
        noise_pred_text = pipe.unet(
            latent_model_input, timestep, encoder_hidden_states=text_embeddings
        ).sample.detach()
        for _ in range(num_opt_steps):
            # predict the noise residual
            noise_pred_uncond = pipe.unet(
                latent_model_input, timestep, encoder_hidden_states=null_text_embeddings
            ).sample

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            prev_latents_pred = pipe.scheduler.step(
                noise_pred, timestep, latents, **extra_step_kwargs
            ).prev_sample
            loss = torch.nn.functional.mse_loss(prev_latents_pred, prev_latents).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        all_null_texts.append(null_text_embeddings.detach().cpu())
        latents = prev_latents_pred.detach()
    return all_latents[-1], all_null_texts


@torch.no_grad()
def reconstruct(
    pipe,
    latents,
    prompt,
    null_text_embeddings,
    guidance_scale=7.5,
    generator=None,
    eta=0.0,
    device=None,
):
    text_embeddings = pipe.encode_prompt(prompt, device, 1, False, None)[0]
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    latents = latents.to(pipe.device)
    for i, (t, null_text_t) in enumerate(
        pipe.progress_bar(zip(pipe.scheduler.timesteps, null_text_embeddings))
    ):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        input_embedding = torch.cat([null_text_t.to(pipe.device), text_embeddings])
        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=input_embedding
        ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample

    # Post-processing
    image = pipe.decode_latents(latents)
    return image