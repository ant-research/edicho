import os
import glob
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import BrushNetModel, UniPCMultistepScheduler
from pipelines.pipeline_brushnet import BrushNetPipeline
from attnctrl.attn_utils import register_attention_editor_diffusers
from attnctrl.edicho_attnctrl import EdichoSelfAttentionControl
from dift.dift_corr_extractor import DIFT_Corr_Extractor
from utils import generate_hash_for_combined_images


def setup_pipeline(base_model_path, brushnet_path, torch_dtype=torch.float16):
    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)
    pipe = BrushNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe


# Init DIFT feature and correspondence extractor (global to reuse across calls).
dift_corr_extractor = DIFT_Corr_Extractor()


def process(pipe,
           input_image_and_mask1,
           input_image_and_mask2,
           prompt,
           seed,
           attnctrl_start_step,
           attnctrl_end_step,
           attnctrl_start_layer,
           sync_latent,
           sync_latent_start_step,
           sync_latent_end_step,
           sync_latent_guidance_scale,
           num_inference_steps=50):
    generator = torch.Generator("cuda").manual_seed(seed)
    prompts = [prompt] * 2

    input_image1, input_mask1 = input_image_and_mask1['image'], input_image_and_mask1['mask']
    input_image2, input_mask2 = input_image_and_mask2['image'], input_image_and_mask2['mask']

    H_init, W_init, C = input_image1.shape[0], input_image1.shape[1], input_image1.shape[2]
    if H_init >= W_init:
        W = 512
        H = H_init * (W / W_init)
        H = int(H // 32 * 32)
    else:
        H = 512
        W = W_init * (H / H_init)
        W = int(W // 32 * 32)

    # Resize image and mask.
    input_image1 = cv2.resize(input_image1, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    input_image2 = cv2.resize(input_image2, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    input_mask1 = cv2.resize(input_mask1, dsize=(W, H), interpolation=cv2.INTER_CUBIC)
    input_mask2 = cv2.resize(input_mask2, dsize=(W, H), interpolation=cv2.INTER_CUBIC)

    mask_pixel1 = torch.tensor(input_mask1.astype(np.float16) / 255.0).cuda()
    mask_pixel2 = torch.tensor(input_mask2.astype(np.float16) / 255.0).cuda()

    mask_image1 = 1.0 * (input_mask1.sum(-1) > 255)[:, :, np.newaxis]
    init_image1 = input_image1 * (1 - mask_image1)
    init_image1 = Image.fromarray(init_image1.astype(np.uint8)).convert("RGB")
    mask_image1 = Image.fromarray(mask_image1.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

    mask_image2 = 1.0 * (input_mask2.sum(-1) > 255)[:, :, np.newaxis]
    init_image2 = input_image2 * (1 - mask_image2)
    init_image2 = Image.fromarray(init_image2.astype(np.uint8)).convert("RGB")
    mask_image2 = Image.fromarray(mask_image2.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")

    init_images = [init_image1, init_image2]
    mask_images = [mask_image1, mask_image2]

    imgs = [input_image1, input_image2]
    dift_feats = dift_corr_extractor.extract_dift(imgs)
    hash_key = generate_hash_for_combined_images(imgs)

    editor = EdichoSelfAttentionControl(
        start_step=attnctrl_start_step,
        end_step=attnctrl_end_step,
        start_layer=attnctrl_start_layer,
        replace_token_rate=1.0,
        dift_feats=dift_feats,
        hash_key=hash_key,
        dift_corr_extractor=dift_corr_extractor,
        imgW=W,
        imgH=H,
        mask_s=torch.tensor(mask_pixel1[:, :, 0]).cuda(),
        mask_t=torch.tensor(mask_pixel2[:, :, 0]).cuda(),
    )

    register_attention_editor_diffusers(pipe, editor)
    print('Registered attention editor!')


    if not sync_latent:
        images = pipe(
            prompts,
            init_images,
            mask_images,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images
    else:
        latentW = W // 8
        latentH = H // 8
        corr_latent = dift_corr_extractor.extract_corr_from_dift(dift_feats, hash_key,
                                                                 token_H=latentH, token_W=latentW, return2D=False)
        images = pipe(
            prompts,
            init_images,
            mask_images,
            num_inference_steps=num_inference_steps,
            generator=generator,
            sync_latent=sync_latent,
            sync_latent_start_step=sync_latent_start_step,
            sync_latent_end_step=sync_latent_end_step,
            sync_latent_guidance_scale=sync_latent_guidance_scale,
            corr_latent=corr_latent
        ).images

    results = []
    for index in range(len(images)):
        result = np.array(images[index])
        input_img = np.array(init_images[index])
        mask = np.array(mask_images[index])
        mask_np = 1.0 * (mask.sum(-1) > 255)[:, :, np.newaxis]

        mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
        mask_blurred = mask_blurred[:, :, np.newaxis]
        mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

        image_pasted = input_img * (1 - mask_np) + result * mask_np
        image_pasted = image_pasted.astype(result.dtype)
        results.append(image_pasted)
    return results


def concatenate_np_images_horizontally(image_arrays):
    if not image_arrays:
        raise ValueError("List is blank!")
    for i in range(1, len(image_arrays)):
        if image_arrays[i].shape[0] != image_arrays[0].shape[0] or image_arrays[i].shape[2] != image_arrays[0].shape[2]:
            raise ValueError("Height and dim of images must be the same!")
    return np.concatenate(image_arrays, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="BrushNet image editing with attention control (argparse version).")

    # IO and prompt
    parser.add_argument("--in_dir", type=str, default="./images/toy", help="Input directory containing input_*.png/jpg and mask_*.jpg")
    parser.add_argument("--save_dir_root", type=str, default="./results", help="Root directory to save results")
    parser.add_argument("--prompt_main", type=str, default="A simple monster toy", help="Main prompt")
    parser.add_argument("--prompt_add", type=str, default=", high quality", help="Additional prompt to append")

    # Model paths
    parser.add_argument("--base_model_path", type=str, default="/path/to/checkpoints/realisticVision_diffuser", help="Base model path")
    parser.add_argument("--brushnet_path", type=str, default="Sanster/brushnet_segmentation_mask", help="BrushNet checkpoint path")

    # Sampling steps and seed
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Attention control configs
    parser.add_argument("--attnctrl_start_step", type=int, default=4, help="Attention control start step")
    parser.add_argument("--attnctrl_end_step", type=int, default=40, help="Attention control end step")
    parser.add_argument("--attnctrl_start_layer", type=int, default=8, help="Attention control start layer")

    # Sync latent configs
    parser.add_argument("--sync_latent", action="store_true", default=True, help="Enable latent synchronization")
    parser.add_argument("--sync_latent_start_step", type=int, default=4, help="Sync latent start step")
    parser.add_argument("--sync_latent_end_step", type=int, default=39, help="Sync latent end step")
    parser.add_argument("--sync_latent_guidance_scale", type=float, default=0.6, help="Sync latent guidance scale")

    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare save directory
    dirname = os.path.basename(args.in_dir.rstrip("/"))
    save_dir = os.path.join(args.save_dir_root, dirname)
    os.makedirs(save_dir, exist_ok=True)

    # Setup pipeline
    pipe = setup_pipeline(
        base_model_path=args.base_model_path,
        brushnet_path=args.brushnet_path,
        torch_dtype=torch.float16,
    )

    prompt = args.prompt_main + args.prompt_add

    # Gather inputs
    img_paths = sorted(glob.glob(os.path.join(args.in_dir, "input_*.png")) + glob.glob(os.path.join(args.in_dir, "input_*.jpg")))[:2]
    mask_paths = sorted(glob.glob(os.path.join(args.in_dir, "mask_*.jpg")))[:2]

    if len(img_paths) < 2 or len(mask_paths) < 2:
        raise RuntimeError(f"Need at least two images and two masks in {args.in_dir}")

    imgs = [np.array(Image.open(i)) for i in img_paths]
    masks = [np.array(Image.open(i)) for i in mask_paths]

    input_image_and_mask1 = {'image': imgs[0], 'mask': masks[0]}
    input_image_and_mask2 = {'image': imgs[1], 'mask': masks[1]}

    # Run process
    results = process(
        pipe=pipe,
        input_image_and_mask1=input_image_and_mask1,
        input_image_and_mask2=input_image_and_mask2,
        prompt=prompt,
        seed=args.seed,
        attnctrl_start_step=args.attnctrl_start_step,
        attnctrl_end_step=args.attnctrl_end_step,
        attnctrl_start_layer=args.attnctrl_start_layer,
        num_inference_steps=args.num_inference_steps,
        sync_latent=args.sync_latent,
        sync_latent_start_step=args.sync_latent_start_step,
        sync_latent_end_step=args.sync_latent_end_step,
        sync_latent_guidance_scale=args.sync_latent_guidance_scale,
    )

    # Save result
    img_concat = Image.fromarray(concatenate_np_images_horizontally(results))
    prompt_clean = args.prompt_main.replace('?', '').replace('#', '').replace('!', '').replace(',', '').replace(' ', '_')
    job_name = f"{prompt_clean}_seed{args.seed}"
    out_path = os.path.join(save_dir, f"{job_name}.jpg")
    img_concat.save(out_path)
    print(f"Results saved at: {out_path}")


if __name__ == "__main__":
    main()