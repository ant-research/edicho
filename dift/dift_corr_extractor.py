"""
    This file contains the class extracting correspondence of the input images based on DIFT.
    Cache (based on hash_md5) of DIFT features and correspondence is designed for efficiency.
"""

import gc
import os
import sys
import time
from PIL import Image
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import PILToTensor

from dift.dift_sd import SDFeaturizer
from utils import generate_hash_for_combined_images


class DIFT_Corr_Extractor:
    def __init__(self, img_size=512, category='dancing', use_null_prompt=True,
                 sd_version='15',
                 save_feats=True,
                 feats_save_dir='./dift_results/features',
                 save_corr_matrix=True,
                 corr_save_dir='./dift_results/corrs',
                 ensemble_size=8):

        self.img_size = img_size
        self.category = category
        self.use_null_prompt = use_null_prompt
        if self.use_null_prompt:
            self.prompt = ''
        else:
            self.prompt = f'a photo of a {category}'
        self.save_feats = save_feats
        self.feats_save_dir = feats_save_dir
        os.makedirs(self.feats_save_dir, exist_ok=True)
        self.save_corr_matrix = save_corr_matrix
        self.corr_save_dir = corr_save_dir
        self.sd_version = sd_version

        # init DIFT feature extractor.
        if sd_version == '15':
            sd_id = "runwayml/stable-diffusion-v1-5"
            self.dift_init_res = 32
        else:
            sd_id = "stabilityai/stable-diffusion-2-1"
            self.dift_init_res = 48
        self.dift_extractor = SDFeaturizer(sd_id)
        self.ensemble_size = ensemble_size
        print("Init DIFT extractor!")

    # Input: list of numpy arrays
    def extract_dift(self, imgs, resize=False):
        imglist = []
        feats = []
        for img in imgs:
            hash_key = generate_hash_for_combined_images([img])
            dift_path = os.path.join(self.feats_save_dir, f'{hash_key}.pth')
            if os.path.exists(dift_path):
                feat = torch.load(dift_path, map_location='cuda')
            else:
                img = Image.fromarray(img).convert('RGB')
                if resize:
                    img = img.resize((self.img_size, self.img_size))
                imglist.append(img)
                img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
                feat = self.dift_extractor.forward(img_tensor,
                                                   prompt=self.prompt,
                                                   ensemble_size=self.ensemble_size)
                if self.save_feats:
                    torch.save(feat, dift_path)
            feats.append(feat)
        feats = torch.cat(feats, dim=0)
        print('dift shape: ', feats.shape)
        gc.collect()
        torch.cuda.empty_cache()
        return feats

    def extract_corr_from_dift(self, feats, hash_key='None', token_H=64, token_W=64, return2D=False):
        if return2D:
            corr_save_dir = os.path.join(self.corr_save_dir, f'h{token_H}_w{token_W}')
        else:
            corr_save_dir = os.path.join(self.corr_save_dir, f'1D_{token_H*token_W}')
        corr_path = os.path.join(corr_save_dir, f'{hash_key}.pth')
        if os.path.exists(corr_path):
            corr = torch.load(corr_path)
            print('Load correspondence from ', corr_path)
        # Compute correspondence based on dift.
        else:
            upsampler = nn.Upsample(size=(token_H, token_W), mode='bilinear')
            # Upsampling dift features for further matching.
            dift_source = upsampler(feats[0:1])
            dift_target = upsampler(feats[1:])
            B, C, H, W = dift_source.shape  # Upsampled final shape of dift feature.
            dift_source = dift_source.view(B, C, -1)  # [B, C, H*W]
            dift_target = dift_target.view(B, C, -1)  # [B, C, H*W]
            # normalize
            dift_source_norm = F.normalize(dift_source)
            dift_target_norm = F.normalize(dift_target)
            t1 = time.time()
            sim_matrix = torch.matmul(dift_target_norm.transpose(1, 2),
                                      dift_source_norm)  # ~0.6s+2GB for [bs, 1280, 256*256]
            print('dift - corr computing time: ', time.time() - t1, 's')
            sim, indices = torch.max(sim_matrix, dim=1)
            gc.collect()
            torch.cuda.empty_cache()
            if not return2D:
                corr = indices[0]
            else:
                corr = indices[0].view(H, W)
            if self.save_corr_matrix:
                os.makedirs(corr_save_dir, exist_ok=True)
                torch.save(corr, corr_path)  # save corr!
            print(f'Correspondence Matrix saved at {corr_path}!')
        return corr