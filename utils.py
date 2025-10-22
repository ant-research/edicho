import torch
from PIL import Image
import hashlib

# Generate a unique hash key for a combination of image A and B.
# Input: numpy.array  Output: hash_key
def generate_hash_for_combined_images(imgs):
    combined_image = None
    for img in imgs:
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        if combined_image is None:
            combined_image = img
        else:
            combined_image = combine_images_horizontally(combined_image, img)

    combined_image_hash = hashlib.md5(combined_image.tobytes()).hexdigest()
    return combined_image_hash


def combine_images_horizontally(img1, img2):
    width = img1.width + img2.width
    height = max(img1.height, img2.height)
    combined_image = Image.new('RGB', (width, height))
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (img1.width, 0))
    return combined_image


def get_corr_xy_from_matrix(corr, patch_size, x ,y):
    print(f'query x:{x} y:{y}')
    token_H, token_W = corr.shape
    in_x_token = x // patch_size
    in_y_token = y // patch_size
    out_token_index = corr[in_y_token, in_x_token]
    out_y_token = out_token_index // token_W  # which line?
    out_x_token = out_token_index % token_W   # which row?
    print(f'predicted token correspondence - x:{out_x_token} y:{out_y_token}')

    print(f'predicted pixel correspondence: x:{out_x_token * patch_size}-{(out_x_token+1) * patch_size}')
    print(f'                                y:{out_y_token * patch_size}-{(out_y_token+1) * patch_size}')
    out_center_x = out_x_token * patch_size + patch_size // 2
    out_center_y = out_y_token * patch_size + patch_size // 2
    return int(out_center_x), int(out_center_y)


# calculate mean & std of all the tokens. shape: [bs, head_num, 4096, 64] -> [bs, head_num, 1, 64]
def calc_mean_std(feat, eps=1e-5):
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

# expand the mean & std of the first image to the whole batch. Input shape: [num_sample, head_num, 1, 64]
def expand_first_sample_feat(feat, scale=1.0):
    b = feat.shape[0]
    # extract the noise of the reference (1st tensor).
    feat_style = feat[:1].unsqueeze(1)  # [1, 1, head_num, 1, dim]
    if scale == 1:
        feat_style = feat_style.expand(1, b, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)

# Token-based Adain norm of two parts - source and target.
# Calculating the mean and std of each token first and expand the mean and std of the first images.
# Finally, use the expanded mean and std (of 1st image) to denorm the tokens.
def adain(feat):
    # assert len(feat.shape) == 4, 'Adain only accepts 4-dim tensor!'
    feat_mean, feat_std = calc_mean_std(feat)
    feat_mean_first = expand_first_sample_feat(feat_mean)
    feat_std_first = expand_first_sample_feat(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_std_first + feat_mean_first
    return feat


# norm with global std and mean
def global_norm(feat):
    feat_mean, feat_std = calc_mean_std(feat)
    feat = (feat - feat_mean) / feat_std
    return feat