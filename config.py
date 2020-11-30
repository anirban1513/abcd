import cv2
import albumentations as A
from albumentations.augmentations.transforms import Equalize, Posterize, Downscale
from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, CenterCrop,    
    RandomCrop, Resize, Crop, Compose, HueSaturationValue,
    Transpose, RandomRotate90, ElasticTransform, GridDistortion, 
    OpticalDistortion, RandomSizedCrop, Resize, CenterCrop,
    VerticalFlip, HorizontalFlip, OneOf, CLAHE, Normalize,
    RandomBrightnessContrast, Cutout, RandomGamma, ShiftScaleRotate ,
    GaussNoise, Blur, MotionBlur, GaussianBlur, 
)

SEED = 69
n_epochs = 15
rate = 0.70
device = 'cuda:0'
data_dir = '/home/ubuntu/data/'
slices = 1
loss_thr = 1e6
balanced_sampler = True
img_path = f'{data_dir}/Lung_{2*slices+1}_channel/images'
label_path = f'{data_dir}/Lung_{2*slices+1}_channel/labels'
pseudo_label_path = f'{data_dir}/Lung_{2*slices+1}_channel/pseudo_labels'
encoder_model = 'gluon_seresnext101_32x4d'
model_name= 'Unet' # Will come up with a better name later
model_dir = 'model_dir'
history_dir = 'history_dir'
load_model = False
apply_log = False
img_dim = 512
batch_size = 24
learning_rate = 2.25e-3
num_workers = 4
mixed_precision = True
patience = 3
train_aug = A.Compose([A.CenterCrop(p=0.3, height=300, width=300),
A.augmentations.transforms.RandomCrop(int(0.875*img_dim), int(0.875*img_dim), p=0.3),
A.augmentations.transforms.Rotate(limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
A.augmentations.transforms.Resize(img_dim, img_dim, interpolation=1, always_apply=True, p=0.6),
Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, always_apply=False, p=0.2),
# A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=0.3),
# A.augmentations.transforms.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, always_apply=False, p=0.4),
OneOf([
        GaussNoise(var_limit=0.1),
        Blur(),
        GaussianBlur(blur_limit=3),
        # RandomGamma(p=0.7),
        ], p=0.3),
A.HorizontalFlip(p=0.3)])
val_aug = Compose([Normalize(always_apply=True)])
