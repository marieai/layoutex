import os

train_dir = 'train'
data_dir = 'generated-OUTLINE'


val_dir = 'test'
imgs_dir = 'mask'

noisy_dir = 'image'
debug_dir = 'debug'

patch_dir = '/home/greg/datasets/dataset/rms/pix2pix-overlay'
asset_dir = '/home/greg/datasets/dataset/rms/assets'

txt_file_dir = 'text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 3000
train_percentage = 0.8

test_dir = os.path.join(data_dir, val_dir, noisy_dir)
