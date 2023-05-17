import os

train_dir = 'train'
data_dir = 'generated-OUTLINE'


val_dir = 'test'
imgs_dir = 'mask'

noisy_dir = 'image'
debug_dir = 'debug'

patch_dir = '/home/sstauffer/specops/gitlab/layoutex/assets/pix2pix-overlay'
asset_dir = '/home/sstauffer/specops/gitlab/layoutex/assets'

txt_file_dir = '/home/sstauffer/specops/words/words_alpha.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 100
train_percentage = 0.8

test_dir = os.path.join(data_dir, val_dir, noisy_dir)
