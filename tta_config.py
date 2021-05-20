import networks
import tensorflow as tf


orig_data_root_hcp = '../../../../../usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/HCP/3T_Structurals_Preprocessed/'
orig_data_root_abide = '../../../../../usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/ABIDE_Neerav/CALTECH/'
test_dataset = 'HCPT2'
preproc_folder_test = 'test_data/'
normalize = True
run_number = 1

# normalize architecture
model_handle_normalizer = networks.net2D_i2i
norm_kernel_size = 3
norm_num_hidden_layers = 2
norm_num_filters_per_layer = 16
norm_activation = 'rbf'
norm_batch_norm = False

# image to label mapper
model_handle_i2l = networks.unet2D_i2l

# data settings
image_size = (256, 256)
image_depth_hcp = 256
image_depth_caltech = 256
target_resolution_brain = (0.7, 0.7)
target_resolution_prostate = (0.625, 0.625)
nlabels = 15
batch_size = 16

# training settings
continue_run = True
max_epochs = 100
optimizer_handle = tf.train.AdamOptimizer
learning_rate = 1e-3


# data augmentation settings
da_ratio = 0.3
sigma = 20
alpha = 1000
trans_min = -10
trans_max = 10
rot_min = -10
rot_max = 10
scale_min = 0.9
scale_max = 1.1
gamma_min = 0.5
gamma_max = 2.0
brightness_min = 0.0
brightness_max = 0.1
noise_min = 0.0
noise_max = 0.1
