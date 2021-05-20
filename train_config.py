import networks
import tensorflow as tf

run_number = 3

# training dataset
train_dataset = 'HCPT1'
orig_data_root_hcp = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/HCP/3T_Structurals_Preprocessed/'
preproc_folder_hcp = 'data/'

# data aug settins
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

# data settings
image_size = (256,256)
image_depth_hcp = 256
target_resolution_brain = (0.7, 0.7)
nlabels = 15
loss_type_i2l = 'dice'

# models settings
model_handle_normalizer = networks.net2D_i2i
norm_kernel_size = 3
norm_num_hidden_layers = 2
norm_num_filters_per_layer = 16
norm_activation = 'rbf'
norm_batch_norm = False
model_handle_i2l = networks.unet2D_i2l

# training setting
max_epochs = 300
batch_size = 16
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
summary_writing_frequency = 10
train_eval_frequency = 1
val_eval_frequency = 1
save_frequency = 10
continue_run = True
