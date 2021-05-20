# test_time_adaptation_with_empirical_gaussian_prior

Main network architecture and part of the code are from https://github.com/neerakara/test-time-adaptable-neural-networks-for-domain-generalization.


## Train task model
To train the task model, change configuration in *train_config.py* and run *train_TaskModel.py*.


## Test-time adaptation
### 1. Empirical Gaussian prior

Firstly, run *calculate_prior.py* to calculate and save different Gassuain priors, by setting the 'prior_type' to: 
> 'egp_vp': calculate the mean and variance for each channel and each layer.  
> 'egp_ndc': calculate the mean vector and covariance (inversed) for each layer.  
> 'egp_sw': calculate subject-wise priors: mean of mean, variance of mean, mean of variance and variance of variance.

According to the prior type, change the loss function for TTA in _model.py_. Then change configuration in _tta_config.py_ and run _tta_EGP.py_.


### 2. Autoencoder

For training autoencoders, change configuration in *train_config.py* and run *train_AE.py*. Set 'feature_type' in *train_AE.py* to the location of the feature layer used for training AE. Feature layers include 'NI' (for the normalized image), 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2', 'conv6_2', 'conv7_2'.

For TTA with AEs, change configuration in _tta_config.py_ and run _tta_AE.py_.


## Prediction

Functions for prediction are in _predict.py_. Models of the adapted normalization network are saved separately for each patient subject. Run _predict_seg(image, model_dir_SD, model_dir_TD, patname)_ function to get the normalized images and predicted masks after TTA, by setting 'model_dir_TD' to the model dir saving adapted models for all patient subjects, and 'patname' to the according patient name.
