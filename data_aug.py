import numpy as np
import utils
import scipy.ndimage.interpolation
from skimage import transform


def augment_images(images,
                   data_aug_ratio,
                   sigma,
                   alpha,
                   trans_min,
                   trans_max,
                   rot_min,
                   rot_max,
                   scale_min,
                   scale_max,
                   gamma_min,
                   gamma_max,
                   brightness_min,
                   brightness_max,
                   noise_min,
                   noise_max):
    images_ = np.copy(images)

    for i in range(images.shape[0]):
        # elastic deformation
        if np.random.rand() < data_aug_ratio:
            images_[i,:,:] = utils.elastic_transform_image(images_[i,:,:], sigma=sigma, alpha=alpha)
        # translation
        if np.random.rand() < data_aug_ratio:
            random_shift_x = np.random.uniform(trans_min, trans_max)
            random_shift_y = np.random.uniform(trans_min, trans_max)
            images_[i, :, :] = scipy.ndimage.interpolation.shift(images_[i, :, :], shift=(random_shift_x, random_shift_y), order=1)

        # rotation
        if np.random.rand() < data_aug_ratio:
            random_angle = np.random.uniform(rot_min, rot_max)
            images_[i, :, :] = scipy.ndimage.interpolation.rotate(images_[i, :, :], reshape=False, angle=random_angle, axes=(1, 0), order=1)

        # scaling
        if np.random.rand() < data_aug_ratio:
            n_x, n_y = images_.shape[1], images_.shape[2]
            scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
            images_i_tmp = transform.rescale(images_[i, :, :], scale_val, order=1, preserve_range=True, mode='constant')

            images_[i, :, :] = utils.crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)

        # contrast
        if np.random.rand() < data_aug_ratio:
            # gamma contrast augmentation
            c = np.round(np.random.uniform(gamma_min, gamma_max), 2)
            # print('c: ', c)
            images_[i, :, :] = images_[i, :, :] ** c

        # brightness
        if np.random.rand() < data_aug_ratio:
            c = np.round(np.random.uniform(brightness_min, brightness_max), 2)
            images_[i, :, :] = images_[i, :, :] + c

        # noise
        if np.random.rand() < data_aug_ratio:
            n = np.random.normal(noise_min, noise_max, size=images_[i, :, :].shape)
            images_[i, :, :] = images_[i, :, :] + n

        return images_


def augment_images_and_labels(images, labels,
                              data_aug_ratio,
                              sigma,
                              alpha,
                              trans_min,
                              trans_max,
                              rot_min,
                              rot_max,
                              scale_min,
                              scale_max,
                              gamma_min,
                              gamma_max,
                              brightness_min,
                              brightness_max,
                              noise_min,
                              noise_max):
    images_ = np.copy(images)
    labels_ = np.copy(labels)

    for i in range(images.shape[0]):
        # elastic deformation
        if np.random.rand() < data_aug_ratio:
            images_[i,:,:], labels_[i,:,:] = utils.elastic_transform_image_and_label(images_[i,:,:], labels_[i,:,:],
                                                                                     sigma=sigma, alpha=alpha)
        # translation
        if np.random.rand() < data_aug_ratio:
            random_shift_x = np.random.uniform(trans_min, trans_max)
            random_shift_y = np.random.uniform(trans_min, trans_max)
            images_[i, :, :] = scipy.ndimage.interpolation.shift(images_[i, :, :], shift=(random_shift_x, random_shift_y), order=1)
            labels_[i, :, :] = scipy.ndimage.interpolation.shift(labels_[i, :, :], shift=(random_shift_x, random_shift_y), order=0)

        # rotation
        if np.random.rand() < data_aug_ratio:
            random_angle = np.random.uniform(rot_min, rot_max)
            images_[i, :, :] = scipy.ndimage.interpolation.rotate(images_[i, :, :], reshape=False, angle=random_angle, axes=(1, 0), order=1)
            labels_[i, :, :] = scipy.ndimage.interpolation.rotate(labels_[i, :, :], reshape=False, angle=random_angle, axes=(1, 0), order=0)

        # scaling
        if np.random.rand() < data_aug_ratio:
            n_x, n_y = images_.shape[1], images_.shape[2]
            scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
            images_i_tmp = transform.rescale(images_[i, :, :], scale_val, order=1, preserve_range=True, mode='constant')

            labels_i_tmp = transform.rescale(labels_[i, :, :], scale_val, order=0, preserve_range=True,mode='constant')
            images_[i, :, :] = utils.crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)
            labels_[i, :, :] = utils.crop_or_pad_slice_to_size(labels_i_tmp, n_x, n_y)

        # contrast
        if np.random.rand() < data_aug_ratio:
            # gamma contrast augmentation
            c = np.round(np.random.uniform(gamma_min, gamma_max), 2)
            # print('c: ', c)
            images_[i, :, :] = images_[i, :, :] ** c

        # brightness
        if np.random.rand() < data_aug_ratio:
            c = np.round(np.random.uniform(brightness_min, brightness_max), 2)
            images_[i, :, :] = images_[i, :, :] + c

        # noise
        if np.random.rand() < data_aug_ratio:
            n = np.random.normal(noise_min, noise_max, size=images_[i, :, :].shape)
            images_[i, :, :] = images_[i, :, :] + n

        return images_, labels_

