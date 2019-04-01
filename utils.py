import yaml
import os
import numpy as np
import torch


def _load_image(path):
    """
        Reads image image from the given path and returns an numpy array.
    """
    image = np.load(path)
    assert image.dtype == np.uint8
    assert image.shape == (64, 64, 3)
    return image


def _read_image(file_name):
    """
        Returns a tuple of image as numpy array and label as int,
        given the csv row.
    """
    input_folder = "test_images/"
    img_path = os.path.join(input_folder, file_name)
    image = _load_image(img_path)
    assert image.dtype == np.uint8
    image = image.astype(np.float32)
    assert image.dtype == np.float32
    return image


def read_images():
    """
        Returns a list containing tuples of images as numpy arrays
        and the correspoding label.
        In case of an untargeted attack the label is the ground truth label.
        In case of a targeted attack the label is the target label.
    """
    filepath = "test_images/labels.yml"
    with open(filepath, 'r') as ymlfile:
        data = yaml.load(ymlfile)

    data_key = list(data.keys())
    data_key.sort()
    return [(key, _read_image(key), data[key]) for key in data_key]


def check_image(image):
    # image should a 64 x 64 x 3 RGB image
    assert(isinstance(image, np.ndarray))
    assert(image.shape == (64, 64, 3))
    if image.dtype == np.float32:
        # we accept float32, but only if the values
        # are between 0 and 255 and we convert them
        # to integers
        if image.min() < 0:
            logger.warning('clipped value smaller than 0 to 0')
        if image.max() > 255:
            logger.warning('clipped value greater than 255 to 255')
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    assert image.dtype == np.uint8
    return image


def store_adversarial(file_name, adversarial):
    """
        Given the filename, stores the adversarial as .npy file.
    """
    if adversarial is not None:
        adversarial = check_image(adversarial)

    path = os.path.join("results", file_name)
    path_without_extension = os.path.splitext(path)[0]
    np.save(path_without_extension, adversarial)
    '''
    from scipy import misc
    misc.imsave(path_without_extension+".jpg", adversarial)
    '''


def compute_MAD():

    def load_image(path):
        x = np.load(path)
        assert x.shape == (64, 64, 3)
        assert x.dtype == np.uint8
        return x
    def distance(X, Y):
        assert X.dtype == np.uint8
        assert Y.dtype == np.uint8
        X = X.astype(np.float64) / 255
        Y = Y.astype(np.float64) / 255
        return np.linalg.norm(X - Y)
    # distance if no adversarial was found (worst case)
    def worst_case_distance(X):
        assert X.dtype == np.uint8
        worst_case = np.zeros_like(X)
        worst_case[X < 128] = 255
        return distance(X, worst_case)

    distances = []
    real_distances = []
    for file in os.listdir('results/'):
        original = load_image('test_images/{}'.format(file))
        try:
            adversarial = load_image('results/{}'.format(file))
        except AssertionError:
            #print('adversarial for {} is invalid'.format(file))
            adversarial = None
        if adversarial is None:
            _distance = float(worst_case_distance(original))
        else:
            _distance = float(distance(original, adversarial))
        real_distances.append(_distance)

    real_distances = np.array(real_distances)
    distances = real_distances * 255

    print("\tMedian Distance:  %.6f"  %np.median(real_distances[distances > 50]))
    print("\tMean Distance:    %.6f"  %np.mean(real_distances[distances > 50]))