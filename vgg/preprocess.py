

def prep(train_x, train_y, test_x, test_y):
    """Preprocessing applied to features.

    From each pixel, subtract the mean RGB value of the training set.
    """
    mean_rgb = train_x.mean(axis=(0, 1, 2))
    train_x = train_x - mean_rgb
    test_x = test_x - mean_rgb
    return train_x, train_y, test_x, test_y
