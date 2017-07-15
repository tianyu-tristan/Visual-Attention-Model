class Config(object):

    # training
    batch_size = 128 ##
    eval_batch_size = 50
    loc_std = 0.03 ##
    original_size = 32 ##
    num_channels = 1

    # glimpse
    win_size = 12 ##?
    bandwidth = win_size**2
    depth = 1
    sensor_size = win_size**2 * depth
    scale = 2 ##
    minRadius = 8
    hg_size = hl_size = 128
    g_size = 256
    loc_dim = 2 ##
    cell_size = 512 ##
    num_glimpses = 3 ##
    max_num_digits = 5 ##
    num_classes = 11 ##
    unit_pixel_ratio = 12 ## pixels
    max_grad_norm = 5. # gradient clipping
    max_num_error = 1 ## per sequence, stop gradients when reached

    step = 100000
    lr_start = 1e-2 ##
    lr_min = 1e-4

    # Monte Carlo sampling
    M = 1
