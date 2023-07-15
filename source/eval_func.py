import numpy as np
import SimpleITK as sitk

from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, distance_transform_edt
from skimage.metrics import structural_similarity, mean_squared_error


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype('bool'))
    reference = np.atleast_1d(reference.astype('bool'))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def dc(result, reference):
    result = np.atleast_1d(result.astype('bool'))
    reference = np.atleast_1d(reference.astype('bool'))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dice = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dice = 0.0

    return dice


def hd(result, reference, voxelspacing=None, connectivity=1, percentage=None):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    if percentage == None:
        distance = max(hd1.max(), hd2.max())
    elif isinstance(percentage, (int, float)):
        distance = np.percentile(np.hstack((hd1, hd2)), percentage)
    else:
        raise ValueError
    return distance


def seg_eval(results, reference, label_mapping=None):
    if label_mapping is None:
        category_ids = np.intersect1d(np.unique(results), np.unique(reference)).astype('int').tolist()
        category_ids.remove(0)
    else:
        category_ids = list(label_mapping.keys())

    evaluation_dict = {}

    running_dice = 0.
    running_dist = 0.
    running_percent_dist = 0.

    for category_id in category_ids:
        binary_pre = results == category_id
        binary_ref = reference == category_id
        dice = dc(binary_pre, binary_ref)
        percent_dist = hd(binary_pre, binary_ref, percentage=95)
        dist = hd(binary_pre, binary_ref)

        if label_mapping is not None:
            evaluation_dict[f'{label_mapping[category_id]}Dice'] = dice
            evaluation_dict[f'{label_mapping[category_id]}HD95'] = percent_dist
            evaluation_dict[f'{label_mapping[category_id]}HD'] = dist

        running_dist += dist
        running_dice += dice
        running_percent_dist += percent_dist

    running_dice /= 3
    running_dist /= 3
    running_percent_dist /= 3

    evaluation_dict['AvgDice'] = running_dice
    evaluation_dict['AvgHD95'] = running_percent_dist
    evaluation_dict['AvgHD'] = running_dist

    return evaluation_dict


def sim_eval(results, reference):
    ssim = structural_similarity(results, reference, data_range=1.0)
    mse = mean_squared_error(results, reference)

    evaluation_dict = {'SSIM': ssim, 'MSE': mse}

    return evaluation_dict


def negative_jacobin(flow):
    w, h, c = np.shape(flow)

    flow_image = sitk.GetImageFromArray(flow.astype('float64'), isVector=True)
    determinant = sitk.DisplacementFieldJacobianDeterminant(flow_image)
    neg_jacobin = (sitk.GetArrayFromImage(determinant)) < 0
    cnt = np.sum(neg_jacobin)
    norm_cnt = cnt / (h * w)
    return cnt, norm_cnt * 100
