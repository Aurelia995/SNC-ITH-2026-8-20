import numpy as np
from scipy import ndimage

def calITHscore(label_map, min_area=200, thresh=2):
    """
    Calculate ITHscore from clustering label map
    Args:
        label_map: Numpy array. Clustering label map
        min_area: Int. For tumor area (pixels) smaller than "min_area", we don't consider connected-region smaller than "thresh"
        thresh: Int. The threshold for connected-region's area, only valid for tumor < min_area
    Returns:
        ith_score: Float. The level of ITH, between 0 and 1
    """
    size = np.sum(label_map > 0)  # Record the number of total pixels
    num_regions_list = []
    max_area_list = []
    for i in np.unique(label_map)[1:]:  # For each gray level except 0 (background)
        flag = 1  # Flag to count this gray level, in case this gray level has only one pixel
        # Find (8-) connected-components. "num_regions" is the number of connected components
        labeled, num_regions = ndimage.label(label_map==i, structure=ndimage.generate_binary_structure(2,2))
        max_area = 0
        for j in np.unique(labeled)[1:]:  # 0 is background (here is all the other regions)
            # Ignore the region with only 1 or "thresh" px
            if size <= min_area:
                if np.sum(labeled == j) <= thresh:
                    num_regions -= 1
                    if num_regions == 0:  # In case there is only one region
                        flag = 0
                else:
                    temp_area = np.sum(labeled == j)
                    if temp_area > max_area:
                        max_area = temp_area
            else:
                if np.sum(labeled == j) <= 1:
                    num_regions -= 1
                    if num_regions == 0:  # In case there is only one region
                        flag = 0
                else:
                    temp_area = np.sum(labeled == j)
                    if temp_area > max_area:
                        max_area = temp_area
        if flag == 1:
            num_regions_list.append(num_regions)
            max_area_list.append(max_area)
    # Calculate the ITH score
    ith_score = 0
    print(num_regions_list)
    for k in range(len(num_regions_list)):
        ith_score += float(max_area_list[k]) / num_regions_list[k]
    # Normalize each area with total size
    ith_score = ith_score / size
    ith_score = 1 - ith_score

    return ith_score
