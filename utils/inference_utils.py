import numpy as np
from tabulate import tabulate

def pad_to_next_multiple_of_32(img):
    """
    Pad each dim to multiples of 2^5 = 32
    Assuming the encoder is a ResNet with 5 blocks, it max pools images 5 times.
    To avoid runtime errors images are padded to the the next multiple of 32. Separately for each dimension

    Args:
        img ([np.ndarray]): Image (1 channel).
    """
    pad = []
    for dim in img.shape[1:]:
        if dim % 32 == 0:
            pad.append(0)
        else:
            pad.append(32 - dim % 32)
    return np.pad(img, ((0,0),(0, pad[0]),(0, pad[1])))


def summarize_results(data, identifiers, header, filter_values='all'):
    """Compute means for a list of metrics given for many samples.
    | Category    |   Mean DH |   Epicardial(MYO) DH |   Endocardial(LV) DH |
    |:------------|----------:|---------------------:|---------------------:|
    | T1_Mapping_ |   1.76248 |              1.51705 |              2.00791 |
    | T2_Mapping_ |   1.2696  |              1.27899 |              1.2602  |

    Args:
        data (list or list of list): [description]
        identifiers (list of str): ideitifier for each data point, e.g. image path, filtering is performed using this
        header (list of str): [description]
        filter_values (list of str, optional): group data by matching filter values to identifiers. Defaults to 'all'.
    """
    if filter_values == 'all':
        filter_values = ['']

    results_table = []
    for category in filter_values: 
        per_class_filtered_by_category = [d for id, d in zip(identifiers, data) if category in id]
        per_class_averaged_by_category = np.nanmean(np.stack(per_class_filtered_by_category), axis=0)
        avearaged_by_class_and_category = np.mean(per_class_averaged_by_category)
        results_table.append([category, avearaged_by_class_and_category, *per_class_averaged_by_category.tolist()])
    
    # if len(filter_values) > 1:
    #     numeric_columns = [row[1:] for row in results_table]
    #     results_table.append(["**Mean**", *np.mean(np.stack(numeric_columns), axis=0).tolist()]) # This doesn't give an accurate Mean for all samples...

    data_array = np.stack(data)
    means = [np.nanmean(data_array),] + np.nanmean(data_array, axis=0).tolist()
    stds  = [np.nanstd(data_array),] +  np.nanstd(data_array, axis=0).tolist()
    results_table.append(["**Mean**", *means])
    results_table.append(["**Std**", *stds])
    
    print(tabulate(results_table, header, tablefmt="pipe"))

    means = dict(zip(header[1:], means))
    stds = dict(zip(header[1:], stds))

    return means, stds
