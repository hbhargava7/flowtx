# Copyright 2023 Hersh K. Bhargava (https://hershbhargava.com)
# University of California, San Francisco

import pandas as pd

def filter_df_by_dict(df, _dict):
    """Filter df by a dic

    Parameters
    ----------
    df : _type_
        _description_
    _dict : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    filtered_df = df
    for key, value in _dict.items():
        if key == 'well wells':
            filtered_df = filtered_df[filtered_df[key].str.contains(_dict[key])]
        else:
            filtered_df = filtered_df[filtered_df[key] == value]

    if len(filtered_df) > 1:
        print('Multiple rows found for %s' % _dict)

    elif len(filtered_df) == 0:
        print('Nothing found for %s' % _dict)

    return filtered_df