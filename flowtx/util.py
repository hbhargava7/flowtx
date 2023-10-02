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

def expand_well_addresses(input_str):
    """
    Convert a string representation of well ranges on a 96-well plate to a list of individual well addresses.
    
    Parameters:
    - input_str (str): A string representation of well ranges, e.g., "A3-C6,C9-E1,G2".
    
    Returns:
    - list: A list of individual well addresses based on the input ranges.
    
    Example:
    >>> expand_well_addresses("A3-C6,C9-E1,G2")
    ['A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C1', 'C9', 'C10', 'C11', 'C12', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'E1', 'G2']
    
    Raises:
    - ValueError: If the range is not increasing.
    """
    
    # Split the input string by comma to get individual ranges or addresses
    ranges = input_str.split(",")
    
    # Initialize an empty list to store the well addresses
    addresses = []
    
    for r in ranges:
        # If there's a dash, it's a range
        if "-" in r:
            start, end = r.split("-")
            
            # Check if the range is increasing
            if start > end:
                raise ValueError(f"Invalid range: {start}-{end}. The range should be increasing.")
            
            # Get the row letter and column number for the start and end of the range
            start_row, start_col = start[0], int(start[1:])
            end_row, end_col = end[0], int(end[1:])
            
            current_row, current_col = start_row, start_col
            while current_row <= end_row:
                while (current_row != end_row and current_col <= 12) or (current_row == end_row and current_col <= end_col):
                    addresses.append(current_row + str(current_col))
                    current_col += 1
                
                # Reset column and move to next row
                current_col = 1
                current_row = chr(ord(current_row) + 1)
        else:
            # If there's no dash, it's a single well address
            addresses.append(r)
    
    return addresses

