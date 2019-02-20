import csv
import numpy as np
import pandas as pd


def read_csv_header(csv_path):
    """
    Open the CSV file at the given path and return it's header column names as a list.
    If it has no header (as determined by csv.Sniffer), return None.
    
    This function handles a special case that Python's ``csv.Sniffer``
    does not handle properly: The case of a single-column csv file.
    In that case, if the first column begins with a non-number,
    it is assumed to have a header; otherwise not.
    
    Note:
        If the file contains a single column of strings,
        it is assumed that the first row is the header.
    
    Returns:
        If the file has no header row, None.
        Otherwise, returns a list of strings.
    """
    with open(csv_path, 'r') as csv_file:
        first_line = csv_file.readline()
        csv_file.seek(0)
        if ',' not in first_line:
            # csv.Sniffer doesn't work if there's only one column in the file
            try:
                int(first_line)
                has_header = False
            except:
                has_header = True
        else:
            first_three_lines = ''.join(csv_file.readline() for _ in range(3))
            has_header = csv.Sniffer().has_header(first_three_lines)
            csv_file.seek(0)

        if not has_header:
            return None
    
        rows = iter(csv.reader(csv_file))
        header = next(rows)
        return header


def csv_has_header(csv_path):
    """
    Return True if the CSV file at the given path
    appears to have a header row, False otherwise.

    Note:
        If the file contains a single column of strings,
        it is assumed that the first row is the header.
    """
    return (read_csv_header(csv_path) is not None)


def read_csv_col(csv_path, col=0, dtype=np.uint64):
    """
    Read a single column from a CSV file as a ``pd.Series``.
    
    Args:
        csv_path:
            Path to the CSV file to open
        
        col:
            Index of the desired column to read
        
        dtype:
            The dtype of the returned Series.
    
    Returns:
        ``pd.Series``, named according to the column's header (if any).
    """
    int(col) # must be an int
    header_names = read_csv_header(csv_path)
    if header_names:
        header_row = 0
        names = [header_names[col]]
    else:
        header_row = None
        names = ['noname']

    s = pd.read_csv(csv_path, header=header_row, usecols=[col], names=names, dtype=dtype)[names[0]]
    
    if header_row is None:
        s.name = None
    return s


