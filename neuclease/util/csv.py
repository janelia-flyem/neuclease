import csv
import numpy as np
import pandas as pd


def read_csv_header(csv_path):
    """
    Open the CSV file at the given path and return it's header column names as a list.
    If it has no header (as determined by csv.Sniffer), return None.
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
            has_header = csv.Sniffer().has_header(csv_file.read(1024))
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
    """
    return (read_csv_header(csv_path) is not None)


def read_csv_col(csv_path, col=0, dtype=np.uint64):
    """
    Read a single column from a CSV file as a pd.Series.
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


