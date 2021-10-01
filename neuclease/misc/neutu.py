import os
import json
import getpass
import datetime
from math import ceil, log10

import pandas as pd

from neuclease.util import dump_json, iter_batches


def create_bookmark_file(df, output_path=None, default_text=None):
    """
    Create a NeuTu bookmark file from a set of points.

    Args:
        df:
            DataFrame with columns ['x', 'y', 'z'], and optionally 'body' and 'text'

        output_path:
            If given, the bookmark file will be written to the given path.

        default_text:
            Optional. If provided, any rows without a 'text' entry will be given this text.

    Returns:
        dict, the contents of a bookmarks file.
        (Typically, you'll just ignore the output and use the output_path argument instead.)
    """
    df = df.copy()
    if df.index.name == 'body':
        df['body'] = df.index

    assert set(df.columns) >= {*'xyz'}, "Input must contain columns for x,y,z"

    df['location'] = df[[*'xyz']].apply(list, axis=1)
    if 'body' in df.columns:
        df['body ID'] = df['body']

    if default_text:
        if 'text' in df.columns:
            blank_rows = df.eval('text.isnull() or text == ""')
            df.loc[blank_rows, 'text'] = default_text
        else:
            df['text'] = default_text

    cols = {'location', 'body ID', 'text'} & {*df.columns}
    data = df[cols].to_dict('records')

    # Does any of this metadata actually matter?
    metadata = {
        "description": "bookmarks",
        "date": datetime.datetime.now().strftime("%d-%B-%Y %H:%M"),
        "username": getpass.getuser(),
        "computer": os.uname().nodename,
        "session path": os.getcwd(),
        "coordinate system": "dvid",
        "software": __name__,
        "software version": "0",
        "software revision": "0",
        "file version": 1,
    }

    contents = {
        "metadata": metadata,
        "data": data
    }

    if output_path:
        dump_json(contents, output_path, unsplit_int_lists=True)

    return contents


def create_cleaving_assignments(bodies, output_dir, prefix='cleaving-', batch_size=20):
    os.makedirs(output_dir)
    digits = int(ceil(log10(len(bodies) / batch_size)))

    batch_grouping = []
    for i, batch in enumerate(iter_batches(bodies, batch_size)):
        path = f"{output_dir}/{prefix}{{i:0{digits}d}}.json".format(i=i)
        create_cleaving_assignment(batch, path)
        batch_grouping.extend((body, i) for body in batch)

    return pd.DataFrame(batch_grouping, columns=['body', 'assignment']).set_index('body')['assignment']


def create_cleaving_assignment(bodies, output_path):
    task_list = []
    for body in bodies:
        task = {
            "task type": "body cleave",
            "body ID": int(body),
            "maximum level": 1
        }
        task_list.append(task)

    assignment = {
        "file type": "Neu3 task list",
        "file version": 1,
        "task list": task_list
    }
    with open(output_path, 'w') as f:
        json.dump(assignment, f, indent=2)
