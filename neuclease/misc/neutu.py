import os
import json
import logging
import getpass
import datetime
import subprocess
from math import ceil, log10

import pandas as pd

from neuclease.util import dump_json, iter_batches

logger = logging.getLogger(__name__)

def create_bookmark_files(df, output_dir, prefix='bookmarks-', batch_size=100, default_text=None):
    os.makedirs(output_dir)
    digits = int(ceil(log10(len(df) / batch_size)))
    for i, bdf in enumerate(iter_batches(df, batch_size)):
        path = f"{output_dir}/{prefix}{{i:0{digits}d}}.json".format(i=i)
        create_bookmark_file(bdf, path, default_text)


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
        batch_grouping.extend((body, i, path) for body in batch)

    return pd.DataFrame(batch_grouping, columns=['body', 'assignment', 'file']).set_index('body')


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


def prepare_cleaving_assignment_setup(bodies, output_dir, bucket_path, sheet_path, prefix='cleaving-', batch_size=20):
    assert bucket_path.startswith('gs://')
    bucket_path = bucket_path[len('gs://'):]

    try:
        subprocess.run(f'gsutil ls gs://{bucket_path}', shell=True, check=True, capture_output=True)
    except subprocess.SubprocessError as ex:
        raise RuntimeError(f"Can't access gs://{bucket_path}") from ex

    df = create_cleaving_assignments(bodies, output_dir, prefix, batch_size)
    logger.info("Uploading assignment files")

    # Explicitly *unset* content type, to trigger browsers to download the file, not display it as JSON.
    # Also, forbid caching.
    cmd = f"gsutil -m -h 'Cache-Control:public, no-store' -h 'Content-Type' cp -r {output_dir} gs://{bucket_path}/"
    subprocess.run(cmd, shell=True, check=True)

    df['file'] = f'https://storage.googleapis.com/{bucket_path}/' + df['file']

    # Delete redundant file paths -- keep only the first row in each assignment group.
    df['file'] = df.groupby('assignment')['file'].head(1)
    df['file'].fillna("", inplace=True)

    df['user'] = ''
    df['date started'] = ''
    df['date completed'] = ''
    df['notes'] = ''
    df.to_csv(sheet_path, index=True, header=True)

    return df
