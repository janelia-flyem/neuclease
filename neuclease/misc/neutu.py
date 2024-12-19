import os
import json
import logging
import getpass
import platform
import datetime
import subprocess

import numpy as np
import pandas as pd

from neuclease.util import dump_json, iter_batches

logger = logging.getLogger(__name__)


def create_bookmark_files(df, output_dir, prefix='bookmarks-', batch_size=100, default_text=None, grayscale=None, segmentation=None, dvid_src=None, dataset=None, layers=None):
    os.makedirs(output_dir)
    batch_ids = []
    paths = []
    counts = []
    batch_dfs = []

    if isinstance(batch_size, str):
        # Batch by column(s)
        batch_ids, batches = zip(*df.groupby(batch_size, sort=False))
        digits = len(str(df[batch_size].nunique()-1))
    else:
        batches = iter_batches(df, batch_size)
        batch_ids = np.arange(len(batches))
        digits = len(str(len(df) // batch_size))

    for i, bdf in enumerate(batches):
        path = f"{output_dir}/{prefix}{i:0{digits}d}.json"
        paths.append(path)
        counts.append(len(bdf))
        batch_dfs.append(bdf.assign(assignment=i))
        create_bookmark_file(bdf, path, default_text, grayscale, segmentation, dvid_src, dataset, layers)
    return batch_ids, counts, paths, batch_dfs


def create_bookmark_file(df, output_path=None, default_text=None, grayscale=None, segmentation=None, dvid_src=None, dataset=None, layers=None):
    """
    Create a NeuTu bookmark file from a set of points.

    Args:
        df:
            DataFrame with columns ['x', 'y', 'z'], and
            optionally 'body',  'text', 'extra' (extra body IDs), 'suggest_extra' (if True, show extra IDs by default)

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

    if 'extra' in df.columns:
        df["extra body IDs"] = df['extra'].map(lambda x: x if x else [])

    if 'suggest_extra' in df.columns:
        df["suggest extra"] = df['suggest_extra'] & df['extra body IDs'].map(len).astype(bool)

    cols = [*{'location', 'body ID', 'text', 'extra body IDs', 'suggest extras'} & {*df.columns}]
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

    if segmentation:
        contents = {
            "segmentation source": segmentation,
            **contents
        }

    if grayscale:
        contents = {
            "grayscale source": grayscale,
            **contents
        }

    if dvid_src:
        contents = {
            "DVID source": dvid_src,
            **contents
        }

    if dataset:
        contents = {
            "dataset": dataset,
            **contents
        }

    if layers:
        contents = {
            "layers": layers,
            **contents
        }

    if output_path:
        dump_json(contents, output_path, unsplit_int_lists=True)

    return contents


def read_bookmark_points(path):
    """
    Read a bookmark JSON file and return a dataframe
    with the enclosed points, and body IDs (if present).
    """
    with open(path, 'r') as f:
        j = json.load(f)
    df = pd.DataFrame(j['data']).rename(columns={'body ID': 'body'})
    df[[*'xyz']] = df['location'].tolist()
    del df['location']
    return df

def prepare_bookmark_assignment_setup(df, output_dir, bucket_path, csv_path, prefix='bookmarks-', batch_size=50, default_text=None, grayscale=None, segmentation=None, dvid_src=None, dataset=None, layers=None, *, include_bodylist=False):
    """
    This function will help prepare a set of orphan-link (bookmark) assignments.
    Unlike the analogous cleaving setup (below), the resulting spreadsheet
    doesn't contain a separate row for every body.
    We just put each assignment on its own row.

        1. Generates JSON files
        2. Uploads them to a google bucket (assuming you have permission)
        3. Exports a CSV. You'll have to manually import that CSV file into
           a Google sheet to share with the proofreaders.

    Before running this function, you need to to log in to the Google Cloud
    system by entering the following command in your terminal:

        gcloud auth login <your-email-here>

    Finally, here's a checklist of things you should consider doing in the google sheet:

        - Protect column headers
        - Protect left-hand columns (body, assignment, file)
        - Data validation for 'date started' and 'date completed'
            - Must be a date

    Args:
        df:
            DataFrame of xyz coordinates.
        output_dir:
            Name of a local directory to create for the assignments
        bucket_path:
            The bucket name + directory to which the assignment directory will be copied.
            Example: gs://foobar-flyem-assignments/cns
        csv_path:
            A CSV file will be written to the given path.
            That's the CSV file you'll want to import into Google Sheets
        prefix:
            If provided, each assignment file will be named with this prefix
            (and followed with an assignment number, e.g. bookmarks-001.json)
        batch_size:
            How many bodies per assignment

    Returns:
        The assignment CSV data.

    Example:

            tracking_df = prepare_bookmark_assignment_setup(
                df,
                'my-orphan-link-tasks',
                'gs://my-assignments/foo/my-orphan-link'
                'my-orphan-link-tracking.csv',
                'my-orphan-link-',
            )
    """
    assert bucket_path.startswith('gs://')
    bucket_path = bucket_path[len('gs://'):]

    # First, verify that we have permission to edit the bucket.
    with open("/tmp/test-file.txt", 'w') as f:
        f.write("Just testing my bucket access...\n")
    try:
        p = subprocess.run(f"gsutil cp /tmp/test-file.txt gs://{bucket_path}/test-file.txt", shell=True, check=True, capture_output=True)
        p = subprocess.run(f"gsutil rm gs://{bucket_path}/test-file.txt", shell=True, check=True, capture_output=True)
    except subprocess.SubprocessError as ex:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError(f"Can't access gs://{bucket_path}") from ex

    if not isinstance(df, pd.DataFrame):
        raise NotImplementedError

    batch_ids, counts, output_paths, batch_dfs = create_bookmark_files(df, output_dir, prefix, batch_size, default_text, grayscale, segmentation, dvid_src, dataset, layers)
    tracking_df = pd.DataFrame({
        'assignment': batch_ids,
        'body_count': counts,
        'file': output_paths,
    })
    logger.info("Uploading bookmark files")

    upload_assignment_files(output_dir, bucket_path)
    tracking_df['file'] = f'https://storage.googleapis.com/{bucket_path}/' + tracking_df['file']

    # Delete redundant file paths -- keep only the first row in each assignment group.
    tracking_df['file'] = tracking_df.groupby('assignment')['file'].head(1)
    tracking_df['file'] = tracking_df['file'].fillna("")

    tracking_df['user'] = ''
    tracking_df['date started'] = ''
    tracking_df['date completed'] = ''
    tracking_df['notes'] = ''

    if include_bodylist:
        batch_df = pd.concat(batch_dfs)
        if batch_df.index.name == 'body':
            batch_df['body'] = batch_df.index
        tracking_df['bodies'] = batch_df.groupby('assignment')['body'].agg(list)

    tracking_df.to_csv(csv_path, index=False, header=True)
    return tracking_df


def create_cleaving_assignments(bodies, output_dir, prefix='cleaving-', batch_size=20):
    os.makedirs(output_dir)
    digits = len(str(len(bodies) // batch_size))

    batch_grouping = []
    for i, batch in enumerate(iter_batches(bodies, batch_size)):
        path = f"{output_dir}/{prefix}{i:0{digits}d}.json"
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


def prepare_cleaving_assignment_setup(bodies, output_dir, bucket_path, csv_path, prefix='cleaving-', batch_size=10):
    """
    This function will help prepare a set of cleaving assignments.

        1. Generates JSON files
        2. Uploads them to a google bucket (assuming you have permission)
        3. Exports a CSV with the structure we like to use.  You'll have to manually
           import that CSV file into a Google sheet to share with the proofreaders.

    Before running this function, you need to to log in to the Google Cloud
    system by entering the following command in your terminal:

        gcloud auth login <your-email-here>

    Finally, here's a checklist of things you should consider doing in the google sheet:

        - Protect column headers
        - Protect left-hand columns (body, assignment, file)
        - Conditional formatting to highlight assignment grouping:
            - Select "Custom formula is" and enter "=mod($B2,2)"
        - Conditional formatting to highlight uncompleted assignments
            - Select "Custom formula is" and enter "=and(mod(row(K2), 10)=2, J2<>"", not(K2))"
        - Data validation for 'date started' and 'date completed'
            - Must be a date

    Args:
        bodies:
            List of bodies to assign
        output_dir:
            Name of a local directory to create for the assignments
        bucket_path:
            The bucket name + directory to which the assignment directory will be copied.
            Example: gs://foobar-flyem-assignments/cns
        csv_path:
            A CSV file will be written to the given path.
            That's the CSV file you'll want to import into Google Sheets
        prefix:
            If provided, each assignment file will be named with this prefix
            (and followed with an assignment number, e.g. cleaving-001.json)
        batch_size:
            How many bodies per assignment

    Returns:
        The assignment CSV data.
    """
    assert bucket_path.startswith('gs://')
    bucket_path = bucket_path[len('gs://'):]

    # First, verify that we have permission to edit the bucket.
    with open("/tmp/test-file.txt", 'w') as f:
        f.write("Just testing my bucket access...\n")
    try:
        p = subprocess.run(f"gsutil cp /tmp/test-file.txt gs://{bucket_path}/test-file.txt", shell=True, check=True, capture_output=True)
        p = subprocess.run(f"gsutil rm gs://{bucket_path}/test-file.txt", shell=True, check=True, capture_output=True)
    except subprocess.SubprocessError as ex:
        print(p.stdout)
        print(p.stderr)
        raise RuntimeError(f"Can't access gs://{bucket_path}") from ex

    if isinstance(bodies, pd.DataFrame):
        assert bodies.index.name == 'body'
        df = create_cleaving_assignments(bodies.index, output_dir, prefix, batch_size)
        assert not ({*bodies.columns} & {*df.columns}), \
            "Make sure the DataFrame you provided doesn't have column names that clash with the columns this function adds"
        df = bodies.merge(df, 'left', on='body')
    else:
        df = create_cleaving_assignments(bodies, output_dir, prefix, batch_size)

    logger.info("Uploading assignment files")
    upload_assignment_files(output_dir, bucket_path)

    df['file'] = f'https://storage.googleapis.com/{bucket_path}/' + df['file']

    # Delete redundant file paths -- keep only the first row in each assignment group.
    df['file'] = df.groupby('assignment')['file'].head(1)
    df['file'] = df['file'].fillna("")

    df['user'] = ''
    df['date started'] = ''
    df['date completed'] = ''
    df['notes'] = ''

    assert df.index.name == 'body'

    df.to_csv(csv_path, index=True, header=True)
    return df


def create_connection_validation_assignment(df, output_path):
    """
    Create a single connection validation assignment.

    Args:
        df:
            A dataframe of PSD coordinates, with columns ['x', 'y', 'z'].
        output_path:
            Where to store the assignment JSON file.
    Returns:
        The assignment json data, which was also written to the specified file.
    """
    assignment = {
        "file type": "connection validation",
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "username": getpass.getuser(),
        "software": "NeuTu",
        "coordinate system": "dvid",
        "file version": 1,
        "points": df[[*'xyz']].values.tolist()
    }
    dump_json(assignment, output_path, unsplit_int_lists=True)
    return assignment


def create_connection_validation_assignments(df, output_dir, prefix='connection-validation-', batch_size=100):
    """
    Create a directory of connection validation assignments.

    Args:
        df:
            A dataframe of PSD coordinates, with columns ['x', 'y', 'z'].
        output_dir:
            A directory will be created at the given path and populated with assignment JSON files.
        prefix:
            Each assignment file will be named with the given prefix and an assignment number.
        batch_size:
            The number of tasks per assignment file.
    """
    os.makedirs(output_dir)
    digits = len(str(len(df) // batch_size))
    for i, batch_df in enumerate(iter_batches(df, batch_size)):
        path = f"{output_dir}/{prefix}{i:0{digits}d}.json"
        create_connection_validation_assignment(batch_df, path)


def upload_assignment_files(local_dir, bucket_path):
    # Explicitly *unset* content type, to trigger browsers to download the file, not display it as JSON.
    # Also, forbid caching.
    opts = [
        "-h 'Cache-Control:public, no-store'",
        "-h 'Content-Type'",
    ]
    if platform.system() == 'Darwin':
        # gsutil gives the following warning:
        #
        #   If you experience problems with multiprocessing on MacOS,
        #   they might be related to https://bugs.python.org/issue33725.
        #   You can disable multiprocessing by editing your .boto config or
        #   by adding the following flag to your command: `-o "GSUtil:parallel_process_count=1"`.
        #   Note that multithreading is still available even if you disable multiprocessing.
        opts += [
            "-o GSUtil:parallel_process_count=1",
            "-o GSUtil:parallel_thread_count=8",
        ]
    cmd = f"gsutil -m {' '.join(opts)} cp -r {local_dir} gs://{bucket_path}/"
    _ = subprocess.run(cmd, shell=True, check=True, capture_output=True)
