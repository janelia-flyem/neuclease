import os
import getpass
import datetime

from neuclease.util import dump_json


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
