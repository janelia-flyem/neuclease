"""
Utilities for converting a neuroglancer video script to a Google Sheet and back.

The neuroglancer video_tool uses a very simple format for video scripts:

<url>
<duration>
<url>
<duration>
...

That format is simple to create, but annoying to edit.

This utility will load a video script into a Google Sheet in which each
link has been converted to a JSON state and written out to a single column,
with subsequent links in subsequent columns, with JSON settings aligned so
that settings are easily compared across a whole row.

You can edit settings in the Google sheet, and then import them back into
a video script using this utility.

Recommended Google Sheet template:
https://docs.google.com/spreadsheets/d/1SC-szBtssuqMnVrNdZv9LHj4UB0rUFMwZaxeDnekA4c/edit?gid=162774670#gid=162774670

Copy that template and pass the URL of your copied sheet to this tool.
Make sure your sheet is visible to "anyone with the link" and editable at least by you
(via your GOOGLE_APPLICATION_CREDENTIALS).

Example usage:

    SHEET_URL=https://docs.google.com/spreadsheets/d/15Q1oRLZQd1MWZOjRr_DyIkFbbhMZCSP4sbU78JXkoSM/edit?gid=162774670#gid=162774670
    python video_tool_utils.py script-to-sheet video_script.txt $SHEET_URL
    python video_tool_utils.py sheet-to-script $SHEET_URL video_script.txt

To install the neuroglancer video_tool:

    pixi init selenium geckodriver pillow numpy requests tornado six google-apitools google-auth atomicwrites ffmpeg
    pixi add --pypi neuroglancer

Note:
    This tool accepts (and produces) a non-standard format for neuroglancer video scripts
    which allows comments (starting with '#') and blank lines.
    When writing to a video script, comments can be excluded with the --no-comments flag.
    Otherwise, you'll have to exclude them yourself before running the neuroglancer video_tool:

        pixi run python -m neuroglancer.tool.video_tool render \\
            --browser firefox --no-headless --refresh-browser-timeout=10 \\
            --hide-axis-lines --height=2160 --width=2160 \\
            <(grep -v '^#' my-video.txt | grep -v '^$') my-video-frames

        pixi run ffmpeg -r 24 -f image2 -s 2160x2160 -i my-video-frames/%07d.png \\
            -vcodec libx264 -crf 25 -pix_fmt yuv420p my-video.mp4
"""
import argparse
import csv
import json
import os
import re
import urllib.parse
import warnings
from collections import namedtuple

import gspread
import pandas as pd

ScriptItem = namedtuple('ScriptItem', ['comment', 'transition_duration', 'link', 'state', 'link_line_number'])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Read command
    read_parser = subparsers.add_parser('script-to-sheet', help='Read video script and upload to a Google spreadsheet')
    read_parser.add_argument('video_script_path')
    read_parser.add_argument('output_sheet', default='', nargs='?')

    # Write command
    write_parser = subparsers.add_parser('sheet-to-script', help='Write from Google Sheet to video script')
    write_parser.add_argument(
        '--no-comments', action='store_true',
        help='Do not include comments in the output script, making it a valid neuroglancer '
             'video script without any postprocessing needed.')
    write_parser.add_argument('google_sheet_url')
    write_parser.add_argument('output_script_path')
    write_parser.add_argument(
        '--ng-server', default='neuroglancer-demo.appspot.com',
        help='Neuroglancer server URL (default: neuroglancer-demo.appspot.com)'
    )

    args = parser.parse_args()

    if args.command == 'script-to-sheet':
        item_df = video_script_to_dataframe(args.video_script_path)
        if args.output_sheet.startswith('http'):
            overwrite_google_sheet(item_df, args.output_sheet)
        else:
            write_tsv(item_df, args.output_sheet)
    elif args.command == 'sheet-to-script':
        load_from_google_sheet(args.google_sheet_url, args.output_script_path, not args.no_comments, args.ng_server)


def video_script_to_dataframe(video_script_path: str):
    print(f"Reading video script from {video_script_path}")
    with open(video_script_path, 'r') as f:
        lines = f.readlines()

    script_items = []
    cur_comment = ''
    cur_duration = 0
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if line == '':
            continue

        if line.startswith('#'):
            cur_comment += line + '\n'
            continue

        if not line.startswith('http'):
            try:
                duration = float(line)
            except ValueError:
                raise RuntimeError(f"Line {i} is not a comment, link, or duration:\n{line}")  # noqa
            else:
                if cur_duration != 0:
                    raise RuntimeError(f"Line {i} lists a duration after no link.")
                cur_duration = duration
                continue

        state = parse_nglink(line)
        item = ScriptItem(cur_comment, cur_duration, line, state, i)
        script_items.append(item)
        cur_comment = ''
        cur_duration = 0

    nested_keys = [list_nested_keys(item.state) for item in script_items]
    merged_keys = nested_keys[0]
    for nk in nested_keys[1:]:
        merged_keys = merge_nested_keys(merged_keys, nk)

    state_jsons = []
    for item in script_items:
        j = dump_aligned_json(item.state, merged_keys)
        state_jsons.append(j)

    state_series = []
    for j in state_jsons:
        state_series.append(pd.Series(j.splitlines()))

    state_df = pd.DataFrame(state_series)

    item_df = pd.DataFrame({
        "comment": [item.comment for item in script_items],
        "transition_duration": [item.transition_duration for item in script_items],
        "link_line_number": [item.link_line_number for item in script_items],
    })
    item_df = pd.concat([item_df, state_df], axis=1)
    return item_df


def write_tsv(item_df: pd.DataFrame, output_tsv_path: str):
    print(f"Writing TSV to {output_tsv_path}")
    item_df.T.to_csv(
        output_tsv_path,
        header=False,
        index=True,
        sep='\t',
        quoting=csv.QUOTE_ALL
    )


def overwrite_google_sheet(item_df: pd.DataFrame, sheet_url: str):
    gid = re.search(r'gid=(\d+)', sheet_url).group(1)

    gc = gspread.service_account(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    sh = gc.open_by_url(sheet_url)
    ws = sh.get_worksheet_by_id(gid)
    print(f"Overwriting Google sheet '{ws.title}'")
    ws.clear()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Method signature's arguments 'range_name' and 'values'.*")
        ws.update(
            values=item_df.T.reset_index().values.tolist(),
            range_name='A1'
        )


def load_from_google_sheet(worksheet_url: str, output_script_path: str, include_comments=True, ng_server: str = 'neuroglancer-demo.appspot.com'):
    if not ng_server.startswith('http'):
        ng_server = f'https://{ng_server}'

    gid = re.search(r'gid=(\d+)', worksheet_url).group(1)

    sheet_url = re.match(r'(.*)/.*?\?.*', worksheet_url).group(1)
    export_url = f'{sheet_url}/export?format=csv&gid={gid}'
    print(f"Loading from Google sheet: '{export_url}'")
    df = pd.read_csv(export_url, header=None, dtype=str).set_index(0).fillna('').T
    df['comment'] = df['comment'].str.strip()
    df['state'] = df.loc[:, '0':].sum(axis=1).map(json.loads)
    df['link'] = df['state'].map(lambda state: format_nglink(ng_server, state))

    print(f"Writing script to {output_script_path}")
    with open(output_script_path, 'w') as f:
        for i, row in enumerate(df.itertuples()):
            if include_comments and row.comment:
                f.write(f'{row.comment}\n')
            if i != 0:
                f.write(f'{row.transition_duration}\n')
            f.write(f'{row.link}\n')
            if include_comments:
                f.write('\n')


def parse_nglink(link):
    _, pseudo_json = link.split('#!')
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return data


def format_nglink(ng_server, link_json_settings):
    return ng_server + '/#!' + urllib.parse.quote(json.dumps(link_json_settings))


def list_nested_keys(obj):
    keys = []
    _list_nested_keys(obj, (), keys)
    return keys


def _list_nested_keys(obj, base_key, keys):
    match obj:
        case float() | int() | bool() | str():
            keys.append(base_key)
        case list() if all(isinstance(v, (float, int, bool)) for v in obj):
            keys.append(base_key)
        case list() if all(isinstance(v, str) for v in obj) and pd.Series(pd.to_numeric(obj, errors='coerce')).notnull().all():
            keys.append(base_key)
        case list():
            keys.append((*base_key, '['))
            for i, v in enumerate(obj):
                _list_nested_keys(v, (*base_key, i), keys)
            keys.append((*base_key, ']'))
        case dict():
            keys.append((*base_key, '{'))
            for k, v in obj.items():
                _list_nested_keys(v, (*base_key, k), keys)
            keys.append((*base_key, '}'))
        case _:
            raise ValueError(f"Unknown type: {type(obj)}")


def merge_nested_keys(a, b):
    common_keys = set(a) & set(b)
    a_common = list(k for k in a if k in common_keys)
    b_common = list(k for k in b if k in common_keys)
    if a_common != b_common:
        raise ValueError(f"Common keys occur in different orders: {a_common} vs. {b_common}")

    a, b = list(a), list(b)

    new_keys = []
    while a or b:
        while a and (ka := a.pop(0)) not in common_keys:
            new_keys.append(ka)

        while b and (kb := b.pop(0)) not in common_keys:
            new_keys.append(kb)

        if ka == kb:
            new_keys.append(ka)

    return new_keys


def dump_aligned_json(obj, all_nested_keys, indent=2):
    key_offsets = {k: i for i, k in enumerate(all_nested_keys)}
    lines = [''] * len(all_nested_keys)

    match obj:
        case dict():
            key = ('{',)
        case list():
            key = ('[',)
        case _:
            raise ValueError(f"Unknown type: {type(obj)}")

    _dump_aligned_json(obj, key, 0, key_offsets, indent, lines, '')
    return '\n'.join(lines)


def _dump_aligned_json(obj, key, indent_level, key_offsets, indent, lines, trailing_comma):
    padding = " " * indent_level * indent
    match obj:
        case float() | int() | bool():
            line = lines[key_offsets[key]]
            lines[key_offsets[key]] = padding + line + str(obj).lower() + trailing_comma
        case str():
            line = lines[key_offsets[key]]
            lines[key_offsets[key]] = padding + line + f'"{obj}"' + trailing_comma
        case list() if all(isinstance(v, (float, int, bool)) for v in obj):
            line = lines[key_offsets[key]]
            lines[key_offsets[key]] = padding + line + "[" + ", ".join(map(str, obj)) + "]" + trailing_comma
        case list() if all(isinstance(v, str) for v in obj) and pd.Series(pd.to_numeric(obj, errors='coerce')).notnull().all():
            line = lines[key_offsets[key]]
            lines[key_offsets[key]] = padding + line + "[" + ", ".join(f'"{v}"' for v in obj) + "]" + trailing_comma
        case list():
            *key, suffix = key
            assert suffix == '['
            line = lines[key_offsets[(*key, '[')]]
            lines[key_offsets[(*key, '[')]] = padding + line + "["
            for i, v in enumerate(obj):
                match v:
                    case float() | int() | bool() | str():
                        item_key = (*key, i)
                    case list() if all(isinstance(x, (float, int)) for x in v):
                        item_key = (*key, i)
                    case list():
                        item_key = (*key, i, '[')
                    case dict():
                        item_key = (*key, i, '{')
                    case _:
                        raise ValueError(f"Unknown type: {type(v)}")

                tc = ',' if i < len(obj) - 1 else ''
                _dump_aligned_json(v, item_key, indent_level+1, key_offsets, indent, lines, tc)
            lines[key_offsets[(*key, ']')]] += padding + "]" + trailing_comma
            return lines
        case dict():
            *key, suffix = key
            assert suffix == '{'
            line = lines[key_offsets[(*key, '{')]]
            lines[key_offsets[(*key, '{')]] = padding + line + "{"
            for i, (k, v) in enumerate(obj.items()):
                match v:
                    case float() | int() | bool() | str():
                        item_key = (*key, k)
                    case list() if all(isinstance(x, (float, int, bool)) for x in v):
                        item_key = (*key, k)
                    case list() if all(isinstance(x, str) for x in v) and pd.Series(pd.to_numeric(v, errors='coerce')).notnull().all():
                        item_key = (*key, k)
                    case list():
                        item_key = (*key, k, '[')
                    case dict():
                        item_key = (*key, k, '{')
                    case _:
                        raise ValueError(f"Unknown type: {type(v)}")

                lines[key_offsets[item_key]] += padding + f'"{k}": '
                tc = ',' if i < len(obj) - 1 else ''
                _dump_aligned_json(v, item_key, indent_level+1, key_offsets, indent, lines, tc)
            lines[key_offsets[(*key, '}')]] += padding + "}" + trailing_comma
        case _:
            raise ValueError(f"Unknown type: {type(obj)}")


if __name__ == '__main__':
    main()
