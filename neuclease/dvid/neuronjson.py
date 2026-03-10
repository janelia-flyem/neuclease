import re
from functools import wraps

import pandas as pd

from . import dvid_api_wrapper
from .keyvalue._keyvalue import _body_annotations_dataframe, DEFAULT_BODY_STATUS_CATEGORIES


@dvid_api_wrapper
def _fetch_query(server, uuid, instance='segmentation_annotations', query=None, endpoint='query', *, show=None, fields=None, status_categories=DEFAULT_BODY_STATUS_CATEGORIES, format='pandas', session=None):
    assert endpoint in ('all', 'query')
    assert show in ('user', 'time', 'all', None)
    assert format in ('pandas', 'json')

    params = {}
    if show:
        params['show'] = show

    if fields:
        if isinstance(fields, str):
            fields = [fields]
        params['fields'] = ','.join(fields)

    url = f'{server}/api/node/{uuid}/{instance}/{endpoint}'
    r = session.get(url, params=params, json=query)
    r.raise_for_status()
    values = r.json() or []

    if format == 'pandas':
        ann = _body_annotations_dataframe(values, status_categories)
        if fields and 'status' not in fields:
            ann = ann.drop(columns=['status'], errors='ignore')
        return ann
    else:
        return sorted(values, key=lambda d: d['bodyid'])


def _parse_column_name(col):
    """
    Parse column names into (field, attribute) tuples
    """
    # Match patterns like 'a', 'a_user', 'a_time'
    match = re.match(r'^(.*?)(?:_(user|time))?$', col)
    if match:
        field = match.group(1)
        attr = match.group(2) if match.group(2) else 'value'
        return (field, attr)
    return (col, 'value')


def _melt_all(df):
    df = df.drop(columns=['bodyid', 'json'], errors='ignore')

    # Create MultiIndex columns
    df.columns = pd.MultiIndex.from_tuples(
        [_parse_column_name(c) for c in df.columns],
        names=['field', 'attribute']
    )
    df = df.stack(level='field', future_stack=True).reset_index()
    return df


def fetch_all(server, uuid, instance='segmentation_annotations', show=None, fields=None, melt=False, status_categories=DEFAULT_BODY_STATUS_CATEGORIES, format='pandas', session=None):
    assert format == 'pandas' or not melt, \
        'melt=True is only supported when format="pandas"'
    df = _fetch_query(server, uuid, instance, show=show, fields=fields, status_categories=status_categories, format=format, session=session, endpoint='all')
    if melt:
        df = _melt_all(df)
    return df

@wraps(_fetch_query)
def fetch_query(server, uuid, instance='segmentation_annotations', query=None, **kwargs):
    return _fetch_query(server, uuid, instance, query, endpoint='query', **kwargs)


@dvid_api_wrapper
def fetch_schema(server, uuid, instance, *, session=None):
    r = session.get(f"{server}/api/node/{uuid}/{instance}/schema")
    r.raise_for_status()
    return r.json()


@dvid_api_wrapper
def fetch_json_schema(server, uuid, instance, *, session=None):
    r = session.get(f"{server}/api/node/{uuid}/{instance}/json_schema")
    r.raise_for_status()
    return r.json()


@dvid_api_wrapper
def post_json_schema(server, uuid, instance, schema, *, session=None):
    assert '$schema' in schema
    r = session.post(f"{server}/api/node/{uuid}/{instance}/json_schema", json=schema)
    r.raise_for_status()


@dvid_api_wrapper
def fetch_schema_batch(server, uuid, instance, *, session=None):
    r = session.get(f"{server}/api/node/{uuid}/{instance}/schema_batch")
    r.raise_for_status()
    return r.json()


@dvid_api_wrapper
def post_json_schema(server, uuid, instance, schema, *, session=None):
    r = session.post(f"{server}/api/node/{uuid}/{instance}/json_schema", json=schema)
    r.raise_for_status()

