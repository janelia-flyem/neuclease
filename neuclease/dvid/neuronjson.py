from functools import wraps
from . import dvid_api_wrapper
from .keyvalue._keyvalue import _body_annotations_dataframe


@dvid_api_wrapper
def _fetch_query(server, uuid, instance='segmentation_annotations', query=None, endpoint='query', *, show=None, fields=None, status_categories=None, format='pandas', session=None):
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
        return _body_annotations_dataframe(values, status_categories)
    else:
        return sorted(values, key=lambda d: d['bodyid'])


@wraps(_fetch_query)
def fetch_all(*args, **kwargs):
    return _fetch_query(*args, **kwargs, endpoint='all')


@wraps(_fetch_query)
def fetch_query(server, uuid, instance='segmentation_annotations', query=None, **kwargs):
    return _fetch_query(server, uuid, instance, query, **kwargs)


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
def fetch_schema_batch(server, uuid, instance, *, session=None):
    r = session.get(f"{server}/api/node/{uuid}/{instance}/schema_batch")
    r.raise_for_status()
    return r.json()
