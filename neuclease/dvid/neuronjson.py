from . import dvid_api_wrapper
from .keyvalue._keyvalue import _body_annotations_dataframe


@dvid_api_wrapper
def fetch_all(server, uuid, instance='segmentation_annotations', *, show=None, fields=None, status_categories=None, format='pandas', session=None):
    assert show in ('user', 'time', 'all', None)
    assert format in ('pandas', 'json')

    params = {}
    if show:
        params['show'] = show

    if fields:
        if isinstance(fields, str):
            fields = [fields]
        params['fields'] = ','.join(fields)

    url = f'{server}/api/node/{uuid}/{instance}/all'
    r = session.get(url, params=params)
    r.raise_for_status()
    values = r.json() or []

    if format == 'pandas':
        return _body_annotations_dataframe(values, status_categories)
    else:
        return sorted(values, key=lambda d: d['bodyid'])
