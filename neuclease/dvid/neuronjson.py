from . import dvid_api_wrapper, fetch_generic_json
from .keyvalue._keyvalue import _body_annotations_dataframe


@dvid_api_wrapper
def fetch_all(server, uuid, instance='segmentation_annotations', show=None, *, status_categories=None, format='pandas', session=None):
    params = None
    assert show in ('user', 'time', 'all', None)
    if show:
        params = {'show': show}

    url = f'{server}/api/node/{uuid}/{instance}/all'
    values = fetch_generic_json(url, params=params, session=session)
    return _body_annotations_dataframe(values, status_categories)


