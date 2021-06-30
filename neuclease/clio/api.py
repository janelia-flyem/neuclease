import pandas as pd

from . import clio_api_wrapper


@clio_api_wrapper
def fetch_annotations(dataset, user=None, format='pandas', *, base=None, session=None):
    assert format in ('json', 'pandas')
    if user:
        params = {"user": user}
    else:
        params = {}

    r = session.get(f"{base}/v2/annotations/{dataset}", params=params)
    r.raise_for_status()

    if format == 'json':
        return r

    if format == 'pandas':
        return pd.DataFrame(r.json())

@clio_api_wrapper
def fetch_json_annotations_all(dataset, annotation_type='neurons', format='pandas', *, base=None, session=None):
    assert format in ('json', 'pandas')
    r = session.get(f"{base}/v2/json-annotations/{dataset}/{annotation_type}/all")
    r.raise_for_status()

    if format == 'json':
        return r.json()

    if format == 'pandas':
        return pd.DataFrame(r.json())
