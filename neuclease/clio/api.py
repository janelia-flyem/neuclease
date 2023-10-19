import pandas as pd

from . import clio_api_wrapper

# Clio API Swagger docs can be found here:
# https://clio-store-vwzoicitea-uk.a.run.app/docs


@clio_api_wrapper
def fetch_datasets(*, base=None, session=None):
    r = session.get(f'{base}/v2/datasets')
    r.raise_for_status()
    return r.json()


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
def fetch_json_annotations_all(dataset, annotation_type='neurons', *, fields=None, show=None, format='pandas', base=None, session=None):
    assert fields is None, "At this time, the 'fields' argument is not supported by the clio service.  (DVID does support it, though.)"
    assert show in ('user', 'time', 'all', None)
    assert format in ('pandas', 'json')

    params = {}
    if show:
        params['show'] = show

    if fields:
        if isinstance(fields, str):
            fields = [fields]
        params['fields'] = ','.join(fields)

    r = session.get(f"{base}/v2/json-annotations/{dataset}/{annotation_type}/all", params=params)
    r.raise_for_status()
    values = r.json() or []

    if format == 'pandas':
        return pd.DataFrame(values)
    else:
        return sorted(values, key=lambda d: d['bodyid'])


@clio_api_wrapper
def fetch_users(*, base=None, session=None):
    r = session.get(f"{base}/v2/users")
    r.raise_for_status()
    return r.json()
