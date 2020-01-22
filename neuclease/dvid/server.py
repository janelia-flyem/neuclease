from . import dvid_api_wrapper, fetch_generic_json


@dvid_api_wrapper
def fetch_info(server, *, session=None):
    """
    Fetch the server info.
    
    See also: ``neuclease.dvid.wrapper_proxies.fetch_info()``
    """
    return fetch_generic_json(f'{server}/api/server/info', session=session)

# Synonym
fetch_server_info = fetch_info

@dvid_api_wrapper
def reload_metadata(server, *, session=None):
    r = session.post(f"{server}/api/server/reload-metadata")
    r.raise_for_status()


