from . import dvid_api_wrapper, fetch_generic_json

@dvid_api_wrapper
def fetch_server_info(server):
    return fetch_generic_json(f'http://{server}/api/server/info')


