from . import sanitize_server, fetch_generic_json

@sanitize_server
def fetch_server_info(server):
    return fetch_generic_json(f'http://{server}/api/server/info')


