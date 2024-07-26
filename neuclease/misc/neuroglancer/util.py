import re
import json
import urllib
import logging

from .storage import download_ngstate

logger = logging.getLogger(__name__)


def parse_nglink(link):
    _, pseudo_json = link.split('#!')
    if pseudo_json.endswith('.json'):
        return download_ngstate(pseudo_json)
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return data


def format_nglink(ng_server, link_json_settings):
    return ng_server + '/#!' + urllib.parse.quote(json.dumps(link_json_settings))


def layer_dict(state):
    return {layer['name']: layer for layer in state['layers']}


def layer_state(state, name):
    layer = None
    matches = []
    for layer in state['layers']:
        if re.match(name, layer['name']):
            matches.append(layer)
    if len(matches) > 1:
        raise RuntimeError(f"Found more than one layer matching to the regex '{name}'")
    return layer
