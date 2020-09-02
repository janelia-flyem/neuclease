from collections.abc import Mapping
from . import dvid_api_wrapper


@dvid_api_wrapper
def post_tags(server, uuid, instance, tags, replace=False, *, session=None):
    """
    Post tags to the metadata of either a keyvalue instance or annotation instance.

    Args:
        server:
            dvid server, e.g. 'emdata3:8900'

        uuid:
            dvid uuid, e.g. 'abc9'

        instance:
            keyvalue or annotation instance name, e.g. 'focused_merged'

        tags:
            A dictionary of tag {names : values}.  Tag names must be strings.
            Tag values will be converted to strings.

        replace:
            If True, erase existing tags and replace them with these new ones.
            Otherwise, just append these new tags to the existing ones.
    """
    assert isinstance(tags, Mapping)

    # Tags must be strings, so convert.
    for k in tags.keys():
        assert isinstance(k, str), "tag names must be strings"
        tags[k] = str(tags[k])

    params = {}
    if replace:
        params['replace'] = 'true'

    r = session.post(f'{server}/api/node/{uuid}/{instance}/tags', json=tags, params=params)
    r.raise_for_status()
