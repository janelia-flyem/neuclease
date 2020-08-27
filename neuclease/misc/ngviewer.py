import sys
import logging

import neuroglancer
from neuroglancer import Viewer, LocalVolume, CoordinateSpace

logger = logging.getLogger(__name__)


def init_ngserver(bind='0.0.0.0', port=8080):
    neuroglancer.set_server_bind_address('0.0.0.0', port)


def create_viewer():
    viewer = Viewer()
    try:
        import ipykernel.iostream
        from IPython.display import HTML, display
        if isinstance(sys.stdout, ipykernel.iostream.OutStream):
            display(HTML(f'<a href="{viewer.get_viewer_url()}">{viewer.get_viewer_url()}</a>'))
    except ImportError:
        logger.info(f"Created viewer {viewer.get_viewer_url()}")
    return viewer


def update_layers(viewer, clear=True, axes='zyx', units='nm', scales=[8,8,8], **volumes):
    cnames = [*axes][::-1]
    scales = scales[::-1]
    with viewer.txn() as s:
        if clear:
            s.layers.clear()

        for name, vol in volumes.items():
            cspace = CoordinateSpace(names=cnames, units=units, scales=scales)
            layer = LocalVolume(vol.transpose(), cspace)
            s.layers.append(name=name, layer=layer)


