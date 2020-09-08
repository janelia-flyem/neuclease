import sys
import logging

import neuroglancer
from neuroglancer import Viewer, LocalVolume, CoordinateSpace, ManagedLayer, PointAnnotation

logger = logging.getLogger(__name__)


def init_ngserver(bind='0.0.0.0', port=8080):
    neuroglancer.set_server_bind_address('0.0.0.0', port)


def create_viewer(axes='xyz', units='nm', scales=[8,8,8]):
    viewer = Viewer()
    dimensions = CoordinateSpace(names=[*axes], units=units, scales=scales)
    with viewer.txn() as s:
        s.dimensions = dimensions

    try:
        import ipykernel.iostream
        from IPython.display import HTML, display
        if isinstance(sys.stdout, ipykernel.iostream.OutStream):
            display(HTML(f'<a href="{viewer.get_viewer_url()}">{viewer.get_viewer_url()}</a>'))
    except ImportError:
        logger.info(f"Created viewer {viewer.get_viewer_url()}")
    return viewer


def update_layers(viewer, clear=False, axes='zyx', units='nm', scales=[8,8,8], volume_type=None, voxel_offset=(0,0,0), **volumes):
    """
    Args:
        volume_type:
            Either 'image' or 'segmentation'. If None, neuroglancer guesses from dtype
    """
    with viewer.txn() as s:
        if clear:
            s.layers.clear()

        for name, vol in volumes.items():
            cspace = CoordinateSpace(names=[*axes], units='nm', scales=scales)
            s.layers[name] = ManagedLayer(name, LocalVolume(vol, cspace, volume_type, voxel_offset))
