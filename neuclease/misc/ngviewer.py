import sys
import copy
import logging
from textwrap import dedent

import numpy as np
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


def update_layers(viewer, clear=False, axes='zyx', units='nm', scales=[8,8,8], volume_type=None, voxel_offset=(0,0,0), segments=None, **volumes):
    """
    Args:
        segments:
            List of segment IDs to show as selected in the layer.
            Should only be used with segmentation layers.
            Will be applied to all layers, so you should only use this with
            multiple layers if they happen to have corresponding layer IDs.

        volume_type:
            Either 'image' or 'segmentation'. If None, neuroglancer guesses from dtype
    """
    with viewer.txn() as s:
        if clear:
            s.layers.clear()

        kwargs = {}
        if segments is not None:
            kwargs['segments'] = [*map(str, segments)]

        for name, vol in volumes.items():
            cspace = CoordinateSpace(names=[*axes], units='nm', scales=scales)
            s.layers[name] = ManagedLayer(name, LocalVolume(vol, cspace, volume_type, voxel_offset), **kwargs)


def update_seg_layer(v, name, vol, scale, box, res0=8, segments=None):
    """
    Args:
        segments:
            List of segment IDs to show as selected in the layer.
            Should only be used with segmentation layers.
            Will be applied to all layers, so you should only use this with
            multiple layers if they happen to have corresponding layer IDs.
    """
    scales = (2**scale)*np.array([res0,res0,res0])
    layers = {name: vol}
    update_layers(v, False, 'zyx', 'nm', scales, 'segmentation', box[0], segments, **layers)


def update_img_layer(v, name, vol, scale, box, res0=8):
    scales = (2**scale)*np.array([res0,res0,res0])
    layers = {name: vol}
    update_layers(v, False, 'zyx', 'nm', scales, 'image', box[0], **layers)


def update_mask_layer(v, name, vol, scale, box, res0=8):
    """
    Add an image layer for the given binary volume,
    with the following appearance:
    - transparent where vol == 1
    - dark everywhere else
    """
    vol = vol.astype(bool).view(np.uint8)
    update_img_layer(v, name, vol, scale, box, res0)
    s = copy.deepcopy(v.state)
    s.layers[name].opacity = 0.5
    s.layers[name].shader = dedent("""\
        void main() {
        uint8_t v = getDataValue();
            if (float(v.value) != 0.0) {
            emitTransparent();
            }
        else {
            emitGrayscale(toNormalized(v));
        }
        }
    """)
    v.set_state(s)


def quickview(*unnamed_vols, port=8080, **named_volumes):
    import time
    import webbrowser
    import neuroglancer.server
    init_ngserver('localhost', port=port)
    viewer = create_viewer()
    url = 'http://localhost:' + viewer.get_viewer_url().split(':')[2]
    print(url)
    webbrowser.open(url)

    volumes = {}
    for i, vol in enumerate(unnamed_vols):
        volumes[f'volume_{i}'] = vol
    volumes.update(named_volumes)
    update_layers(viewer, **volumes)

    try:
        while True:
            # Wait for KeyboardInterrupt
            time.sleep(1.0)
    finally:
        neuroglancer.server.stop()


if __name__ == '__main__':
    import os.path
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=8080, type=int)
    parser.add_argument('volumes', nargs='+')
    args = parser.parse_args()

    volumes = {}
    for path in args.volumes:
        name = os.path.basename(path).replace('_', '_').replace(' ', '_')
        volumes[name] = np.load(path)

    quickview(args.port, **volumes)
