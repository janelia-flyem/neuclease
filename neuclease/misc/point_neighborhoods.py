from functools import lru_cache
from collections.abc import Iterable

import numpy as np
import pandas as pd

from vol2mesh import Mesh

from neuclease.util import round_box, box_to_slicing as b2s, tqdm_proxy
from neuclease.dvid import post_key
from neuclease.dvid.labelmap import fetch_labelmap_voxels, post_labelmap_voxels


def write_point_neighborhoods(seg_src, seg_dst, points_zyx, radius=125, src_bodies=None, dst_bodies=None):\
    """
    For each point in the given list, create a mask of the portion
    of a particular body that falls within a given distance of the
    point.

    Args:
        seg_src:
            tuple (server, uuid, instance) specifying where to fetch neuron segmentation from.
        seg_dst:
            tuple (server, uuid, instance) specifying where to write neighborhood segmentation to.
        points_zyx:
            Array of coordinates to create neighborhoods around
        radius:
            Radius (in voxels) of the neighborhood to create
        src_bodies:
            Either a single body ID or a list of body IDs (corresponding to the list of points_zyx).
            Specifies which body the neighborhood around each point should be constructed for.
            If not provided, the body for each neighborhood will be chosen automatically,
            by determining which body each point in points_zyx falls within.
        dst_bodies:
            List of new body IDs.
            Specifies the IDs to use as the 'body ID' for the neighborhood segments when writing to
            the destination instance.  If no list is given, then new body IDs are automatically
            generated with a formula that uses the coordinate around which the neighborhood was created.
            Note that the default formula does not take the source body into account,
            so if there are duplicate points provided in points_zyx, the destination body IDs will
            be duplicated too, unless you supply your own destination body IDs here.

    Returns:
        In addition to writing the neighborhood segments to the seg_dst instance,
        this function returns a dataframe with basic stats about the neighborhoods
        that were written.
    """
    if isinstance(points_zyx, pd.DataFrame):
        points_zyx = points_zyx[[*'zyx']].values
    else:
        points_zyx = np.asarray(points_zyx)

    results = []
    for i, point in enumerate(tqdm_proxy(points_zyx)):
        if isinstance(src_bodies, Iterable):
            src_body = src_bodies[i]
        else:
            src_body = src_bodies

        if isinstance(dst_bodies, Iterable):
            dst_body = dst_bodies[i]
        else:
            dst_body = dst_bodies

        src_body, dst_body, dst_voxels = process_point(seg_src, seg_dst, point, radius, src_body, dst_body)
        results.append( (src_body, dst_body, dst_voxels) )

    results_df = pd.DataFrame(results, columns=['src_body', 'dst_body', 'dst_voxels'])
    results_df['z'] = points_zyx[:, 0]
    results_df['y'] = points_zyx[:, 1]
    results_df['x'] = points_zyx[:, 2]

    return results_df[[*'xyz', 'src_body', 'dst_body', 'dst_voxels']]


def process_point(seg_src, seg_dst, point, radius, src_body, dst_body):
    r = radius
    src_box = np.asarray(( point - r, point + r + 1 ))
    src_vol = fetch_labelmap_voxels(*seg_src, src_box)

    if src_body is None:
        src_body = src_vol[r,r,r]

    if dst_body is None:
        dst_body = encode_point_to_uint64(point // 4, 17)

    mask = (src_vol == src_body) & sphere_mask(r)

    dst_box = round_box(src_box, 64, 'out')
    dst_vol = fetch_labelmap_voxels(*seg_dst, dst_box)

    dst_view = dst_vol[b2s(*(src_box - dst_box[0]))]
    dst_view[mask] = dst_body

    post_labelmap_voxels(*seg_dst, dst_box[0], dst_vol, downres=True)

    # Mesh needs to be written in nm, hence 8x
    mesh = Mesh.from_binary_vol(mask, 8*src_box, smoothing_rounds=2)
    mesh.simplify(0.05, in_memory=True)
    post_key(*seg_dst[:2], f'{seg_dst[2]}_meshes', f'{dst_body}.ngmesh', mesh.serialize(fmt='ngmesh'))

    return src_body, dst_body, mask.sum()


@lru_cache(maxsize=1)
def sphere_mask(radius):
    r = radius
    cz, cy, cx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    distances = np.sqrt(cz**2 + cy**2 + cx**2)
    return (distances <= r)


def encode_point_to_uint64(point_zyx, bitwidth):
    point_zyx = point_zyx.astype(np.uint64)
    assert 3*bitwidth <= 53, "You shouldn't use body values greater than 2^53"
    encoded = np.uint64(0)
    encoded |= point_zyx[0] << np.uint64(2*bitwidth)
    encoded |= point_zyx[1] << np.uint64(bitwidth)
    encoded |= point_zyx[2]
    return encoded


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        from neuclease.dvid.annotation import fetch_annotation_label
        syn_df = fetch_annotation_label('emdata4:8900', 'e83362', 'synapses', 1071121755, format='pandas')

        seg_src = ('emdata3:8900', 'd31b6', 'segmentation')
        #seg_dst = ('emdata3:8900', 'd31b6', 'neighborhood-masks')
        #seg_dst = ('hemibrain-dvid2.janelia.org:8000', '1792aadcfc2641a7ba56c47606b59a5b', 'neighborhood-masks')
        seg_dst = ('hemibrain-dvid2.janelia.org:8000', '09ef592ae7c84201a107b14775b02dbb', 'neighborhood-masks-philip')

        results_df = write_point_neighborhoods(seg_src, seg_dst, syn_df.iloc[[590,667,1256]], 250)
        print(results_df.to_csv(index=False, header=True))

    """
    http://emdata4.int.janelia.org:8900/neuroglancer/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B29333%2C20810%2C15580%5D%2C%22crossSectionScale%22:0.6961921935554136%2C%22crossSectionDepth%22:-7.239642214791052%2C%22projectionOrientation%22:%5B0.46037495136260986%2C0.3384398818016052%2C0.0005697967717424035%2C0.8206784129142761%5D%2C%22projectionScale%22:799.9333989929243%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22blend%22:%22default%22%2C%22name%22:%22emdata%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata4.int.janelia.org:8900/20631f94c3f446d7864bc55bf515706e/segmentation%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:0.58%2C%22objectAlpha%22:0.34%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22segmentation%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata4.int.janelia.org:8900/20631f94c3f446d7864bc55bf515706e/mito_20190717.27250582%22%2C%22subsources%22:%7B%22default%22:true%2C%22skeletons%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito_20190717.27250582%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/62f6394a18d4490c93892fbd9f1416b5/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-sv%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:1%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-bodies%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:1%2C%22saturation%22:0%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-bodies-white%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/neighborhood-masks%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22objectAlpha%22:0.45%2C%22segments%22:%5B%2266916272315557%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22neighborhood-masks%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22neighborhood-masks%22%2C%22visible%22:true%7D%2C%22layout%22:%224panel%22%7D
    http://emdata4.int.janelia.org:8900/neuroglancer/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B22873.3359375%2C20451.51953125%2C15845.8095703125%5D%2C%22crossSectionScale%22:0.6961921935554136%2C%22crossSectionDepth%22:-7.239642214791052%2C%22projectionOrientation%22:%5B0.19774292409420013%2C0.2364424765110016%2C-0.24986568093299866%2C0.9179105758666992%5D%2C%22projectionScale%22:1610.8680480180612%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22blend%22:%22default%22%2C%22name%22:%22emdata%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata4.int.janelia.org:8900/20631f94c3f446d7864bc55bf515706e/segmentation%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:0.58%2C%22objectAlpha%22:0.34%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22segmentation%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata4.int.janelia.org:8900/20631f94c3f446d7864bc55bf515706e/mito_20190717.27250582%22%2C%22subsources%22:%7B%22default%22:true%2C%22skeletons%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito_20190717.27250582%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/62f6394a18d4490c93892fbd9f1416b5/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-sv%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:1%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-bodies%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:1%2C%22saturation%22:0%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-bodies-white%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/neighborhood-masks%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22objectAlpha%22:0.45%2C%22segments%22:%5B%2268050132014678%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22neighborhood-masks%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22neighborhood-masks%22%2C%22visible%22:true%7D%2C%22layout%22:%224panel%22%7D
    http://emdata4.int.janelia.org:8900/neuroglancer/#!%7B%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B30074%2C22562%2C17345%5D%2C%22crossSectionScale%22:0.6961921935554136%2C%22crossSectionDepth%22:-7.239642214791052%2C%22projectionOrientation%22:%5B-0.27656280994415283%2C0.39428943395614624%2C-0.8697095513343811%2C-0.10795421153306961%5D%2C%22projectionScale%22:977.040859874398%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22blend%22:%22default%22%2C%22name%22:%22emdata%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata4.int.janelia.org:8900/20631f94c3f446d7864bc55bf515706e/segmentation%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:0.58%2C%22objectAlpha%22:0.34%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22segmentation%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata4.int.janelia.org:8900/20631f94c3f446d7864bc55bf515706e/mito_20190717.27250582%22%2C%22subsources%22:%7B%22default%22:true%2C%22skeletons%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito_20190717.27250582%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/62f6394a18d4490c93892fbd9f1416b5/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-sv%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:1%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-bodies%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/masked-mito-cc%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22selectedAlpha%22:1%2C%22saturation%22:0%2C%22segments%22:%5B%221071121755%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22mito-bodies-white%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/neighborhood-masks%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22objectAlpha%22:0.45%2C%22segments%22:%5B%2274492652035422%22%5D%2C%22skeletonRendering%22:%7B%22mode2d%22:%22lines_and_points%22%2C%22mode3d%22:%22lines%22%7D%2C%22name%22:%22neighborhood-masks%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22neighborhood-masks%22%2C%22visible%22:true%7D%2C%22layout%22:%224panel%22%7D
    """

    """
    {
    "file version": 1,
    "grayscale source": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
    "mito ROI source": "dvid://http://emdata3:8900/d31b64ac81444923a0319961736a6c31/neighborhood-masks",
    "DVID source": "https://hemibrain-dvid2.janelia.org/#/repo/09ef592ae7c84201a107b14775b02dbb",
    "task list": [
        {
        "task type": "mito count",
        "focal point": [29333, 20810, 15580]
        },
        {
        "task type": "mito count",
        "focal point": [22873, 20452, 15845]
        },
        {
        "task type": "mito count",
        "focal point": [30074, 22562, 17345]
        }
    ]
    }
    """
