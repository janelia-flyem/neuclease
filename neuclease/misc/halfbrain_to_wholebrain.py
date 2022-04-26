import logging
from functools import wraps

import numpy as np
import pandas as pd
from requests import HTTPError

from neuclease.util import Timer, tqdm_proxy as tqdm, round_coord
from neuclease.dvid import fetch_sparsevol, fetch_labelmap_specificblocks
from neuclease.dvid.rle import blockwise_masks_from_ranges


class FindHBWBOverlaps:
    """
    Find overlapping segments between our halfbrain
    segmentation and the whole-brain segmentation.

    Example:

        from neuclease.misc.halfbrain_to_wholebrain import find_hbwb_overlaps
        counts = find_hbwb_overlaps(2457385246, scale=3, supervoxels=False, show_progress=True)

    Multiprocessing Example:

        from functools import partial
        import pandas as pd
        from neuclease.util import compute_parallel

        fn = partial(find_hbwb_overlaps, scale=6, show_progress=False)
        all_counts = compute_parallel(fn, my_body_list, processes=16)
        all_counts_df = pd.concat(all_counts, ignore_index=True)
        all_counts_df = all_counts_df.sort_values(['halfbrain_body', 'count'], ascending=[True, False])
    """
    def __init__(self, halfbrain_seg=None, brain_seg=None):
        self.halfbrain_seg = halfbrain_seg
        self.brain_seg = brain_seg
        if halfbrain_seg is None:
            self.halfbrain_seg = (
                'http://emdata6.int.janelia.org:8000',
                'eaef6946397647d3a6889568c2f96f4b',
                'segmentation'
            )

        self.brain_seg = brain_seg
        if brain_seg is None:
            self.brain_seg = (
                'http://emdata6.int.janelia.org:9000',
                '138069aba8e94612b37d119778a89a1c',
                'segmentation'
            )

    def find_hbwb_overlaps(self, halfbrain_body, scale=0, supervoxels=False, threshold_frac=0.05, show_progress=True):
        """
        For a given body (or supervoxel) in our half-brain segmentation, determine
        which bodies it overlaps with in out whole-brain segmentation.

        Returns a table of whole-brain body IDs along with the count of
        overlapping voxels and fraction of the original body.

        Note:
            The results will include body 0 if the body's mask overlaps
            with label 0 in the whole-brain segmentation.
        """
        # Notes:
        # - The last tab of the halfbrain (starting at 35202 in the half-brain),
        #   was realigned, so we don't look at that region.
        # - Same for the first tab of the halfbrain, ending at 6772
        # - The translation offet between half-brain and whole-brain is (4096, 4096, 4096)
        HALFBRAIN_FIRST_TAB_X = round_coord(6772, 64, 'up') // 2**scale
        HALFBRAIN_LAST_TAB_X = round_coord(35202, 64, 'down') // 2**scale

        if show_progress:
            log_level = logging.INFO
        else:
            log_level = logging.DEBUG

        label_type = {True: 'sv', False: 'body'}[supervoxels]

        def empty_counts():
            """Return an empty dataframe with the appropriate columns."""
            empty_index = pd.Index(np.array([], dtype=np.uint64), name=f'brain_{label_type}')
            counts = pd.Series([], name='count', dtype=np.int64, index=empty_index)
            df = counts.to_frame()
            df['halfbrain_frac'] = 0.0
            df[f'halfbrain_{label_type}'] = halfbrain_body
            return df.reset_index()[[f'halfbrain_{label_type}', f'brain_{label_type}', 'count', 'halfbrain_frac']]

        with Timer(f"Body {halfbrain_body}: Fetching sparsevol", None, log_level):
            try:
                rng = fetch_sparsevol(*self.halfbrain_seg, halfbrain_body, scale=scale, supervoxels=supervoxels, format='ranges')
                body_size = (rng[:, 3] - rng[:, 2] + 1).sum()
            except HTTPError:
                return empty_counts()

        with Timer(f"Body {halfbrain_body}: Setting up masks", None, log_level):
            boxes, masks = blockwise_masks_from_ranges(rng, (64,64,64))

        # Fast path for objects that lie completely within the taboo region.
        if (boxes[:, 1, 2] <= HALFBRAIN_FIRST_TAB_X).all():
            return empty_counts()
        if (boxes[:, 0, 2] >= HALFBRAIN_LAST_TAB_X).all():
            return empty_counts()

        with Timer(f"Body {halfbrain_body}: Fetching specificblocks", None, log_level):
            brain_corners = 4096 // (2**scale) + boxes[:, 0, :]
            seg_dict = fetch_labelmap_specificblocks(*self.brain_seg, brain_corners, scale=scale, supervoxels=supervoxels, format='callable-blocks')
            assert len(boxes) == len(seg_dict), \
                f"Body {halfbrain_body}: Mismatch between masks and seg: {len(boxes)} != {len(seg_dict)}"

        with Timer(f"Body {halfbrain_body}: Counting voxels", None, log_level):
            block_counts = []
            seg_items = seg_dict.items()
            mask_items = zip(boxes, masks)
            for (mask_box, mask), (seg_corner, compressed_seg) in tqdm(zip(mask_items, seg_items), disable=not show_progress, total=len(seg_items)):
                assert (mask_box[0] + 4096 // 2**scale == seg_corner).all(), \
                    f"Body {halfbrain_body}: Mask corner doesn't match seg_corner: {mask_box[0]} != {seg_corner}"
                if mask_box[1, 2] <= HALFBRAIN_FIRST_TAB_X:
                    continue
                if mask_box[0, 2] >= HALFBRAIN_LAST_TAB_X:
                    continue

                OUT_OF_MASK = 2**63
                seg = np.where(mask, compressed_seg(), OUT_OF_MASK)
                vc = pd.value_counts(seg.ravel()).rename_axis(f'brain_{label_type}').rename('count')
                vc = vc[vc.index != OUT_OF_MASK]
                block_counts.append(vc)

        if len(block_counts) == 0:
            return empty_counts()

        counts = pd.concat(block_counts).groupby(level=0).sum()
        counts.sort_values(ascending=False, inplace=True)
        df = counts.to_frame()
        df['halfbrain_frac'] = df['count'] / body_size
        df['halfbrain_frac_within_frozen'] = df['count'] / df['count'].sum()
        df = df[df['halfbrain_frac'] >= threshold_frac]
        df[f'halfbrain_{label_type}'] = halfbrain_body
        return df.reset_index()[[f'halfbrain_{label_type}', f'brain_{label_type}', 'count', 'halfbrain_frac']]

    @wraps(find_hbwb_overlaps)
    def __call__(self, *args, **kwargs):
        return self.find_hbwb_overlaps(*args, **kwargs)


find_hbwb_overlaps = FindHBWBOverlaps()


if __name__ == '__main__':
    #
    # debug stuff...
    #
    brain_server = 'http://emdata6.int.janelia.org:9000'
    brain_root = (brain_server, 'f3969dc575d74e4f922a8966709958c8')
    brain_agglo = (brain_server, '138069aba8e94612b37d119778a89a1c')
    brain_import = (brain_server, 'd26e15af06784aedbfcac1ff82684b51')
    halfbrain_base = ('http://emdata6.int.janelia.org:8000', '0ea0d0e28874440a93bf156f9fbd65ca')

    find_hbwb_overlaps = FindHBWBOverlaps(
        (*halfbrain_base, 'segmentation'),
        (*brain_import, 'segmentation')
    )

    counts = find_hbwb_overlaps(1467281664, scale=3, supervoxels=False, show_progress=True)
    print(counts)