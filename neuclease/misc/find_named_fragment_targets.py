import numpy as np
import pandas as pd
from skimage.morphology import binary_dilation

from neuclease.util import compute_parallel
from neuclease.dvid import fetch_body_annotations, fetch_sparsevol_coarse


FRAGMENT_DILATION = 4

# Specify UUID
brain_master = ('emdata6.int.janelia.org:9000', 'c4ba4be')
brain_seg = (*brain_master, 'segmentation')


def _fetch_svc(body, dilate=0):
    """
    Fetch the sparse-coarsevol (svc) for a given body.

    Args:
        body:
            body ID
        dilate:
            If provided, expand (dilate) the SVC in
            all dimensions by the given radius.

    Returns:
        DataFrame with columns [z, y, x, body]
    """
    if dilate == 0:
        svc = fetch_sparsevol_coarse(*brain_seg, body, format='coords')
    else:
        mask, box = fetch_sparsevol_coarse(*brain_seg, body, format='mask')
        mask = np.pad(mask, dilate)
        box += [[-dilate, -dilate, -dilate],
                [ dilate,  dilate,  dilate]]
        assert (box[1] - box[0] == mask.shape).all()
        mask = binary_dilation(mask, np.ones(3 * [1 + 2 * dilate], bool))
        svc = box[0] + np.array(mask.nonzero()).transpose()

    df = pd.DataFrame(svc, columns=[*'zyx'])
    df = df.drop_duplicates()
    df['body'] = body
    return df


def _fetch_svc_and_dilate(body):
    return _fetch_svc(body, FRAGMENT_DILATION)


def main():
    # Fetch annotations from DVID
    ann = fetch_body_annotations(*brain_master)

    # Parse instances, select 'fragment' bodies, and determine their 'target' types.
    # Example 'fragment' instance: '(Dm9_fragment)'
    frag_rows = ann['instance'].fillna('').str.lower().str.contains('fragment')
    fragment_ann = ann.loc[frag_rows].copy()
    fragment_ann['target_type'] = fragment_ann['instance'].map(lambda s: s[1:].split('_')[0])

    # Select possible 'target' bodies (e.g. type Dm9)
    target_types = fragment_ann['target_type'].unique()
    target_ann = ann.query('type in @target_types and body not in @fragment_ann.index').copy()

    # Fetch all fragment coarse-sparsevols, but dilate them along the way.
    _dfs = compute_parallel(_fetch_svc_and_dilate, fragment_ann.index, processes=16)
    frag_svc_df = pd.concat(_dfs, ignore_index=True)
    frag_svc_df = frag_svc_df.merge(fragment_ann['target_type'], 'left', on='body')

    # Fetch target sparsevols (no dilation)
    _dfs = compute_parallel(_fetch_svc, target_ann.index, processes=16)
    target_svc_df = pd.concat(_dfs, ignore_index=True)
    target_svc_df = target_svc_df.merge(target_ann['type'], 'left', on='body')

    # Find all pairs of fragments and targets which share overlapping SVC coordinates.
    overlap_df = (
        frag_svc_df.rename(columns={'target_type': 'type'})
        .merge(target_svc_df, 'left', on=['type', *'zyx'], suffixes=['_frag', '_target'])

        # Replace NaN with 0 for easier filtering in the next step.
        .fillna(0)
        .astype({'body_target': int})
    )

    # Condense from (frag, target) pairs to (frag, [target, target, target, ...])
    final_target_lists = (
        overlap_df[['body_frag', 'body_target', 'type']]
        .drop_duplicates()
        .groupby(['type', 'body_frag'])['body_target']
        .agg(list)
        .sort_index()

        # Drop 0 placeholders, leaving empty lists
        # in cases where no targets were found.
        .map(lambda targets: [t for t in targets if t != 0])
    )

    final_target_lists.to_csv('named-fragment-dilated-target-lists-including-empty.csv', index=True, header=True)
    return final_target_lists


if __name__ == "__main__":
    final_target_lists = main()
