import numpy as np
from numba import jit

# Michal's descriptions of various field names used below:
#
# id_a, id_b -- the two supervoxel IDs
# xa, ya, za -- point from which segmentation of 'a' was started, 8 nm coordinates
# xb, yb, zb -- point from which segmentation of 'b' was started, 8 nm coordinates
# caa, cab, cba, cbb -- cXY means: fraction of voxels from the original segment Y recovered when seeding from X
# iou -- Jaccard index of the two local segmentations
# da, db -- dX means: fraction of voxels that changed value from >0.8 to <0.5 when segmenting & seeding from X;
#                     the higher this value is, the more "internally inconsistent" the segmentation resolution
#                     potentially is; higher thresholds for iou, cXY might be warranted


@jit
def calc_speculative_scores(table):
    """
    Michal's formula for computing speculative merge scores.
    
    Args:
        table: A np.ndarray with fields 'caa', 'cab', 'cba', 'cbb'
    
    Notes:
        - Higher is better!
        - Initially we are using a threshold of >=0.1
    
    From Email:
        Subject: focused proofreading iteration 1 constraints
        Date: Wednesday, July 25, 2018 at 11:38 AM
    """
    scores = np.empty(len(table), np.float32)
    for i in range(len(table)):
        caa = table['caa'][i]
        cab = table['cab'][i]
        cba = table['cba'][i]
        cbb = table['cbb'][i]
        scores[i] = max(min(caa, cab), min(cba, cbb))
    return scores



##
## FFN Agglo score calculation (not used by us)
##

@jit(nopython=True)
def calc_score_32nm(caa, cab, cba, cbb, iou, da, db):
    if (caa >= 0.6 and cab >= 0.6 and cba >= 0.6 and cbb >= 0.6
    and iou > 0.8 and (da <= 0.02 or db <= 0.02)):
        return 0.0 + (1.0 - iou)
    elif (caa >= 0.6 and cab >= 0.6 and cba >= 0.6 and cbb >= 0.6 and iou > 0.4):
        return 5.0 + (1.0 - iou)
    elif (caa > 0.8 and cab > 0.8):
        return 11.0 + (1.0 - min(caa, cab))
    else:
        return 11.0 + (1.0 - min(cba, cbb))


@jit(nopython=True)
def calc_score_16nm(caa, cab, cba, cbb, iou, da, db):
    if ( caa >= 0.6 and cab >= 0.6 and cba >= 0.6 and cbb >= 0.6
    and iou > 0.8 and (da <= 0.02 or db <= 0.02)):
        # This is probably wrong...
        return 3.0 + (1.0 - iou)

# This is the original function...
#         if d.id_b = 0 or e.id_b = 0:
#             if b.class == 6 and c.class == 6:
#                 return 1.0 + (1.0 - iou)
#             else:
#                 return 2.0 + (1.0 - iou)
#         else:
#             if b.class = 6 and c.class = 6:
#                 return 3.0 + (1.0 - iou)
#             else:
#                 return 4.0 + (1.0 - iou)
    else:
        if caa >= 0.6 and cab >= 0.6 and cba >= 0.6 and cbb >= 0.6 and iou >= 0.4:
            # This is probably wrong...
            return 8.0 + (1.0 - iou)

# This is the original function...
#             if d.id_b = 0 or e.id_b = 0:
#                 if b.class = 6 and c.class = 6:
#                     return 6.0 + (1.0 - iou),
#                 else:
#                     return 7.0 + (1.0 - iou)
#             else:
#                 if b.class = 6 and c.class = 6:
#                     return 8.0 + (1.0 - iou)
#                 else:
#                     return 9.0 + (1.0 - iou)
        else:
            if caa > 0.9 and cab > 0.9:
                return 12.0 + (1.0 - min(caa, cab))
            else:
                return 12.0 + (1.0 - min(cba, cbb))


@jit(nopython=True)
def calc_score_8nm(caa, cab, cba, cbb, iou, da, db):
    return 10.0 + (1.0 - iou)


@jit(nopython=True)
def calc_score(resolution, caa, cab, cba, cbb, iou, da, db):
    if resolution == 32:
        return calc_score_32nm(caa, cab, cba, cbb, iou, da, db)
    if resolution == 16:
        return calc_score_16nm(caa, cab, cba, cbb, iou, da, db)
    if resolution == 8:
        return calc_score_8nm(caa, cab, cba, cbb, iou, da, db)
    return np.inf
        
@jit(nopython=True)
def calc_agglo_scores(table):
    scores = np.empty(len(table), np.float32)
    for i in range(len(table)):
        resolution = table['resolution'][i]
        caa = table['caa'][i]
        cab = table['cab'][i]
        cba = table['cba'][i]
        cbb = table['cbb'][i]
        iou = table['iou'][i]
        da = table['da'][i]
        db = table['db'][i]
        scores[i] = calc_score(resolution, caa, cab, cba, cbb, iou, da, db)
    return scores