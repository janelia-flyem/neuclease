import numpy
from  scipy.ndimage.morphology import binary_dilation

def find_best_plane(segchunk, label1, label2):
    """
    Find best plane to view edge between label1 and label2.
    (currently just finds two points close by an 'edgy' location, presumably
    some plane parallel to the line connection pt1 and pt2 centered on either
    pt1 or pt2 is a reasonable location)
    
    Author: Stephen Plaza
    
    Args:
        segchunk (3D numpy label array):
            small array that presumably contains the edge between label1 and label2
            Note: Must be a cube
        
        label1: (number) 
             id1
        label2 :(number)
            id2
    Returns:
        Points for label1 and label2
        (z,y,z), (z,y,x) 
        If there is no edge found, all coordinates are -1
    """
    # dilate label 1
    dilatelabel1arr = binary_dilation(segchunk == label1)
    label2arr = segchunk == label2
    # find label2 locations that overlap dilated label1
    # TODO?: iteratively dilate until an intersection is found
    label2arr[dilatelabel1arr == 0] = 0
    if numpy.count_nonzero(label2arr) == 0:
        return  (-1,-1,-1), (-1,-1,-1)
    # find point with the most nearby edge
    # (ideally this should probably using 2D planes but since
    # we have arbitrary cut-planes it might be better to optimize for most
    # edge points a given 3D region)
    def maxoct(chunk):
        zsize,ysize,xsize = chunk.shape
        zstep = zsize//2
        ystep = ysize//2
        xstep = xsize//2
        maxcount = 0
        bestpt = None
        newchunk = None
        for ziter in range(0,zsize,zstep):
            for yiter in range(0,ysize,ystep):
                for xiter in range(0,xsize,xstep):
                    zmax = zsize // 2
                    ymax = ysize // 2
                    xmax = xsize // 2
                    if ziter > 0 or zmax == 0:
                        zmax = zsize
                    if yiter > 0 or ymax == 0:
                        ymax = ysize
                    if xiter > 0 or xmax == 0:
                        xmax = xsize
                    
                    count = numpy.count_nonzero(chunk[ziter:zmax,yiter:ymax,xiter:xmax]) 
                    if count > maxcount:
                        maxcount = count
                        bestpt = ziter,yiter,xiter
                        newchunk = chunk[ziter:zmax,yiter:ymax,xiter:xmax]
        return newchunk, bestpt 
    currchunk = label2arr
    label2location = 0,0,0
    while currchunk.shape != (1,1,1):
        currchunk, newoffset = maxoct(currchunk)
        label2location = label2location[0] + newoffset[0], \
                            label2location[1] + newoffset[1], \
                            label2location[2] + newoffset[2]
    # extract plane that separates two labels
    # the current best location should be on label2 by construction
    assert(segchunk[label2location] == label2)
    # find label1 nearby
    z,y,x =  label2location
    zmax,ymax,xmax = segchunk.shape
    label1location = -1,-1,-1
    for ziter in range(z-1, z+2):
        for yiter in range(y-1, y+2):
            for xiter in range(x-1, x+2):
                if ziter < 0 or yiter < 0 or xiter < 0:
                    continue
                if ziter == zmax or yiter == ymax or xiter == xmax:
                    continue
                if segchunk[ziter,yiter,xiter] == label1:
                    label1location = ziter,yiter,xiter
                
    assert(label1location != (-1,-1,-1))
    
    return label1location, label2location

if __name__ == "__main__":
    #####################################
    ### Simple test
    from libdvid import DVIDNodeService
    ns = DVIDNodeService("emdata3:8900", "a776")
    # find edge by label1=330009078 and label2=330345807
    labels = ns.get_labels3D("segmentation", (256,256,256), (4600, 25400, 22250))
    pt1, pt2 = findBestPlane(labels, 330009078, 330345807)
    print((pt1[0]+4600,pt1[1]+25400,pt1[2]+22250),(pt2[0]+4600,pt2[1]+25400,pt2[2]+22250))
