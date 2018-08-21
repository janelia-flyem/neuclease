"""
Find all hotknife edges around the given X-plane.
"""
import logging
import argparse

from neuclease import configure_default_logging
from neuclease.util import Timer
from neuclease.focused.hotknife import HEMIBRAIN_TAB_BOUNDARIES, find_all_hotknife_edges_for_plane

logger = logging.getLogger(__name__)

__doc__ += f"""\

Hint: The Hemibrain hotknife plane X-coordinates are:

    {HEMIBRAIN_TAB_BOUNDARIES.tolist()[1:-1]}
"""

def main():
    configure_default_logging()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--x-center', '-x', type=int, required=True,
                        help="Which X-coordinate to center the analysis around. Should be a hot-knife boundary")
    parser.add_argument('--spacing-radius', '-r', type=int, default=8,
                        help="How far away from the X-center coordinate the sample planes should be.  Always specified in scale-0 pixels.")
    parser.add_argument('--tile-width', '-t', type=int, default=2048,
                        help="Width of the tiles to use internally when fetching/analyzing the X-planes around the center coordinate.  Specified in scale-0 pixels.")
    parser.add_argument('--min-overlap', '-m', type=int, default=100,
                        help="Required overlap for returned edges, always given in scale-0 coordinates")
    parser.add_argument('--scale', '-s', type=int, default=2,
                        help="At which scale to perform the analysis. (Regardless, results will always be given in scale-0 pixels.)")
    parser.add_argument('--output', '-o', type=str, help="Where to write the output (as CSV)")
    parser.add_argument('dvid_server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    args = parser.parse_args()

    instance_info = (args.dvid_server, args.uuid, args.labelmap_instance)
    tile_shape = 2*(args.tile_width,)

    if args.output is None:    
        args.output = f"hotknife-edges-x{args.x_center:05d}-r{args.spacing_radius}-m{args.min_overlap}-s{args.scale}.csv"
    
    edge_table = find_all_hotknife_edges_for_plane(*instance_info, args.x_center, tile_shape, args.spacing_radius, args.min_overlap, scale=args.scale)
    with Timer(f"Writing to {args.output}", logger):
        edge_table.to_csv(args.output, index=False, header=True)
        
    logger.info("DONE.")


if __name__ == "__main__":
    main()
