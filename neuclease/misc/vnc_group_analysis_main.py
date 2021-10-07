"""
Compare the 'group' (instance) fields in DVID and Clio for all annotated bodies,
flag conflicts, and export non-conflicts so the two databases can be synchronized
in the non-conflicting cases.

Exports a directory of reports and CSV files.

After your results are obtained, you can also use this script to commit
the non-conflicting clio-to-dvid changes via the -c option.
In that case, all other options are ignored (except uuid).

Example:

    # Generate reports
    vnc_group_analysis -o group-reports

    # View the conflict report
    open group-reports/*.html

    # Commit non-conflicting group updates from Clio to DVID
    vnc_group_analysis -c group-reports/clio-to-dvid-updates.csv

    # Note: Elsewhere, you should push non-conflicting updates to from DVID to Clio
    ls group-reports/dvid-to-clio-updates.csv
"""
import logging
import argparse
logger = logging.getLogger(__name__)

VNC_PROD = "emdata5.janelia.org:8400"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dvid-server', '-d',
                        help=f'Optional.  By default, uses {VNC_PROD}')
    parser.add_argument('--uuid', '-u',
                        help='Optional.  By default, use the master branch HEAD.')
    parser.add_argument('--output-dir', '-o', default='clio-dvid-group-reports',
                        help='Directory to export results. Default is clio-dvid-group-reports/')
    parser.add_argument('--plot-format', '-f', default='png', choices=['png', 'hv'],
                        help='Format of the exported conflict set plots (png vs. holoviews)')
    parser.add_argument('--skip-plots', '-s', action='store_true',
                        help="If this flag is present, don't generate plots; just generate the CSV files. (Much faster.)")
    parser.add_argument('--commit-dvid-updates', '-c',
                        help="To commit the non-conflicting clio->dvid updates to DVID, pass the CSV file here in this argument.")
    args = parser.parse_args()
    return args


def main():
    # Fast path for --help
    parse_args()

    from neuclease import configure_default_logging
    configure_default_logging()

    logger.info("Importing libraries")

    from .vnc_group_analysis import main as vga_main
    return vga_main()


if __name__ == "__main__":
    main()
