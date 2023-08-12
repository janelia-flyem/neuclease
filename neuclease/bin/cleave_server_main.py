#!/usr/bin/env python3
import neuclease.cleave.cleave_server


# main function declared so this file can be used as console_script in setup.py
def main():
    neuclease.cleave.cleave_server.main(debug_mode=False)


if __name__ == "__main__":
    # CLEAVE_SERVER_MACHINE = 'bergs-lm4'
    # CLEAVE_PORT = '7080'
    # DVID_SERVER = 'emdata6.int.janelia.org'
    # DVID_PORT = '9000'
    # PRIMARY_UUID = '41d6ec06b9554b1383a236cacef8185f'

    # BIGQUERY_TABLE = 'janelia-flyem.cns_uploads.intrabody-edges-2023-07-30-41d6ec'

    # import os
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/janelia-flyem-0523020aea5e.json'

    # args = f"""\
    #     --bigquery-table {BIGQUERY_TABLE}
    #     --port {CLEAVE_PORT}
    #     --log-dir /tmp/logs/cns-full-{DVID_SERVER}-{DVID_PORT}-{CLEAVE_SERVER_MACHINE}-{CLEAVE_PORT}
    #     --primary-dvid-server {DVID_SERVER}:{DVID_PORT}
    #     --primary-uuid {PRIMARY_UUID}
    #     --primary-labelmap-instance segmentation
    #     --max-cached-bodies=0
    #     --disable-extra-edge-cache
    # """.split()

    # import sys
    # sys.argv += args

    main()
