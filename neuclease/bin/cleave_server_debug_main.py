#!/usr/bin/env python3
import sys
import neuclease.cleave.cleave_server

def main():
    debug_mode = False  # flask debug mode
    stdout_logging = True
    ## DEBUG
    # if len(sys.argv) == 1:
    #     _debug_mode = True
    #     import os
    #     log_dir = os.path.dirname(neuclease.__file__) + '/../logs'
    #     sys.argv += [#"--merge-table", "/magnetic/workspace/neuclease/tiny-merge-table.npy",
    #                  #"--mapping-file", "/magnetic/workspace/neuclease/tiny-mapping.npy",
    #                  #"--primary-dvid-server", "emdata3:8900",
    #                  #"--primary-uuid", "017a",
    #                  #"--primary-labelmap-instance", "segmentation",
    #                  #"--suspend-before-launch",

    #                  "--merge-table", "/tmp/merge-table-5812998448.csv",
    #                  "--primary-dvid-server", "emdata1:8900",
    #                  "--primary-uuid", "642cfed9e8704d0b83ccca2ee3688528",
    #                  "--primary-labelmap-instance", "segmentation",
    #                  "--log-dir", log_dir]

#     cmd_args = """\
#         -p 5555 \
#         --log-dir /tmp//debug-cleave-logs
#         --primary-dvid-server http://hemibrain-dvid2.janelia.org:8000 \
#         --primary-uuid 2b4ff297131d4dbd8ff2433a2ca0a113 \
#         --primary-labelmap-instance segmentation \
#         --primary-kafka-log /tmp/empty.jsonl \
#         --skip-focused-merge-update \
#         --skip-split-sv-update \
# """.split()
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/miniforge/envs/flyem-312/etc/conda/activate.d/janelia-flyem-0523020aea5e.json'
    cmd_args = "--bigquery-table janelia-flyem.yakuba_vnc_seg_2311117_32fb16fb_fill8_v2_grid1k.top200k-bodies-intrabody-edges-2024-10-01-a6f8e7f --port 7500 --log-dir /Users/bergs/workspace/neuclease/yakuba-test/logs --primary-dvid-server emdata7.int.janelia.org:8700 --primary-uuid a6f8e7f422f94cbaa71ce8ef58350c43 --primary-labelmap-instance segmentation --max-cached-bodies=100000"
    sys.argv += cmd_args.split()

    neuclease.cleave.cleave_server.main(debug_mode, stdout_logging)

## Example requests:
# """
# {"body-id": 673509195, "mesh-instance": "segmentation_meshes_tars", "port": 8900, "request-timestamp": "2018-05-10 13:40:56.117063", "seeds": {"1": [675222237], "2": [1266560684], "3": [1142805921], "5": [1329312351], "6": [1328298063], "7": [1264523335], "8": [1233488801, 1358310013], "9": [1357286646]}, "segmentation-instance": "segmentation", "server": "emdata3.int.janelia.org", "user": "bergs", "uuid": "017a"}
# {"body-id": 5812980088, "mesh-instance": "segmentation_meshes_tars", "port": 8900, "request-timestamp": "2018-05-10 13:48:32.071343", "seeds": {"1": [299622182, 769164613], "2": [727964335], "3": [1290606913], "4": [485167093], "5": [769514136]}, "segmentation-instance": "segmentation", "server": "emdata3.int.janelia.org", "user": "bergs", "uuid": "017a"}
# {"body-id": 5812980124, "mesh-instance": "segmentation_meshes_tars", "port": 8900, "request-timestamp": "2018-05-10 13:51:46.112896", "seeds": {"1": [391090531], "2": [453151532, 515221115, 515221301, 515557950, 515562175, 515562381, 515562454, 546597327, 577632049, 608330428, 608667239, 639701979, 639702027, 639702182, 670736831, 670736971, 670737150, 670737574]}, "segmentation-instance": "segmentation", "server": "emdata3.int.janelia.org", "user": "bergs", "uuid": "017a"}
# {"body-id": 5812980898, "mesh-instance": "segmentation_meshes_tars", "port": 8900, "request-timestamp": "2018-05-10 13:54:00.042885", "seeds": {"1": [449551305], "2": [1261194539], "3": [1229822848], "4": [883458155, 883458603], "5": [790693775]}, "segmentation-instance": "segmentation", "server": "emdata3.int.janelia.org", "user": "bergs", "uuid": "017a"}
# """

if __name__ == "__main__":
    main()
