import os
import sys
import time
import signal
import subprocess

import pytest
import requests

import neuclease

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##

@pytest.fixture(scope="module")
def cleave_server_setup(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    dvid_server, dvid_port = dvid_server.split(':')
    dvid_port = int(dvid_port)
    cleave_server_port = '5555'

    launch_script = os.path.dirname(neuclease.__file__) + '/bin/cleave_server_main.py'
    server_proc = subprocess.Popen(["python", launch_script,
                                    "--port", cleave_server_port,
                                    "--merge-table", merge_table_path,
                                    "--primary-dvid-server", dvid_server,
                                    "--primary-labelmap-instance", "segmentation"])

    # Give the server time to initialize
    time.sleep(2.0)
    assert server_proc.poll() is None, "cleave server process couldn't be started"

    yield dvid_server, dvid_port, dvid_repo, cleave_server_port
        
    server_proc.send_signal(signal.SIGTERM)
    server_proc.wait(2.0)

def test_simple_request(cleave_server_setup):
    """
    Make a trivial request for a cleave.
    """
    dvid_server, dvid_port, dvid_repo, port = cleave_server_setup
    
    data = { "user": "bergs",
             "body-id": 1,
             "port": dvid_port,
             "seeds": {"1": [1], "2": [5]},
             "server": dvid_server,
             "uuid": dvid_repo,
             "segmentation-instance": "segmentation",
             "mesh-instance": "segmentation_meshes_tars" }

    try:
        r = requests.post(f'http://localhost:{port}/compute-cleave', json=data)
        r.raise_for_status()
    except requests.RequestException:
        sys.stderr.write(r.content.decode() + '\n')
        raise

    # There's a weak edge between 3 and 4. See conftest.init_labelmap_nodes().
    assignments = r.json()["assignments"]
    assert assignments["1"] == [1,2,3]
    assert assignments["2"] == [4,5]
            

def test_fetch_log(cleave_server_setup):
    """
    Trival test to make sure the default page returns the log file.
    """
    _dvid_server, _dvid_port, _dvid_repo, port = cleave_server_setup

    try:
        r = requests.get(f'http://localhost:{port}')
        r.raise_for_status()
    except requests.RequestException:
        sys.stderr.write(r.content.decode() + '\n')
        raise

    assert 'INFO' in r.content.decode()

if __name__ == "__main__":
    pytest.main(['--pyargs', 'neuclease.tests.test_server'])
