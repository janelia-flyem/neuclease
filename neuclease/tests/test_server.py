import os
import sys
import time
import signal
import logging
import functools
import subprocess

import pytest
import requests

import neuclease

from neuclease.tests.conftest import TEST_DATA_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

##
## These tests rely on the global setupfunction 'labelmap_setup',
## defined in conftest.py and used here via pytest magic
##

def show_request_exceptions(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.RequestException as ex:
            r = ex.response
            sys.stderr.write(r.content.decode() + '\n')
            sys.stderr.write(f"See server log in {TEST_DATA_DIR}\n")
            raise
    return wrapper

@pytest.fixture(scope="module")
def cleave_server_setup(labelmap_setup):
    dvid_server, dvid_repo, merge_table_path, _mapping_path, _supervoxel_vol = labelmap_setup
    dvid_server, dvid_port = dvid_server.split(':')
    dvid_port = int(dvid_port)
    cleave_server_port = '5555'

    launch_script = os.path.dirname(neuclease.__file__) + '/bin/cleave_server_main.py'
    server_proc = subprocess.Popen(["python", launch_script,
                                    "--port", cleave_server_port,
                                    "--debug-export-dir", os.path.dirname(merge_table_path),
                                    "--merge-table", merge_table_path,
                                    "--primary-dvid-server", f"{dvid_server}:{dvid_port}",
                                    "--primary-uuid", dvid_repo,
                                    "--primary-labelmap-instance", "segmentation",
                                    "--testing",
                                    f"--log-dir={TEST_DATA_DIR}"])

    # Give the server time to initialize
    max_tries = 10
    while max_tries > 0:
        try:
            time.sleep(1.0)
            requests.get(f'http://127.0.0.1:{cleave_server_port}')
            break
        except requests.ConnectionError:
            logger.info("Cleave server is not started yet.  Waiting...")
            max_tries -= 1

    if server_proc.poll() is not None:
        msg = "Cleave server process couldn't be started.\n"
        logfile = os.path.splitext(merge_table_path)[0] + '.log'
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                log_contents = f.read()
            log_tail = '\n'.join(log_contents.split('\n')[-20:])
            msg += "Log tail:\n" + log_tail
        raise RuntimeError(msg)

    yield dvid_server, dvid_port, dvid_repo, cleave_server_port
        
    server_proc.send_signal(signal.SIGTERM)
    server_proc.wait(2.0)


@show_request_exceptions
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

    r = requests.post(f'http://127.0.0.1:{port}/compute-cleave', json=data)
    r.raise_for_status()

    # There's a weak edge between 3 and 4. See conftest.init_labelmap_nodes().
    assignments = r.json()["assignments"]
    assert assignments["1"] == [1,2,3]
    assert assignments["2"] == [4,5]
            

@show_request_exceptions
def test_fetch_log(cleave_server_setup):
    """
    Trival test to make sure the default page returns the log file.
    """
    _dvid_server, _dvid_port, _dvid_repo, port = cleave_server_setup

    r = requests.get(f'http://127.0.0.1:{port}')
    r.raise_for_status()

    assert 'INFO' in r.content.decode()


@show_request_exceptions
def test_set_primary_uuid(cleave_server_setup):
    dvid_server, dvid_port, dvid_repo, port = cleave_server_setup
    r = requests.post(f'http://127.0.0.1:{port}/primary-uuid', json={"uuid": "abc123"})
    r.raise_for_status()
    
    r = requests.get(f'http://127.0.0.1:{port}/primary-uuid')
    r.raise_for_status()
    assert r.json()["uuid"] == "abc123"
    

@show_request_exceptions
def test_body_edge_table(cleave_server_setup):
    dvid_server, dvid_port, dvid_repo, port = cleave_server_setup

    data = { "body-id": 1,
             "port": dvid_port,
             "server": dvid_server,
             "uuid": dvid_repo,
             "segmentation-instance": "segmentation" }

    r = requests.post(f'http://127.0.0.1:{port}/body-edge-table', json=data)
    r.raise_for_status()

    lines = r.content.decode().rstrip().split('\n')
    assert lines[0] == 'id_a,id_b,xa,ya,za,xb,yb,zb,score,body'
    assert len(lines) == 5, '\n' +  '\n'.join(lines)

@show_request_exceptions
def test_change_default_method(cleave_server_setup):
    """
    Change the default cleave method via the /set-default-params endpoint.
    Uses the test cleave 'algorithm' (echo-seeds) and sends a cleave request to verify that it worked.
    Changes the default method BACK after the test is run, so other unit tests won't break.
    """
    dvid_server, dvid_port, dvid_repo, port = cleave_server_setup
    
    r = requests.post(f'http://127.0.0.1:{port}/set-default-params?method=echo-seeds')
    r.raise_for_status()
    assert r.json() == { "method": "echo-seeds" }
    
    try:
        data = { "user": "bergs",
                 "body-id": 1,
                 "port": dvid_port,
                 "seeds": {"1": [1], "2": [5]},
                 "server": dvid_server,
                 "uuid": dvid_repo,
                 "segmentation-instance": "segmentation",
                 "mesh-instance": "segmentation_meshes_tars" }
    
        r = requests.post(f'http://127.0.0.1:{port}/compute-cleave', json=data)
        r.raise_for_status()

        # Since we switched to the 'echo-seeds' method,
        # the assignments merely match the seeds.
        assignments = r.json()["assignments"]
        assert assignments["1"] == [1]
        assert assignments["2"] == [5]

    finally:
        # Change the default method BACK after the test is run, so other unit tests won't break.
        r = requests.post(f'http://127.0.0.1:{port}/set-default-params?method=seeded-watershed')
        r.raise_for_status()
        assert r.json() == { "method": "seeded-watershed" }


@show_request_exceptions
def test_debug_endpoint(cleave_server_setup):
    dvid_server, dvid_port, dvid_repo, port = cleave_server_setup
    
    data = { "user": "bergs",
             "body-id": 1,
             "port": dvid_port,
             "seeds": {"1": [1], "2": [5]},
             "server": dvid_server,
             "uuid": dvid_repo,
             "segmentation-instance": "segmentation",
             "mesh-instance": "segmentation_meshes_tars",
             "comment": "This is a comment." }

    r = requests.post(f'http://127.0.0.1:{port}/debug', json=data)
    r.raise_for_status()
    
    # Echoes the request back to you (and logs it)
    assert r.json() == data


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_server'])
