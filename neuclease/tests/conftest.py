import os
import sys
import time
import signal
import subprocess

import requests
import pytest
import numpy as np
import pandas as pd

import neuclease

from libdvid import DVIDNodeService

TEST_DVID_SERVER_PROC = None # Initialized below

TEST_DATA_DIR = os.path.dirname(neuclease.__file__) + '/../.test-data'
DVID_STORE_PATH = f'{TEST_DATA_DIR}/dvid-datastore'
DVID_CONFIG_PATH = f'{TEST_DATA_DIR}/dvid-config.toml'

DVID_PORT = 8000
TEST_SERVER = f"127.0.0.1:{DVID_PORT}"
TEST_REPO = None # Initialized below
TEST_REPO_ALIAS = 'neuclease-test'

#DVID_SHUTDOWN_TIMEOUT = 10.0
DVID_SHUTDOWN_TIMEOUT = 2.0

DVID_CONFIG = f"""\
[server]
httpAddress = ":{DVID_PORT}"
rpcAddress = ":{DVID_PORT+1}"
webClient = "{sys.prefix}/http/dvid-web-console"

[logging]
logfile = "{TEST_DATA_DIR}/dvid.log"
max_log_size = 500 # MB
max_log_age = 30   # days

[store]
    [store.mutable]
    engine = "basholeveldb"
    path = "{DVID_STORE_PATH}"
"""

@pytest.fixture(scope="session")
def labelmap_setup():
    global TEST_DVID_SERVER_PROC
    TEST_DVID_SERVER_PROC = launch_dvid_server()
    try:
        # Can't reuse previous repos because we lock the repo node
        # (and somehow -fullwrite doesn't work?)
        init_test_repo(reuse_existing=False)
        merge_table_path, mapping_path, supervoxel_vol = init_labelmap_nodes()
        yield TEST_SERVER, TEST_REPO, merge_table_path, mapping_path, supervoxel_vol
    finally:
        print("\nTerminating DVID test server...")
        TEST_DVID_SERVER_PROC.send_signal(signal.SIGTERM)
        try:
            TEST_DVID_SERVER_PROC.wait(DVID_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            print("DVID test server did not shut down cleanly.  Killing...")
            TEST_DVID_SERVER_PROC.send_signal(signal.SIGKILL)
        print("DVID test server is terminated.")


def launch_dvid_server():
    os.makedirs(DVID_STORE_PATH, exist_ok=True)
    with open(DVID_CONFIG_PATH, 'w') as f:
        f.write(DVID_CONFIG)

    dvid_proc = subprocess.Popen(f'dvid -verbose -fullwrite serve {DVID_CONFIG_PATH}', shell=True)
    time.sleep(1.0)
    if dvid_proc.poll() is not None:
        raise RuntimeError(f"dvid couldn't be launched.  Exited with code: {dvid_proc.returncode}")
    return dvid_proc

def init_test_repo(reuse_existing=True):
    global TEST_REPO

    if reuse_existing:
        r = requests.get(f'http://{TEST_SERVER}/api/repos/info')
        r.raise_for_status()
        repos_info = r.json()
        for repo_uuid, repo_info in repos_info.items():
            if repo_info["Alias"] == TEST_REPO_ALIAS:
                TEST_REPO = repo_uuid
                return repo_uuid

    # Create a new repo
    r = requests.post(f'http://{TEST_SERVER}/api/repos',
                      json={'alias': TEST_REPO_ALIAS, 'description': 'Test repo for neuclease integration tests'})
    r.raise_for_status()
    repo_uuid = r.json()['root']
    TEST_REPO = repo_uuid
    return repo_uuid

def create_test_labelmap_instance(uuid, instance_name):
    params = { "typename": "labelmap",
               "dataname": instance_name,
               "BlockSize": "64,64,64",
               "IndexedLabels": "true",
               "CountLabels": "true",
               "MaxDownresLevel": '2' }
    
    r = requests.post(f'http://{TEST_SERVER}/api/repo/{uuid}/instance', json=params)

    if r.status_code == 400 and 'already exists' in r.content.decode():
        return
    else:
        r.raise_for_status()


def init_labelmap_nodes():
    # Five supervoxels are each 1x3x3, arranged in a single row like this:
    # [[[1 1 1 2 2 2 3 3 3 4 4 4 5 5 5]
    #   [1 1 1 2 2 2 3 3 3 4 4 4 5 5 5]
    #   [1 1 1 2 2 2 3 3 3 4 4 4 5 5 5]]]
    supervoxel_vol = np.zeros((1,3,15), np.uint64)
    supervoxel_vol[:] = (np.arange(15, dtype=np.uint64) // 3).reshape(1,1,15)
    supervoxel_vol += 1
    np.set_printoptions(linewidth=100)
    #print(supervoxel_vol)

    # Merge table: Merge them all together
    id_a = np.array([1, 2, 3, 4], np.uint64)
    id_b = np.array([2, 3, 4, 5], np.uint64)

    xa = np.array([2, 5, 8, 11], np.uint32)
    ya = np.array([1, 1, 1, 1], np.uint32)
    za = np.array([0, 0, 0, 0], np.uint32)

    xb = np.array([3, 6, 9, 12], np.uint32)
    yb = np.array([1, 1, 1, 1], np.uint32)
    zb = np.array([0, 0, 0, 0], np.uint32)

    # Weak edge between 3 and 4
    score = np.array([0.4, 0.4, 0.8, 0.4], np.float32)

    merge_table = pd.DataFrame({'id_a': id_a, 'id_b': id_b,
                                'xa': xa, 'ya': ya, 'za': za,
                                'xb': xb, 'yb': yb, 'zb': zb,
                                'score': score})
    merge_table = merge_table[['id_a', 'id_b', 'xa', 'ya', 'za', 'xb', 'yb', 'zb', 'score']]

    merge_table_path = f'{TEST_DATA_DIR}/merge-table.npy'
    np.save(merge_table_path, merge_table.to_records(index=False))
    
    create_test_labelmap_instance(TEST_REPO, 'segmentation')

    # Expand to 64**3
    supervoxel_block = np.zeros((64,64,64), np.uint64)
    supervoxel_block[:1,:3,:15] = supervoxel_vol
    DVIDNodeService(TEST_SERVER, TEST_REPO).put_labels3D('segmentation', supervoxel_block, (0,0,0))

    r = requests.post(f'http://{TEST_SERVER}/api/node/{TEST_REPO}/commit', json={'note': 'supervoxels'})
    r.raise_for_status()

#     # Create a child node for agglo mappings    
#     r = requests.post(f'http://{TEST_SERVER}/api/node/{TEST_REPO}/newversion', json={'note': 'agglo'})
#     r.raise_for_status()
#     agglo_uuid = r.json["child"]

    # Merge everything
    agglo_uuid = TEST_REPO
    r = requests.post(f'http://{TEST_SERVER}/api/node/{agglo_uuid}/segmentation/merge', json=[1, 2, 3, 4, 5])
    r.raise_for_status()

    mapping = np.array([[1,1],[2,1],[3,1],[4,1],[5,1]], np.uint64)
    #mapping = pd.DataFrame(mapping, columns=['sv', 'body']).set_index('sv')['body']
    
    mapping_path = f'{TEST_DATA_DIR}/mapping.npy'
    np.save(mapping_path, mapping)

    return merge_table_path, mapping_path, supervoxel_vol


