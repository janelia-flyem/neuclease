"""
Downloads the results from the mito count proofreading protocol from
a DVID key-value instance and converts them into a DataFrame for
subsequent processing.

The DataFrame is exported in both CSV and pickle format.
(Hint: The pickle format is much easier to use.)

Note:
    No command-line interface.
    Edit the server, uuid, etc. below before running.
"""
import pickle
import pandas as pd
from tqdm import tqdm
from neuclease.dvid import fetch_keys, fetch_keyvalues

SERVER = 'https://hemibrain-dvid2.janelia.org'
UUIDS = ['a7e13', 'a510', 'c9e16']
RESULTS_INSTANCE = 'mito_count'

task_results = []
for uuid in tqdm(UUIDS):
    keys = fetch_keys(SERVER, uuid, RESULTS_INSTANCE)
    kvs = fetch_keyvalues(SERVER, uuid, RESULTS_INSTANCE, keys, as_json=True)
    for result_key, result in kvs.items():
        for k in result.keys():
            # Convert lists to tuples so they can be hashed
            # (more convenient for pandas manipulations)
            if isinstance(result[k], list):
                result[k] = tuple(result[k])

        result['retrieved_from'] = (SERVER, uuid, RESULTS_INSTANCE, result_key)

        # Extract the uuid the proofreader was using from the dvid source,
        # e.g. https://hemibrain-dvid2.janelia.org/#/repo/a7e1
        result['task_uuid'] = result['DVID source'].split('/')[-1]
        task_results.append(result)

mc_df = pd.DataFrame.from_dict(task_results)

# Reorder the columns to put the important stuff first.
cols = ['result', 'neighborhood id', 'neighborhood origin', 'task_uuid', 'user']
cols = cols + list({*mc_df.columns} - {*cols})
mc_df = mc_df[cols]

path = 'mito-count-results.pkl'
print(f"Writing {len(mc_df)} results to {path}")
with open(path, 'wb') as f:
    pickle.dump(mc_df, f)

path = 'mito-count-results.tab-delimited.csv'
print(f"Writing {len(mc_df)} results to {path}")
mc_df.to_csv(path, sep='\t', index=False, header=True)

print("DONE.")
