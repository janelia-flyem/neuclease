import os
import threading
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd

from pymongo import MongoClient

from .util import tqdm_proxy, compute_parallel
from .dvid import fetch_supervoxels

##
## Notes:
##
## The entire set of original hemibrain "iteration-1" adjacencies was loaded into mongo.
## The database can be found here:
## /groups/flyem/data/scratchspace/hemibrain/mongo-edges-indexed/
## There's a README in there that explains how to launch a mongo server.
## Typically, you should copy it to a fast local drive such as /scratch/<USER/ before using it. 
##
## Also note that this is NOT currently the way that edges are stored or used by neuclease's "cleave server".
## This is just a stand-alone file, where it's convenient for me to define these functions.
##

MONGO_CLIENTS = {}
def mongo_client(server):
    """
    Return a MongoClient for the given server.
    The object is cached, so this function will return the same service
    object if called again from the same thread with the same arguments.
    """
    # One per thread/process
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    try:
        mgclient = MONGO_CLIENTS[(pid, thread_id, server)]
    except KeyError:
        mgclient = MongoClient(server)
        MONGO_CLIENTS[(pid, thread_id, server)] = mgclient
    return mgclient


def get_edge_count(mongo_server='127.0.0.1'):
    # The server has one database named 'edges',
    # and that database has one "collection" (table), also named 'edges'.
    return mongo_client(mongo_server)['edges']['edges'].estimated_document_count()


def fetch_all_edges_from_body(mongo_server, dvid_server, uuid, instance, body,
                              min_score=None, is_singleton=False, format='pandas'): # @ReservedAssignment
    """
    Args:
        mongo_server:
            IP address of the mongo server

        dvid_server:
            dvid address, e.g. emdata4:8000
        
        uuid:
            dvid uuid
        
        instance:
            dvid segmentation labelmap instance
        
        body:
            body ID
        
        min_score:
            If provided, edges below this score will be discarded.
            (Typically in our edge tables, higher scores are better.)
        
        is_singleton:
            Optimization. If you happen to know in advance that the given
            body consists of only a single supervoxel, and the supervoxel ID
            matches the body ID, then set this flag to ``True``.
            In that case, we can skip a call to ``fetch_supervoxels``,
            saving time.
        
        format:
            If no edges were found or an error is encountered, returns a tuple (body, message).
            Otherwise:
                - If 'json', then the edges are returned as json data (which
                  is how they were stored in mongo), but loaded into Python lists/dicts.
                - If 'pandas', the edges are converted into a DataFrame (dict keys become column names).

    Returns:
        Retrieved edges, as either a list-of-dicts, or a DataFrame.
        
    """
    assert format in ('pandas', 'json')
    mgclient = mongo_client(mongo_server)

    body = int(body)
    if is_singleton:
        svs = [body]
    else:
        try:
            svs = fetch_supervoxels(dvid_server, uuid, instance, body).tolist()
        except Exception:
            return (body, "Body does not exist in DVID")
    
    if len(svs) == 0:
        return (body, "Body has no supervoxels")
    
    # All edges where one sv is IN the body, and the other is NOT IN the body.
    q1 = {"$and": [{"sv_a": {"$in": svs}}, {"sv_b": {"$nin": svs}}]}
    q2 = {"$and": [{"sv_b": {"$in": svs}}, {"sv_a": {"$nin": svs}}]}
    q = {"$or": [q1,q2]}
    r = mgclient['edges']['edges'].find(q)
    
    try:
        edges = list(r)
    except Exception as ex:
        return (body, str(ex))
    
    if len(edges) == 0:
        return (body, "Body has no edges in the database")

    if min_score is not None:
        edges = [*filter(lambda e: e['score'] >= min_score, edges)]
    
    if len(edges) == 0:
        return (body, "None of the body's edges exceed the min_score")

    for e in edges:
        del e['_id']
        e['body'] = body
    
    if format == 'json':
        return edges
    else:
        # fix dtypes
        df = pd.DataFrame(edges)
        df['sv_a'] = df['sv_a'].astype(np.uint64)
        df['sv_b'] = df['sv_b'].astype(np.uint64)
        df['score'] = df['score'].astype(np.float32)
        df['resolution'] = df['resolution'].astype(np.int8)
        for col in ['za', 'ya', 'xa', 'zb', 'yb', 'xb']:
            df[col] = df[col].astype(np.int32)
        return df


def fetch_all_edges_for_bodies(mongo_server, dvid_server, uuid, instance, bodies, min_score=0.1,
                               all_singletons=False, batch_size=100_000, processes=32):
    """
    Fetch in batches, converting to DataFrame only after each batch.
    """
    batch_error_dfs = []
    
    batch_dfs = []
    for batch_start in tqdm_proxy(range(0, len(bodies), batch_size)):
        batch_bodies = bodies[batch_start:batch_start+batch_size]

        _query_fn = partial(fetch_all_edges_from_body, mongo_server, dvid_server, uuid, instance, # body,
                            min_score=min_score, is_singleton=all_singletons, format='json')

        edge_lists = compute_parallel(_query_fn, batch_bodies, 1000, ordered=False, processes=processes)

        batch_errors = [*filter(lambda e: e is not None and isinstance(e, tuple), edge_lists)]
        if batch_errors:
            batch_error_df = pd.DataFrame(batch_errors, columns=['body', 'error'])
            batch_error_df['error'] = batch_error_df['error'].astype('category')
            batch_error_dfs.append( batch_error_df )
        
        edge_lists = filter(lambda e: (e is not None) and (len(e) > 0) and not isinstance(e, tuple), edge_lists)
        edges = list(chain(*edge_lists))

        # fix dtypes
        df = pd.DataFrame(edges)
        df['sv_a'] = df['sv_a'].astype(np.uint64)
        df['sv_b'] = df['sv_b'].astype(np.uint64)
        df['score'] = df['score'].astype(np.float32)
        df['resolution'] = df['resolution'].astype(np.int8)
        for col in ['za', 'ya', 'xa', 'zb', 'yb', 'xb']:
            df[col] = df[col].astype(np.int32)

        batch_dfs.append(df)

    final_df = pd.concat(batch_dfs, ignore_index=True)
    
    if batch_error_dfs:
        final_error_df = pd.concat(batch_error_dfs, ignore_index=True)
    else:
        final_error_df = None

    return final_df, final_error_df
