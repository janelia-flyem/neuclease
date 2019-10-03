import json
import tempfile
import textwrap
from tempfile import TemporaryFile

import pytest
import numpy as np
from neuclease.util import (uuids_match, read_csv_header, read_csv_col, connected_components,
                            connected_components_nonconsecutive, graph_tool_available,
                            closest_approach, approximate_closest_approach, upsample, is_lexsorted, lexsort_columns,
                            lexsort_inplace, gen_json_objects, ndrange, compute_parallel)

def test_uuids_match():
    assert uuids_match('abcd', 'abcdef') == True
    assert uuids_match('abc9', 'abcdef') == False
    assert uuids_match('abcdef', 'abcd') == True
    assert uuids_match('abcdef', 'abc9') == False
    
    with pytest.raises(AssertionError):
        uuids_match('', 'abc')        

    with pytest.raises(AssertionError):
        uuids_match('abc', '')        

def test_read_csv_header():
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix='test_read_csv_header')
    tmpfile.write(textwrap.dedent("""\
        a,b,c
        0,1,2
        3,4,5
    """))
    tmpfile.flush()
    assert read_csv_header(tmpfile.name) == ['a', 'b', 'c']


def test_read_csv_header_singlecol():
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix='test_read_csv_header_singlecol')
    tmpfile.write(textwrap.dedent("""\
        a
        0
        3
    """))
    tmpfile.flush()
    assert read_csv_header(tmpfile.name) == ['a']


def test_read_csv_header_noheader():
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix='test_read_csv_header_noheader')
    tmpfile.write(textwrap.dedent("""\
        0,1,2
        3,4,5
    """))
    tmpfile.flush()
    assert read_csv_header(tmpfile.name) is None


def test_read_csv_header_noheader_singlecol():
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix='test_read_csv_header_noheader_singlecol')
    tmpfile.write(textwrap.dedent("""\
        0
        3
    """))
    tmpfile.flush()
    assert read_csv_header(tmpfile.name) is None


def test_read_csv_col():
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix='test_read_csv_col')
    tmpfile.write(textwrap.dedent("""\
        a,b,c
        0,1,2
        3,4,5
    """))
    tmpfile.flush()
    
    col0 = read_csv_col(tmpfile.name)
    assert (col0  == [0,3]).all()
    assert col0.name == 'a'

    col1 = read_csv_col(tmpfile.name, 1)
    assert (col1  == [1,4]).all()
    assert col1.name == 'b'

    col2 = read_csv_col(tmpfile.name, 2)
    assert (col2  == [2,5]).all()
    assert col2.name == 'c'


def test_read_csv_col_noheader():
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', prefix='test_read_csv_col')
    tmpfile.write(textwrap.dedent("""\
        0,1,2
        3,4,5
    """))
    tmpfile.flush()
    
    col0 = read_csv_col(tmpfile.name)
    assert (col0  == [0,3]).all()
    assert col0.name is None

    col1 = read_csv_col(tmpfile.name, 1)
    assert (col1  == [1,4]).all()
    assert col1.name is None

    col2 = read_csv_col(tmpfile.name, 2)
    assert (col2  == [2,5]).all()
    assert col2.name is None


@pytest.mark.skipif(not graph_tool_available(), reason="requires graph-tool")
def test_connected_components_gt():
    edges = [[1,2],
             [2,3],
             [4,5],
             [5,6]]

    cc_labels = connected_components(edges, 8, _lib='gt')
    assert cc_labels.shape == (8,)
    assert np.unique(cc_labels).shape == (4,)
    assert (cc_labels[0] != cc_labels[1])
    assert (cc_labels[1:4] == cc_labels[1]).all()
    assert (cc_labels[4:7] == cc_labels[4]).all()
    assert cc_labels[7] != cc_labels[6]


def test_connected_components_nx():
    edges = [[1,2],
             [2,3],
             [4,5],
             [5,6]]

    cc_labels = connected_components(edges, 8, _lib='nx')
    assert cc_labels.shape == (8,)
    assert np.unique(cc_labels).shape == (4,)
    assert (cc_labels[0] != cc_labels[1])
    assert (cc_labels[1:4] == cc_labels[1]).all()
    assert (cc_labels[4:7] == cc_labels[4]).all()
    assert cc_labels[7] != cc_labels[6]


def test_connected_components_nonconsecutive():
    edges = [[1,2],
             [2,3],
             [4,5],
             [5,6]]

    edges = 10 * np.array(edges, np.uint32)
    node_ids = 10 * np.array([0,1,2,3,4,5,6,7], np.uint32)

    cc_labels = connected_components_nonconsecutive(edges, node_ids)
    assert cc_labels.shape == (8,)
    assert np.unique(cc_labels).shape == (4,)
    assert (cc_labels[0] != cc_labels[1])
    assert (cc_labels[1:4] == cc_labels[1]).all()
    assert (cc_labels[4:7] == cc_labels[4]).all()
    assert cc_labels[7] != cc_labels[6]
    

def test_closest_approach():
    _ = 0
    
    img = [[1,1,2,2,2],
           [_,_,_,_,_],
           [3,_,_,4,_],
           [3,3,3,_,_],
           [_,_,_,_,_],]

    img = np.asarray(img, np.uint32)

    point_a, point_b, distance = closest_approach(img, 1, 2)
    assert point_a == (0,1)
    assert point_b == (0,2)
    assert distance == 1.0

    point_a, point_b, distance = closest_approach(img, 1, 3)
    assert point_a == (0,0)
    assert point_b == (2,0)
    assert distance == 2.0

    point_a, point_b, distance = closest_approach(img, 2, 4)
    assert point_a == (0,3)
    assert point_b == (2,3)
    assert distance == 2.0
    
    point_a, point_b, distance = closest_approach(img, 3, 4)
    assert point_a == (3,2)
    assert point_b == (2,3)
    assert np.allclose(np.sqrt(2.0), distance)

    # Bad inputs
    point_a, point_b, distance = closest_approach(img, 1, 1)
    assert distance == 0.0
    point_a, point_b, distance = closest_approach(img, 1, 99)
    assert distance == np.inf
    point_a, point_b, distance = closest_approach(img, 99, 1)
    assert distance == np.inf


def test_approximate_closest_approach():
    _ = 0
    
    img = [[[1,1,2,2,2],
            [_,_,_,_,_],
            [3,_,_,4,_],
            [3,3,3,_,_],
            [_,_,_,_,_]]]

    img = np.asarray(img, np.uint32)
    
    point_a, point_b, distance = approximate_closest_approach(img, 2, 4, scale=1)
    assert point_a == (0,0,2)
    assert point_b == (0,2,3)
    assert distance == np.sqrt(2**2 + 1**2)


def test_upsample():
    img = [[1,2],
           [3,4]]
    
    img = np.asarray(img, dtype=int)
    upsampled = upsample(img, 2)
    
    expected = [[1,1,2,2],
                [1,1,2,2],
                [3,3,4,4],
                [3,3,4,4]]

    assert (upsampled == expected).all()


def test_is_lexsorted():
    data = [[0,0,1],
            [1,0,1],
            [1,1,1],
            [1,2,1],
            [1,2,1],
            [2,2,1],
            [2,2,2]]
    
    data = np.asarray(data)
    
    assert is_lexsorted(data)
    assert not is_lexsorted(data[::-1])


def test_lexsort_columns():
    data = [[0,0,1],
            [1,0,1],
            [1,1,1],
            [1,2,1],
            [1,2,1],
            [2,2,1],
            [2,2,2]]

    data = np.asarray(data)
    np.random.shuffle(data)
    #assert not is_lexsorted(data)
    
    sorted_data = lexsort_columns(data)
    assert is_lexsorted(sorted_data)


def test_lexsort_inplace():
    data = [[0,0,1],
            [1,0,1],
            [1,1,1],
            [1,2,1],
            [1,2,1],
            [2,2,1],
            [2,2,2]]

    data = np.asarray(data)
    np.random.shuffle(data)
    #assert not is_lexsorted(data)
    
    lexsort_inplace(data)
    assert is_lexsorted(data)


def test_gen_json_objects():
    strings = [b'[]',
               b'[{}]',
               b'[{},{}]',
               b'[{}, {"a": {}}]',
               b' [{"a": 123, "b": 456}, {"x": null}] ']

    try:
        for s in strings:
            with TemporaryFile() as f:
                f.write(s)
                assert list(gen_json_objects(f)) == json.loads(s)
    except Exception as ex:
        raise AssertionError(f"Failed to properly parse this string: '{s}'") from ex

    # Try some actual synapse data
    s = (' [ {"Rels": [{"To": [32377, 12263, 24429], "Rel": "PostSynTo"}], '
            '"Kind": "PostSyn", "Prop": {"user": "$fpl", "conf": "0.972182"}, '
            '"Pos": [32382, 12243, 24439], "Tags": []} ] ')

    s = s.encode()

    with TemporaryFile() as f:
        f.write(s)
        it = gen_json_objects(f)
        o = next(it)
        assert o == json.loads(s)[0]

    # Repeat 3 times
    s2 = '[' + ',\n'.join(3*[json.dumps(o)]) + ']'
    s2 = s2.encode()
    with TemporaryFile() as f:
        f.write(s2)
        it = gen_json_objects(f)
        assert list(it) == json.loads(s2)


def test_ndrange():
    r = ndrange((1,2,3), (3,5,6), (1,2,2))
    expected = [(1, 2, 3),
                (1, 2, 5),
                (1, 4, 3),
                (1, 4, 5),
                (2, 2, 3),
                (2, 2, 5),
                (2, 4, 3),
                (2, 4, 5)]

    assert len(r) == len(expected)
    assert list(r) == expected


def _double(x):
    return 2*x

def _add(x, y):
    return x + y

def test_compute_parallel():
    items = list(range(100))
    results = compute_parallel(_double, items, threads=2)
    assert results == list(range(0,200,2))

    items = list(range(100))
    results = compute_parallel(_double, items, processes=2)
    assert results == list(range(0,200,2))

    items = [*zip(range(10), range(100,110))]
    results = compute_parallel(_add, items, processes=2, starmap=True)
    assert results == [sum(item) for item in items]


if __name__ == "__main__":
    pytest.main(['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_util'])
