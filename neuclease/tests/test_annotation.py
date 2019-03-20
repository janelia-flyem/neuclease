import json
import tempfile
import pytest

from neuclease.dvid import load_gary_synapse_json

@pytest.fixture
def sample_gary_synapse_json():
    test_json = {
        "data": [
            {"T-bar":{"confidence":"0.928","location":[64,8490,21973]},
             "partners":[{"location":[86,8499,21987],"confidence":"0.462"},
                         {"location":[82,8481,21975],"confidence":"0.443"},
                         {"location":[70,8489,21987],"confidence":"0.991"}]},
            
            {"T-bar":{"confidence":"0.725","location":[64,8695,20578]},
             "partners":[{"location":[64,8702,20587],"confidence":"0.994"},
                         {"location":[82,8698,20567],"confidence":"0.919"},
                         {"location":[82,8687,20587],"confidence":"0.739"}]},
            
            {"T-bar":{"confidence":"0.896","location":[64,8828,26152]},
             "partners":[{"location":[85,8818,26155],"confidence":"0.885"},
                         {"location":[56,8813,26155],"confidence":"0.999"},
                         {"location":[70,8818,26163],"confidence":"0.952"}]},
            
            {"T-bar":{"confidence":"0.651","location":[64,8927,21152]},
              "partners":[{"location":[68,8935,21135],"confidence":"0.994"},
                         {"location":[81,8930,21151],"confidence":"0.951"},
                         {"location":[64,8953,21162],"confidence":"0.655"},
                         {"location":[77,8941,21162],"confidence":"0.976"}]},
            
            {"T-bar":{"confidence":"0.831","location":[64,9099,24241]},
             "partners":[{"location":[61,9104,24265],"confidence":"0.829"},
                         {"location":[49,9089,24247],"confidence":"0.541"},
                         {"location":[83,9076,24240],"confidence":"0.658"},
                         {"location":[64,9089,24260],"confidence":"0.976"},
                         {"location":[74,9111,24260],"confidence":"0.499"}]}
        ]}
    return test_json


def test_load_gary_synapse_json(sample_gary_synapse_json):
    path = tempfile.mktemp('.json')
    with open(path, 'w') as f:
        json.dump(sample_gary_synapse_json, f)
    
    point_df, partner_df = load_gary_synapse_json(path, processes=2, batch_size=2)

    tbar_locations = [syn["T-bar"]["location"] for syn in sample_gary_synapse_json["data"]]
    assert (point_df.query('kind == "tbar"')[['x', 'y', 'z']].values == tbar_locations).all().all() 
    assert len(point_df.query('kind == "tbar"')) == 5
    assert len(point_df.query('kind == "psd"')) == 18

    assert len(partner_df) == 18
    assert set(point_df.index) == set(partner_df.values.reshape(-1))
    assert len(set(partner_df['tbar_id'])) == 5
    assert len(set(partner_df['psd_id'])) == 18

if __name__ == "__main__":
    #from neuclease import configure_default_logging
    #configure_default_logging()
    args = ['-s', '--tb=native', '--pyargs', 'neuclease.tests.test_annotation']
    pytest.main(args)
