import json

def generate_typereview_assignment(df, output_path, uuid='', comment=''):
    """
    Generate a mergereview-like assignment for doing type comparison.
    
    Args:
        df:
            A pandas DataFrame with columns ["body_a", "body_b", "score"]

        output_path:
            Where to write the output JSON assignment

        uuid:
            Optional.  Indicates which UUID was used to generate these tasks.
        
        comment:
            Optional comment.
    """
    assert {*df.columns} >= {"body_a", "body_b", "score"}, \
        "DataFrame lacks the expected columns."
    
    tasks = []
    for row in df.itertuples():
        task = {
            # neu3 fields
            'task type': "type review",
            'task result id': f"{row.body_a}_{row.body_b}",
            
             "body ID A": row.body_a,
             "body ID B": row.body_b,
             'match_score': float(row.score),
             
            # Debugging fields
            'debug': {
                'original_uuid': uuid,
                'comment': comment
            }
        }
        tasks.append(task)

    assignment = {
        "file type":"Neu3 task list",
        "file version":1,
        "task list": tasks
    }

    with open(output_path, 'w') as f:
        json.dump(assignment, f, indent=2)
