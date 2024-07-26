import json
from itertools import chain

import numpy as np
import pandas as pd


def segment_properties_json(
    df,
    label_col=None,
    *,
    description_col=None,
    string_cols=[],
    number_cols=[],
    tag_cols=[],
    prefix_tags=False,
    drop_empty=True,
    output_path=None,
):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md

    Args:
        df:
            DataFrame whose index contains segment IDs.
            To avoid confusion, we require that the Index be named 'body' or 'segment'.
            All columns will be converted to segment properties. If a column is not
            explicitly listed in the following arguments, its property type will be inferred
            from the column name and dtype.

        label_col:
            Which column to use for the 'label' property which is shown in neuroglancer by default.

        description_col:
            Which column to use as the 'description' property.

        string_cols:
            Columns to represent as 'string' properties.

        number_cols:
            Columns to represent as 'number' properties.

        tag_cols:
            Columns which should be used to generate the (combined) 'tags' property.
            If you want a column to be used for both tags and a different property,
            be sure to list it explicitly in both arguments.  For example:

                segment_properties_json(df, label_col='cell_class', tag_cols=['cell_class', ...])

        prefix_tags:
            If True, all tags will be prefixed with the name of the column they came from,
            e.g. 'status:Anchor'

        drop_empty:
            If any IDs in the input have no non-empty (null or "") properties, then drop them,
            from the output entirely so they don't show up in neuroglancer's default segment list.

        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict) which should be written into an 'info' file to
        host neuroglancer precomputed segment properties.
    """
    assert df.index.name in ('body', 'segment')
    if isinstance(df, pd.Series):
        df = df.to_frame()

    scalar_types = _scalar_property_types(
        df, label_col, description_col, string_cols, number_cols, tag_cols
    )

    if drop_empty:
        null_mask = df.isnull() | (df == '')
        df = df.loc[~null_mask.all(axis=1)]

    json_props = []
    for col, prop_type in scalar_types.items():
        j = _scalar_property_json(df[col], prop_type)
        json_props.append(j)

    if tag_cols:
        j = _tags_property_json(df, tag_cols, prefix_tags)
        json_props.append(j)

    info = {
        '@type': 'neuroglancer_segment_properties',
        'inline': {
            'ids': [*map(str, df.index)],
            'properties': json_props
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(info, f)

    return info


def _scalar_property_types(df, label_col, description_col, string_cols, number_cols, tag_cols):
    """
    Determine the full set of scalar (non-tag) properties that should be emitted
    along with their types, based on the users explicitly provided lists plus
    default types for the unlisted columns in df.
    """
    # Tag columns can also be explicitly listed among the scalar properties, but if they
    # weren't explicitly listed, we don't create default scalar properties for them.
    # We initialize the prop_types with tags first to make sure they don't receive
    # default types, but allow those keys to be overwritten by scalar types from the user.
    prop_types = {c: 'tags' for c in tag_cols}
    prop_types |= {c: 'string' for c in string_cols}
    prop_types |= {c: 'number' for c in number_cols}

    if label_col:
        prop_types[label_col] = 'label'
    if description_col:
        prop_types[description_col] = 'description'

    if unknown_cols := set(prop_types.keys()) - set(df.columns):
        raise RuntimeError(f"Columns not found: {unknown_cols}")

    # If there's only one column and it wasn't explicitly listed,
    # we assume its intended as the 'label' column.
    if not prop_types and len(df.columns) == 1:
        prop_types = {df.columns[0]: 'label'}

    # Infer the types of unlisted columns from either the name or dtype
    for name, dtype in df.dtypes.items():
        if name in prop_types:
            continue
        elif name == 'label':
            prop_types['label'] = 'label'
        elif name == 'description':
            prop_types['description'] = 'description'
        elif np.issubdtype(dtype, np.number):
            prop_types[name] = 'number'
        else:
            prop_types[name] = 'string'

    # drop tag properties
    return {k:v for k,v in prop_types.items() if v != 'tags'}


def _scalar_property_json(s, prop_type):
    """
    Constructs the JSON for a any segment property other than the 'tags' property.
    """
    if prop_type == 'number':
        assert s.dtype not in (np.int64, np.uint64), \
            "Neuroglancer doesn't support 64-bit integer properties.  Use int32 or float64"

        assert not s.isnull().any(), \
            (f"Column {s.name} contans NaN entries. "
             "I'm not sure what to do with NaN values in numeric properties.")

        return {
            'id': s.name,
            'type': 'number',
            'data_type': s.dtype.name,
            'values': s.tolist()
        }

    return {
        'id': s.name,
        'type': prop_type,
        'values': s.fillna("").astype(str).tolist()
    }


def _tags_property_json(df, tags_columns, prepend_colnames):
    """
    Constructs the JSON for the 'tags' segment property.
    """
    df = df[[*tags_columns]].copy()

    # Clean and convert each column to categorical
    # before we combine categories below.
    for c in df.columns:
        df[c] = (
            df[c]
            .astype('string')
            .str.replace(' ', '_')  # spaces are forbidden in tags
            .replace('', None)      # discard empty strings
            .astype('category')
        )
        if prepend_colnames:
            prefix = c.replace(' ', '_')
            prefixed_categories = [
                f'{prefix}:{cat}'
                for cat in df[c].dtype.categories
            ]
            df[c] = df[c].cat.rename_categories(prefixed_categories)

    # Convert to a single big categorical dtype
    all_tags = sorted({*chain(*(df[col].dtype.categories for col in df.columns))})
    df = df.astype(pd.CategoricalDtype(categories=all_tags))

    # Tags are written as a list-of-lists of sorted codes
    codes_df = pd.DataFrame({c: df[c].cat.codes for c in df.columns})
    sorted_codes = np.sort(codes_df.values, axis=1)
    codes_lists = [
        [x for x in row if x != -1]  # Drop nulls
        for row in sorted_codes.tolist()
    ]

    return {
        'id': 'tags',
        'type': 'tags',
        'tags': all_tags,
        'values': codes_lists,
    }
