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
    prefix_tags='all',
    drop_empty=True,
    output_path=None,
):
    """
    Construct segment properties JSON info file according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md

    Args:
        df:
            DataFrame whose index contains segment IDs.
            For clarity of intent, we require that the Index be named 'segment' or 'body'.

            All columns will be converted to segment properties. If a column is not
            explicitly listed in the following arguments, its property type will be inferred
            from the column name and dtype.

            The column dtypes can be string, category, or number (except int64).
            Boolean columns are only valid as tags.

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

            Columns with dtype bool can be used for tags, in which case the False items
            are discarded and the True items are tagged with the column name.

        prefix_tags:
            Either 'all' or 'disambiguate' or None.

            - If 'all', then all tags will be prefixed with the name of their source column,
              (e.g. 'status:Anchor'), other than boolean columns.

            - If 'disambiguate', then only tags which appear in multiple columns
              be prefixed with the name of their source column.

            - If None, then no disambiguation is performed.

        drop_empty:
            If any IDs in the input have no non-empty (null or "") properties,
            then drop them from the output entirely so they don't show up in
            neuroglancer's default segment list.

        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict) which should be written into an 'info' file to
        host neuroglancer precomputed segment properties.
    """
    assert df.index.name in ('body', 'segment')
    assert prefix_tags in ('all', 'disambiguate', None)

    if isinstance(string_cols, str):
        string_cols = [string_cols]
    if isinstance(number_cols, str):
        number_cols = [number_cols]
    if isinstance(tag_cols, str):
        tag_cols = [tag_cols]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if drop_empty:
        df = _drop_empty_rows(df)

    scalar_types = _scalar_property_types(
        df, label_col, description_col, string_cols, number_cols, tag_cols
    )

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
            'ids': [str(idx) for idx in df.index],
            'properties': json_props
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(info, f)

    return info


def _drop_empty_rows(df):
    if len(df) == 0 or len(df.columns) == 0:
        return df

    bool_cols = df.dtypes[df.dtypes == bool].index
    other_cols = df.dtypes[df.dtypes != bool].index

    valid_bool = valid_other = False
    if len(bool_cols) > 0:
        valid_bool = df[bool_cols].any(axis=1)
    if len(other_cols) > 0:
        valid_other = (df[other_cols].notnull() & (df[other_cols] != '')).any(axis=1)

    return df.loc[valid_bool | valid_other]


def _scalar_property_types(df, label_col, description_col, string_cols, number_cols, tag_cols):
    """
    Determine the full set of scalar (non-tag) properties that should be emitted
    along with their types, based on the user's explicitly provided lists plus
    default types for the unlisted columns in df.
    """
    # Tag columns can also be explicitly listed among the scalar properties, but if they
    # weren't explicitly listed, we don't create default scalar properties for them.
    # We include tags prop_types to indicate that they don't require default types, but
    # we add them *first* to allow those keys to be overwritten by scalar types that follow.
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
        if dtype == bool and name not in tag_cols:
            raise RuntimeError("Boolean columns are only valid as tag_cols")
        elif name in prop_types:
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
    Constructs the JSON for any segment property other than the 'tags' property.
    """
    if prop_type != 'number':
        return {
            'id': s.name,
            'type': prop_type,
            'values': s.fillna("").astype(str).tolist()
        }

    assert s.dtype not in (np.int64, np.uint64), \
        ("Neuroglancer doesn't support 64-bit integer properties. "
         "Use int32 or float64 instead.")

    assert not s.isnull().any(), \
        (f"Column {s.name} contans NaN entries. "
         "I'm not sure what to do with NaN values in numeric properties.")

    return {
        'id': s.name,
        'type': 'number',
        'data_type': s.dtype.name,
        'values': s.tolist()
    }


def _tags_property_json(prop_df, tags_columns, prefix_tags):
    """
    Constructs the JSON for the 'tags' segment property.
    """
    df = prop_df[[]].copy()

    # Clean and convert each column to categorical
    # individually before we combine categories below.
    for c in tags_columns:
        add_prefix = (prefix_tags == 'all' and prop_df[c].dtype != bool)
        df[c] = _convert_to_categorical(prop_df[c], add_prefix)

    if prefix_tags == 'disambiguate':
        df = _disambiguate_tags(df)

    # Convert to a single unified categorical dtype
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


def _convert_to_categorical(s, add_prefix):
    """
    Convert the given Series to a Categorical suitable for tags.
    """
    if s.dtype == 'category':
        s = s.cat.remove_unused_categories()

    if s.dtype == bool:
        s = s.astype('category', copy=False)
        s = s.cat.rename_categories({True: s.name})
        if False in s.dtype.categories:
            s = s.cat.remove_categories([False])

    s = s.astype('category', copy=False)

    # We interpret empty string as null
    if '' in s.dtype.categories:
        s = s.cat.remove_categories([''])

    # Spaces are forbidden in tags
    s = s.cat.rename_categories([
        str(cat).replace(' ', '_')
        for cat in s.dtype.categories
    ])

    if add_prefix:
        prefix = s.name.replace(' ', '_')
        s = s.cat.rename_categories([
            f'{prefix}:{cat}'
            for cat in s.dtype.categories
        ])

    return s


def _disambiguate_tags(df):
    """
    Given a dataframe in which all columns are Categoricals,
    find category values that are common across multiple columns
    and prepend such values with a prefix (the column name)
    to make sure no category value is duplicated from one column
    to the next.
    """
    # List the columns in which each tag appears:
    # {tag: [column, column, ...], ...}
    tag_cols = {}
    for col in df.columns:
        for tag in df[col].dtype.categories:
            cols = tag_cols.setdefault(tag, [])
            cols.append(col)

    # Build the renaming mapping for each column:
    # {
    #   column: {tag: new_tag, tag: new_tag, ...},
    #   column: {tag: new_tag, tag: new_tag, ...}, ...
    # }
    all_renames = {}
    for tag, cols in tag_cols.items():
        if len(cols) == 1:
            continue
        for col in cols:
            prefix = col.replace(' ', '_')
            renames = all_renames.setdefault(col, {})
            renames[tag] = f'{prefix}:{tag}'

    # Replace old names with new.
    for col, renames in all_renames.items():
        df[col] = df[col].cat.rename_categories(renames)

    return df
