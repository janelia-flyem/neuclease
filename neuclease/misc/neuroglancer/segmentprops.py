"""
Utility for converting a DataFrame to neuroglancer segment properties (a JSON object).
"""
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
    tag_prefix_mode='all',
    tag_descriptions={},
    col_descriptions={},
    drop_empty=True,
    output_path=None,
):
    """
    Given a DataFrame, construct the JSON representation for neuroglancer segment properties
    (i.e. segment properties 'info' file) according to the neuroglancer spec:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/segment_properties.md

    Args:
        df:
            DataFrame whose index contains segment IDs.
            For clarity of intent, we require that the Index be named 'segment' or 'body'.

            All columns will be converted to segment properties. If a column is not
            explicitly listed in the following arguments, its property type will be inferred
            from the column dtype (or column name, in the case of label/description).

            The column dtypes can be string, category, or number, but 64-bit values
            will be downcast to 32-bit. Boolean columns are only valid as tags.

            The order of the input columns determines the order in which properties will be written.
            Neuroglancer preserves that order when displaying columns of numeric properties.

        label_col:
            Which column to use for the 'label' property which is shown in neuroglancer by default.
            If you name the column 'label', then you need not provide it here.
            Or, if your dataframe contains only one column (and it's non-numeric), we assume it is the label.

        description_col:
            Which column to use as the 'description' property.
            If you name the column 'description', then you need not provide it here.

        string_cols:
            Columns to represent as 'string' properties.
            (Usually unnecessary.  Non-numeric columns become string properties by default,
            unless they are named in tag_cols.)

        number_cols:
            Columns to represent as 'number' properties.
            (Usually unnecessary.  Numeric columns become number properties by default.)

        tag_cols:
            Columns which should be used to generate the (combined) 'tags' property.
            If you want a column to be used for both tags _and_ a different property,
            be sure to list it explicitly in both arguments.  For example:

                segment_properties_json(df, label_col='cell_class', tag_cols=['cell_class', ...])

            In addition to string/category inputs, columns with dtype bool can also be
            used for tags, in which case the False items are discarded and the True items
            are tagged with the column name.

        tag_prefix_mode:
            Either 'all' or 'disambiguate' or None.

            - If 'all', then all tags will be prefixed with the name of their source column,
              (e.g. 'status:Anchor'), other than boolean columns.

            - If 'disambiguate', then only tags which appear in multiple columns
              be prefixed with the name of their source column.

            - If None, then no disambiguation is performed.

        tag_descriptions:
            A dict of {tag: description} describing each tag value.

        col_descriptions:
            A dict of {column: description} describing each property (input column) other than 'tags'.

        drop_empty:
            If any IDs in the input have no non-empty (null or "") properties,
            then drop them from the output entirely so they don't show up in
            neuroglancer's default segment list.

        output_path:
            If provided, export the JSON to a file.

    Returns:
        JSON data (as a dict) which should be written into an 'info' file to
        host neuroglancer precomputed segment properties.

    Example:

        .. code-block: ipython

            In [41]: print(df)
            Out[41]:
                        type              class       hemilineage  has_soma  PreSyn  PostSyn
            segment
            910719  IN13B018   intrinsic_neuron               13B      True     786     3164
            160503    PFNm/p            central          DM3_CX_p      True      40      177
            908822  IN03B079   intrinsic_neuron               03B      True     176      951
            10552      DNp29  descending_neuron  putative_primary      True    1567     4559
            43797     MeVP11  visual_projection               NaN      True     158      504
            232845     R1-R6            sensory               NaN     False      25       18
            547654     R1-R6            sensory               NaN     False      57       29
            803197  IN07B019   intrinsic_neuron               TBD      True     796     3615
            75597      KCg-m            central              MBp4      True     223      802
            165485   vDeltaK            central               NaN      True      36      282

            In [42]: info = segment_properties_json(df, 'type', tag_cols=['class', 'hemilineage', 'has_soma'])

            In [43]: # Note: The output below has been indented by hand for clarity.
                ...: print(json.dumps(info))
            {
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": ["910719", "160503", "908822", "10552", "43797", "232845", "547654", "803197", "75597", "165485"]
                    "properties": [
                        {
                            "id": "type",
                            "type": "label",
                            "values": ["IN13B018", "PFNm/p", "IN03B079", "DNp29", "MeVP11", "R1-R6", "R1-R6", "IN07B019", "KCg-m", "vDeltaK"]
                        },
                        {
                            "id": "PreSyn",
                            "type": "number",
                            "data_type": "int32",
                            "values": [786, 40, 176, 1567, 158, 25, 57, 796, 223, 36]
                        },
                        {
                            "id": "PostSyn",
                            "type": "number",
                            "data_type": "int32",
                            "values": [3164, 177, 951, 4559, 504, 18, 29, 3615, 802, 282]
                        },
                        {
                            "id": "tags",
                            "type": "tags",
                            "tags": [
                                "class:central", "class:descending_neuron", "class:intrinsic_neuron",
                                "class:sensory", "class:visual_projection",
                                "has_soma",
                                "hemilineage:03B", "hemilineage:13B", "hemilineage:DM3_CX_p", "hemilineage:MBp4",
                                "hemilineage:TBD", "hemilineage:putative_primary"
                            ],
                            "values": [
                                [2, 5, 7],
                                [0, 5, 8],
                                [2, 5, 6],
                                [1, 5, 11],
                                [4, 5],
                                [3],
                                [3],
                                [2, 5, 10],
                                [0, 5, 9],
                                [0, 5]
                            ]
                        }
                    ]
                }
            }
    """
    df, string_cols, number_cols, tag_cols = _validate_args(
        df, label_col, description_col, string_cols, number_cols, tag_cols, tag_prefix_mode
    )

    if drop_empty:
        df = _drop_empty_rows(df)

    scalar_types = _scalar_property_types(
        df, label_col, description_col, string_cols, number_cols, tag_cols
    )

    json_props = []
    for col, prop_type in scalar_types.items():
        j = _scalar_property_json(df[col], prop_type, col_descriptions.get(col))
        json_props.append(j)

    if tag_cols:
        tags_prop = _tags_property_json(df, tag_cols, tag_prefix_mode, tag_descriptions)
        json_props.append(tags_prop)

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


def _validate_args(df, label_col, description_col, string_cols, number_cols, tag_cols, tag_prefix_mode):
    """
    Basic checks and convenience conversions (Series -> DataFrame, str -> list)
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(string_cols, str):
        string_cols = [string_cols]
    if isinstance(number_cols, str):
        number_cols = [number_cols]
    if isinstance(tag_cols, str):
        tag_cols = [tag_cols]

    assert df.index.name in ('body', 'segment')
    assert tag_prefix_mode in ('all', 'disambiguate', None)
    assert not (dupes := df.columns.duplicated()).any(), \
        f"Duplicated column names: {df.columns[dupes].tolist()}"

    # Check for columns that were listed in multiple arguments (except tag_cols).
    listed_scalar_cols = [label_col, description_col, *string_cols, *number_cols]
    listed_scalar_cols = [*filter(None, listed_scalar_cols)]
    listed_scalar_cols = pd.Series(listed_scalar_cols)
    dupes = listed_scalar_cols.loc[listed_scalar_cols.duplicated()].unique()
    if dupes := sorted(dupes):
        raise RuntimeError(f"Some columns were included in multiple arguments: {dupes}")

    return df, string_cols, number_cols, tag_cols


def _drop_empty_rows(df):
    if len(df) == 0 or len(df.columns) == 0:
        return df

    # Note that bool cols are guaranteed to be
    # tag cols, so False is considered empty.
    bool_cols = df.dtypes[df.dtypes == bool].index
    other_cols = df.dtypes[df.dtypes != bool].index

    valid_bool = valid_other = False
    if len(bool_cols) > 0:
        valid_bool = df[bool_cols].any(axis=1)
    if len(other_cols) > 0:
        not_null = df[other_cols].notnull()
        not_empty = df[other_cols] != ''
        valid_other = (not_null & not_empty).any(axis=1)

    return df.loc[valid_bool | valid_other]


def _scalar_property_types(df, label_col, description_col, string_cols, number_cols, tag_cols):
    """
    Determine the names and neuroglancer property types of all scalar (non-tag)
    properties that can be extracted from the given DataFrame.

    The property types will be determined from the user's explicitly provided lists
    when possible, otherwise they're inferred from the column name/dtype.

    Here, 'scalar' includes all non-tag types: number, string, label, description.
    """
    # Start with the provided types and then auto-type the other columns below.
    # We initialize first with tags, but permit overwriting those with scalar.
    # (A column is are permitted to be listed twice if it's both tag and scalar.)
    prop_types = {c: 'tags' for c in tag_cols}
    prop_types |= {c: 'string' for c in string_cols}
    prop_types |= {c: 'number' for c in number_cols}

    if label_col:
        prop_types[label_col] = 'label'
    if description_col:
        prop_types[description_col] = 'description'

    if unknown_cols := set(prop_types.keys()) - set(df.columns):
        raise RuntimeError(f"Columns not found: {unknown_cols}")

    # If there's only one column, and it wasn't explicitly listed,
    # AND it's non-numeric, we presume it's meant to be the 'label' column.
    if (
        len(df.columns) == 1
        and not prop_types
        and not np.issubdtype(df.dtypes.iloc[0], np.number)
    ):
        prop_types = {df.columns[0]: 'label'}

    # Infer the types of unlisted columns from either the name or dtype
    for name, dtype in df.dtypes.items():
        if dtype == bool and prop_types.get(name) != 'tags':
            raise RuntimeError(f"Column '{name}': Boolean columns are only valid as tag_cols")
        elif prop_types.get(name) == 'number' and not np.issubdtype(dtype, np.number):
            raise RuntimeError(f"Column '{name}': Not valid as number_cols (dtype: {dtype})")
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

    # Re-order to match original input columns.
    # Property order determines appearance order in neuroglancer.
    prop_types = {c: prop_types[c] for c in df.columns}

    # Return scalar properties only
    prop_types = {k:v for k,v in prop_types.items() if v != 'tags'}
    return prop_types


def _scalar_property_json(s, prop_type, description):
    """
    Constructs the JSON for any segment property
    other than the 'tags' property.
    """
    if prop_type == 'number':
        return _scalar_number_property_json(s, description)

    prop = {
        'id': s.name,
        'type': prop_type,
        'values': s.fillna("").astype(str).tolist()
    }

    if description:
        prop['description'] = description

    return prop


def _scalar_number_property_json(s, description):
    """
    Constructs the JSON for a 'number' property.
    """
    if s.isnull().any():
        raise RuntimeError(
            f"Column {s.name} contans NaN entries. "
            "I'm not sure what to do with NaN values in numeric properties."
        )

    if s.dtype == np.float64:
        s = s.astype(np.float32)

    if s.dtype in (np.int64, np.uint64):
        s = _downcast_int64_series(s)

    prop = {
        'id': s.name,
        'type': 'number',
        'data_type': s.dtype.name,
        'values': s.tolist()
    }

    if description:
        prop['description'] = description

    return prop


def _downcast_int64_series(s):
    # Convert int64 to (u)int32 if we can do so losslessly.
    for dtype32 in (np.int32, np.uint32):
        info32 = np.iinfo(dtype32)
        if s.min() >= info32.min and s.max() <= info32.max:
            return s.astype(dtype32)

    raise RuntimeError(
        f"Can't create a property for column: '{s.name}'. "
        "Neuroglancer doesn't support 64-bit integer properties, "
        "and your data exceeds the limits of (u)int32."
    )


def _tags_property_json(df, tags_columns, tag_prefix_mode, tag_descriptions):
    """
    Constructs the JSON for the 'tags' segment property.
    """
    tags_df = df[[]].copy()

    # Individually convert each column to Categorical
    # before we combine into a unified Categorical below.
    for c in tags_columns:
        tags_df[c] = _convert_to_categorical(df[c])

    _insert_tag_prefixes(tags_df, tag_prefix_mode, df.dtypes)

    # Convert to a single unified Categorical dtype.
    tag_sets = [dtype.categories for dtype in tags_df.dtypes]
    all_tags = sorted({*chain(*tag_sets)})
    tags_df = tags_df.astype(pd.CategoricalDtype(categories=all_tags))

    # Tags are represented as a list-of-lists of sorted codes.
    codes_df = pd.DataFrame({c: s.cat.codes for c, s in tags_df.items()})
    sorted_codes = np.sort(codes_df.to_numpy(), axis=1)
    codes_lists = [
        [x for x in row if x != -1]  # Drop nulls
        for row in sorted_codes.tolist()
    ]

    prop = {
        'id': 'tags',
        'type': 'tags',
        'tags': all_tags,
        'values': codes_lists,
    }

    if tag_descriptions:
        prop['tag_descriptions'] = _tag_description_list(all_tags, tag_descriptions)

    return prop


def _convert_to_categorical(s):
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
    s = _replace_spaces(s)

    # We interpret empty string as null
    if '' in s.dtype.categories:
        s = s.cat.remove_categories([''])

    return s


def _replace_spaces(s):
    """
    Replace all spaces in the given Categorical Series with underscores.
    (Spaces are forbidden in neuroglancer tags.)
    """
    renames = {
        cat: str(cat).replace(' ', '_')
        for cat in s.dtype.categories
    }

    try:
        return s.cat.rename_categories(renames)
    except ValueError:
        # If both "foo bar_baz" and "foo_bar baz" exist in the
        # original data, they will both map to "foo_bar_baz", and
        # rename_categories() complains about the duplicate category.
        return s.str.replace(' ', '_').astype('category')


def _insert_tag_prefixes(df, tag_prefix_mode, orig_dtypes):
    """
    Insert prefixes onto tags in df according to the tag_prefix_mode (if any).
    The columns of df must already be Categorical.

    Tags in columns which were originally bool (before we converted them
    to Categorical) require no prefix, so we refer to orig_dtypes to skip
    those columns.

    Modifies df in-place.
    """
    if tag_prefix_mode is None:
        return

    if tag_prefix_mode == 'disambiguate':
        _disambiguate_tags(df)
        return

    assert tag_prefix_mode == 'all'
    for c, s in list(df.items()):
        if orig_dtypes[c] == 'bool':
            # Boolean columns need no prefix;
            # the column name is the entire tag.
            continue

        # Prefix must not contain spaces (forbidden by neuroglancer)
        # or colons (because we use ':' as the separator).
        prefix = s.name.replace(' ', '_').replace(':', '_')
        df[c] = s.cat.rename_categories([
            f'{prefix}:{cat}'
            for cat in s.dtype.categories
        ])


def _disambiguate_tags(df):
    """
    Given a dataframe in which all columns are Categoricals,
    find category values that are common across multiple columns
    and prepend a prefix (the column name) to such values
    to make sure no category value is duplicated from one column
    to the next.

    Modifies df in-place.
    """
    col_tags = [
        (col, tag)
        for col in df.columns
        for tag in df[col].dtype.categories
    ]
    col_tags = pd.DataFrame(col_tags, columns=['col', 'tag'])

    # Drop everything except the duplicate tags.
    is_dup = col_tags.duplicated('tag', keep=False)
    col_tags = col_tags.loc[is_dup].copy()

    # Prefix must not contain spaces (forbidden by neuroglancer)
    # or colons (because we use ':' as the separator).
    prefixes = col_tags['col'].str.replace(r'[: ]', '_')
    col_tags['new_tag'] = prefixes + ':' + col_tags['tag']

    for col, old_new_df in col_tags.groupby('col')[['tag', 'new_tag']]:
        renames = dict(old_new_df.to_numpy())
        df[col] = df[col].cat.rename_categories(renames)


def _tag_description_list(all_tags, tag_descriptions):
    """
    Given the list of all tags and a mapping of tags to descriptions,
    return the descriptions in the same order as all_tags.
    """
    tag_descriptions = {
        str(k).replace(' ', '_'): v
        for k,v in tag_descriptions.items()
    }

    td = []
    for t in all_tags:
        d = tag_descriptions.get(t, None)

        # If we didn't find it, try stripping the tag prefix (if any)
        if not d and ':' in t:
            t2 = t[1+t.index(':'):]
            d = tag_descriptions.get(t2, None)

        # If we still didn't find it, emit the tag itself.
        td.append(d or t)

    return td


def segment_properties_to_dataframe(js):
    """
    Converts JSON to DataFrame.
    This is primarily for testing and demonstration.

    Note:
        A user's original dataframe is not generally recoverable
        from the segment properties JSON representation
        if tags are present.

        This function translates all tags to boolean columns
        (one column per tag), rather than strings or categoricals
        (which is how the user probably originally had them).
        The resulting dataframe might contain MANY boolean columns,
        likely requiring more memory than the user's original
        dataframe did.

        No attempt is made to parse the prefixes of the tags to group
        them into shared columns.
    """
    all_props = js['inline']['properties']
    scalar_props = [prop for prop in all_props if prop['type'] != 'tags']
    tags_props =   [prop for prop in all_props if prop['type'] == 'tags']

    segment_ids = [*map(int, js['inline']['ids'])]
    prop_ids = [prop['id'] for prop in scalar_props]
    values = [prop['values'] for prop in scalar_props]
    dtypes = {
        prop['id']: prop['data_type']
        for prop in scalar_props
        if 'data_type' in prop
    }

    df = pd.DataFrame(dict(zip(prop_ids, values)), segment_ids)
    df = df.astype(dtypes)
    if not tags_props:
        return df

    unique_tags = tags_props[0]['tags']
    code_lists = tags_props[0]['values']

    # Flatten the lists of codes,
    # but keep track of the rows they came from.
    cols = [*chain(*code_lists)]
    rows = np.repeat(
        range(len(code_lists)),
        [*map(len, code_lists)]
    )

    flags = np.zeros((len(code_lists), len(unique_tags)), bool)
    flags[rows, cols] = True

    tags_df = pd.DataFrame(flags, segment_ids, unique_tags)
    return pd.concat((df, tags_df), axis=1)
