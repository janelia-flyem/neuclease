import os
import json
import urllib
import datetime
from textwrap import dedent

import pandas as pd
from scipy.spatial.transform import Rotation

import holoviews as hv
import hvplot.pandas
from bokeh.layouts import row, gridplot
from bokeh.models import ColumnDataSource, OpenURL, TapTool, BoxSelectTool, HoverTool, TextAreaInput, CustomJS
from bokeh.plotting import figure, output_file, save as bokeh_save, output_notebook, show  # noqa
from bokeh.io import export_png

from neuclease.util import tqdm_proxy


TEMPLATE_LINK = "https://clio-ng.janelia.org/#!%7B%22title%22:%22CNS%20Brain%22%2C%22dimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%2C%22position%22:%5B13580.076171875%2C32452.529296875%2C30724.5703125%5D%2C%22crossSectionScale%22:4.215747439896878%2C%22projectionOrientation%22:%5B-0.019751783460378647%2C0.40617886185646057%2C0.008212516084313393%2C0.913543164730072%5D%2C%22projectionScale%22:46839.621432360225%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://flyem-cns-z0720_07m_br-40-06-derived/clahe_yz_cl0.035%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22rendering%22%2C%22name%22:%22em-clahe%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata6.int.janelia.org:9000/7e4c96/segmentation?dvid-service=https://ngsupport-bmcp5imp6q-uk.a.run.app%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22toolBindings%22:%7B%22Q%22:%22selectSegments%22%7D%2C%22tab%22:%22segments%22%2C%22name%22:%22brain-seg%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://flyem-cns-roi-7c971aa681da83f9a074a1f0e8ef60f4/brain-shell-l2%22%2C%22subsources%22:%7B%22default%22:true%2C%22properties%22:true%2C%22mesh%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22pick%22:false%2C%22tab%22:%22rendering%22%2C%22selectedAlpha%22:0%2C%22saturation%22:0.53%2C%22meshSilhouetteRendering%22:4%2C%22segments%22:%5B%221%22%5D%2C%22colorSeed%22:1336242844%2C%22segmentDefaultColor%22:%22#ffffff%22%2C%22name%22:%22brain-shell%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://flyem-cns-roi-7c971aa681da83f9a074a1f0e8ef60f4/halfbrain-roi%22%2C%22transform%22:%7B%22matrix%22:%5B%5B1%2C0%2C0%2C4096%5D%2C%5B0%2C1%2C0%2C4096%5D%2C%5B0%2C0%2C1%2C4096%5D%5D%2C%22outputDimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%7D%2C%22subsources%22:%7B%22default%22:true%2C%22properties%22:true%2C%22mesh%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22pick%22:false%2C%22tab%22:%22segments%22%2C%22selectedAlpha%22:0.63%2C%22saturation%22:0.5%2C%22meshSilhouetteRendering%22:4%2C%22segments%22:%5B%2217%22%2C%2218%22%2C%2219%22%5D%2C%22name%22:%22halfbrain-neuropil%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22dvid://http://emdata6.int.janelia.org:9000/f3969dc575d74e4f922a8966709958c8/nuclei-seg?dvid-service=https://ngsupport-bmcp5imp6q-uk.a.run.app%22%2C%22tab%22:%22segments%22%2C%22segmentQuery%22:%22%20%22%2C%22name%22:%22nuclei-seg%22%2C%22archived%22:true%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://https://emdata6-erivan.janelia.org/0ea0d0/segmentation?dvid-service=https://ngsupport-bmcp5imp6q-uk.a.run.app%22%2C%22transform%22:%7B%22matrix%22:%5B%5B1%2C0%2C0%2C4096%5D%2C%5B0%2C1%2C0%2C4096%5D%2C%5B0%2C0%2C1%2C4096%5D%5D%2C%22outputDimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%7D%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%2C%22skeletons%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22segments%22%2C%22crossSectionRenderScale%22:2%2C%22name%22:%22halfbrain-reference%22%2C%22archived%22:true%7D%2C%7B%22type%22:%22annotation%22%2C%22source%22:%7B%22url%22:%22local://annotations%22%2C%22transform%22:%7B%22outputDimensions%22:%7B%22x%22:%5B8e-9%2C%22m%22%5D%2C%22y%22:%5B8e-9%2C%22m%22%5D%2C%22z%22:%5B8e-9%2C%22m%22%5D%7D%7D%7D%2C%22tool%22:%22annotatePoint%22%2C%22tab%22:%22annotations%22%2C%22annotations%22:%5B%7B%22pointA%22:%5B39298%2C0%2C9578%5D%2C%22pointB%22:%5B41856%2C55296%2C40316%5D%2C%22type%22:%22axis_aligned_bounding_box%22%2C%22id%22:%227ec7d4d742ac94ed304f4b1d9c535a162c947ab4%22%7D%2C%7B%22pointA%22:%5B4125.13134765625%2C20381.453125%2C28350.43359375%5D%2C%22pointB%22:%5B6771%2C48774.40625%2C44534.76953125%5D%2C%22type%22:%22axis_aligned_bounding_box%22%2C%22id%22:%22dde25826ea384cdc8d89f22f28dc7f5992c43830%22%7D%5D%2C%22name%22:%22realigned-tabs%22%2C%22archived%22:true%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22dvid://http://emdata6.int.janelia.org:9000/f3969dc575d74e4f922a8966709958c8/segmentation?dvid-service=https://ngsupport-bmcp5imp6q-uk.a.run.app%22%2C%22subsources%22:%7B%22default%22:true%2C%22meshes%22:true%2C%22skeletons%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22segments%22%2C%22segmentQuery%22:%222%22%2C%22colorSeed%22:2823534454%2C%22name%22:%22sv%22%2C%22archived%22:true%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://flyem-cns-roi-7c971aa681da83f9a074a1f0e8ef60f4/optic-freeze%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22pick%22:false%2C%22tab%22:%22source%22%2C%22segmentDefaultColor%22:%22#ffffff%22%2C%22name%22:%22optic-freeze%22%2C%22archived%22:true%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22brain-seg%22%7D%2C%22layout%22:%223d%22%2C%22statistics%22:%7B%22size%22:936%7D%2C%22selection%22:%7B%22size%22:291%2C%22visible%22:false%7D%7D"


# The link above uses an internal name.
# This function will swap it to the obscure public one.
def update_template_link_server(server):
    assert server.startswith('http')
    global TEMPLATE_LINK
    TEMPLATE_LINK = TEMPLATE_LINK.replace('http://emdata6.int.janelia.org:9000', server)


# For each ROI, find a suitable viewing plane in the neuroglancer 3D view,
# and then copy the projectionOrientation (a quaternion) from the JSON settings.
projectionOrientations = {
    'ME(R)': [0, -0.33707213401794434, 0, -0.9414788484573364],
    'LO(R)': [-0.12742365896701813, -0.3862779438495636, -0.12784865498542786, 0.9045481085777283],
    'LOP(R)': [0.0161326602101326, -0.09871263056993484, -0.16283509135246277, 0.981570303440094],
}

# These descriptions are used to label our scatter plots.
projectionAxes = {
    'ME(R)': ['<-- posterior / anterior -->', '<-- ventral / dorsal -->'],
    'LO(R)': ['<-- anterior / posterior -->', '<-- ventral / dorsal -->'],
    'LOP(R)': ['<-- anterior / posterior -->', '<-- ventral / dorsal -->'],
}


def rotate_cns_points(syn_pos):
    """
    Perform a rotation on the points (columns 'x', 'y', 'z') in syn_pos
    according to some hand-picked rotations (see below).

    The new points will be stored in columns px, py, pz
    """
    # Load those orientations as a scipy Rotation.
    # Note:
    #    In neuroglancer we moved the camera to the desired plane, but in the plots
    #    below we need to go the other way: we'll move the points to the "camera".
    #    So, here we *invert* the rotation we obtained from neuroglancer.
    rotations = {k: Rotation.from_quat(v).inv() for k,v in projectionOrientations.items()}

    syn_pos[['px', 'py', 'pz']] = syn_pos[[*'xyz']]

    # Compute rotations
    for roi, rot in rotations.items():
        selection = syn_pos.eval('roi == @roi')
        rotated = rot.as_matrix() @ syn_pos.loc[selection, ['x', 'y', 'z']].values.T
        syn_pos.loc[selection, ['px', 'py', 'pz']] = rotated.T


def plot_neuron_positions(syn_pos_df, type_cell, type_syn, roi, template_link=None):
    """
    Generate a fancy plot of mean synapse positions,
    scaled and colored according to the synapse count.

    Returns a bokeh figure, not a holoviews figure.
    """
    template_link = template_link or TEMPLATE_LINK
    df = syn_pos_df.query('type == @type_cell and type_syn == @type_syn and roi == @roi')

    # Making a copy so I can add temporary columns
    df = df.copy()

    # Customize the aspect ratio
    height = 500
    width = 500
    xlim = [df['px'].min(), df['px'].max()]
    xwidth = (xlim[1] - xlim[0])
    xlim[0] -= 0.05 * xwidth
    xlim[1] += 0.05 * xwidth

    ylim = [df['py'].min() - 20, df['py'].max() + 20]
    ywidth = (ylim[1] - ylim[0])
    ylim[0] -= 0.05 * ywidth
    ylim[1] += 0.05 * ywidth

    # Compute a normalized version of the count for the size display
    # FIXME:
    #   This is just a fudge factor that seems to work okay in the ME and LO.
    #   It would be better to determine the ideal scaling factor
    #   according to the spatial spread of the data.
    if len(df) <= 1 or df['count'].std(ddof=0) == 0:
        df['count_scaled'] = 100
    else:
        df['count_scaled'] = (df['count'] - df['count'].mean()) / df['count'].std(ddof=0)
        df['count_scaled'] *= 30
        df['count_scaled'] -= df['count_scaled'].min()
        df['count_scaled'] += 5

    # Bokeh draws points from first to last, with last on top.
    # To make outliers more prominent, sort by deviation from the mean.
    # That puts the 'ordinary' things on the bottom and the outliers are drawn last, on top.
    df['dev'] = (df['count'] - df['count'].mean()).abs()
    df = df.sort_values('dev')

    # Better column name (for the hover tip)
    df = df.rename(columns={'count': type_syn})

    # Specify what should appear in the hover text
    hover = HoverTool(
        tooltips=[
            ("body", "@body"),
            (type_syn, f"@{type_syn}")])

    hv_plot = df.hvplot.scatter(
        'px',
        'py',
        size='count_scaled',
        color=type_syn,
        flip_yaxis=True,
        height=height + 50,
        width=width,
        xlim=xlim,
        ylim=ylim,
        title=f'{type_cell} {type_syn} counts in {roi} ({len(df)} cells)',
        xlabel=projectionAxes.get(roi, '<-- right / left -->')[0],
        ylabel=projectionAxes.get(roi, '<-- ventral / dorsal -->')[1],
        xticks=[1e6],
        yticks=[1e6],
        cmap='viridis',
        tools=['lasso_select', 'tap', hover]
    )

    # Some options must be specified in the following way, via the holoviews opts() method.
    # - Here, axiswise=True says not to force this plot's axes to use the same scale as other
    #   plots in the figure (in case you add multiple plots to a figure)
    # - The active_tools are the ones that are preselected by default
    #   (e.g. by default, mouse-drag will select points, not pan the view)
    hv_plot = hv_plot.opts(axiswise=True, active_tools=['wheel_zoom', 'lasso_select', 'tap'])

    # For further customization, we need to use the bokeh API to manipulate the plot.
    # Hence, we 'render' the holoviews object to convert it into a raw bokeh plot.
    bokeh_plot = hv.render(hv_plot)

    # The JavaScript features below will make use of
    # variables stored in the plot's "ColumnDataSource".
    # The CDS already contains columns for the data in the scatter plot (e.g. px, py, color, etc.),
    # but we need to sneak some extra stuff in there: body IDs and 3D coordinates (mean positions).
    cds = bokeh_plot.select(type=ColumnDataSource)[0]
    cds.add(df['bodyId'], 'body')
    cds.add(df['x'], 'x')
    cds.add(df['y'], 'y')
    cds.add(df['z'], 'z')

    # If the user selects some points, show the body IDs in a text box next to the plot.
    # Also, copy the body IDs to the user's clipboard.
    # TODO: Add a button that opens a neuroglancer link for all selected bodies.
    #       (Right now, the link is only opened when a single body is clicked.)
    # Docs: https://docs.bokeh.org/en/2.4.0/docs/user_guide/interaction/widgets.html#textareainput
    textbox = TextAreaInput(rows=20, cols=20)
    cds.selected.js_on_change(
        'indices',
        CustomJS(args={'cds': cds, 'textbox': textbox}, code="""\
            // Extract the body IDs from the column data that was stored in JavaScript.
            // The user's selected indices are passed via the callback object (cb_obj).
            const inds = cb_obj.indices;
            var bodies = [];
            for (let i = 0; i < inds.length; i++) {
                bodies.push(cds.data['body'][inds[i]]);
            }

            // Write the body IDs into the widget.
            textbox.value = bodies.join('\\n');

            // If possible, also copy the body IDs to the user's clipboard.
            //
            // NOTE:
            //   Javascript doesn't allow writing to the clipboard unless:
            //   This page is served over https (so, this notebook won't work...)
            //   OR the page is hosted via a static local file://
            try {
                navigator.clipboard.writeText(bodies.join('\\n'));
            }
            catch (err) {
                console.error("Couldn't write body list to clipboard:", err)
            }
    """))

    # Pre-generate a neuroglancer link to use if the user clicks on a point in the scatter plot.
    # Start with a generic link, then overwrite some settings.
    link_data = parse_nglink(template_link)
    link_data['position'] = [111111111, 222222222, 333333333]
    if roi in projectionOrientations:
        link_data['projectionOrientation'] = projectionOrientations[roi]

    # Note: In the template link, I know layer [1] is the segmentation layer.
    link_data['layers'][1]['segmentQuery'] = "999999999"
    link_data['layers'][1]['segments'] = ["999999999"]
    template_link = format_nglink('https://clio-ng.janelia.org', link_data)

    # The OpenURL function will replace variables in the link (e.g. @x)
    # with data from the ColumnDataSource (we added these variables to the CDS above).
    template_link = template_link.replace('111111111', '@x')
    template_link = template_link.replace('222222222', '@y')
    template_link = template_link.replace('333333333', '@z')
    template_link = template_link.replace('999999999', '@body')

    # If the user clicks on a point, open the URL.
    # Docs: https://docs.bokeh.org/en/2.4.0/docs/user_guide/interaction/callbacks.html#openurl
    taptool = bokeh_plot.select(type=TapTool)
    taptool.callback = OpenURL(url=template_link)

    # Return
    return row([bokeh_plot, textbox])


def syn_histogram(syn_pos_df, type_cell, type_syn, roi):
    df = syn_pos_df.query('type == @type_cell and type_syn == @type_syn and roi == @roi')
    p = df['count'].hvplot.hist(
        title=f'{type_cell} {type_syn} counts in {roi} ({len(df)} cells)',
        xlabel=f'{type_syn} count',
        ylabel='cell count',
        color='count',
        cmap='viridis'
    )
    # If this plot is later shown with other plots within a Layout,
    # don't alter the plot axes to match the other plots.
    # Keep the X/Y limits that are customized for this specific plot.
    p = p.opts(axiswise=True)
    return p


def write_onepage_png_report(export_dir, cell_type_counts):
    """
    Helper function for emit_reports().

    Produces an index listing of the plots in the given export directory.
    """
    cell_types = sorted(cell_type_counts.index)

    with open(f'{export_dir}/full-report.html', 'w') as f:
        f.write(dedent("""\
            <html>
            <head>
            <title>Synapse Distributions</title>

            <!-- https://mottie.github.io/tablesorter/docs/ -->

            <!-- FIXME: I'm hot-linking here, because I'm a generally irresponsible person. -->

            <!-- jQuery: required (tablesorter works with jQuery 1.2.3+) -->
            <script src="https://mottie.github.io/tablesorter/docs/js/jquery-1.2.6.min.js"></script>

            <!-- Pick a theme, load the plugin & initialize plugin -->
            <link href="https://mottie.github.io/tablesorter/dist/css/theme.default.min.css" rel="stylesheet">
            <script src="https://mottie.github.io/tablesorter/dist/js/jquery.tablesorter.min.js"></script>
            <script src="https://mottie.github.io/tablesorter/dist/js/jquery.tablesorter.widgets.min.js"></script>
            <script>
            $(function(){
                $('table').tablesorter({
                    widgets        : ['zebra', 'columns'],
                    usNumberFormat : false,
                    sortReset      : true,
                    sortRestart    : true
                });
            });
            </script>

            <!-- Override the default style to shrink the width of the table -->
            <style>
                .tablesorter-default{
                    width:200px
                }
            </style>

            </head>
            <body>
            """))
        f.write("<h2>Synapse Distributions</h2>\n")
        f.write('<table class="tablesorter">\n')
        f.write(dedent("""\
            <thead>
              <tr>
                <th>Cell Type</th>
                <th>count</th>
                <th>link</th>
              </tr>
            </thead>
        """))
        f.write('<tbody>\n')
        for i, cell in enumerate(cell_types):
            f.write(dedent(f"""\
            \n
              <tr>
                <td><a href="#{i}">{cell}</a><br></td>
                <td><a href="#{i}">{cell_type_counts.loc[cell]}</a><br></td>
                <td><a href="html/{cell}.html">(html)</a><br></td>
              </tr>
        """))
        f.write('</tbody>\n')
        f.write('</table>\n')

        for i, cell in enumerate(cell_types):
            f.write(dedent(f"""\
                <h2 id="{i}">{cell} <a href="html/{cell}.html">(html)</a></h2>
                <br>
                <img src="png/{cell}.png">
                <br>
            """))
        f.write(dedent("""\
            </body>
            </html>
        """))


def emit_reports(stats, cell_types=None, rois=None, export=True, template_link=None):
    """
    Produce a grid of plots for each of the given cell types,
    using the table of synapse stats for each body and roi.

    Optionally export the layouts to disk as both html and png.

    Reutrns a list of bokeh layouts.
    """
    # Filter stats for requested types/rois
    cell_types = cell_types or stats['type'].unique()
    rois = rois or stats['roi'].unique()
    stats = stats.query('type in @cell_types and roi in @rois').copy()

    stats['type'] = stats['type'].str.replace('/', '-')
    stats['roi'] = pd.Categorical(stats['roi'], categories=rois, ordered=True)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    export_dir = f'reports-{today}'
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(f'{export_dir}/png', exist_ok=True)
    os.makedirs(f'{export_dir}/html', exist_ok=True)

    layouts = []
    for cell_type, ctdf in tqdm_proxy(stats.groupby('type', sort=False), total=stats['type'].nunique()):
        plots = []

        # Make plots for all ROIs that that contain any cells of this type
        for roi, rdf in ctdf.groupby('roi', sort=True, observed=False):
            # Histogram
            for type_syn, df in rdf.groupby('type_syn', sort=True, observed=False):
                if len(df) == 0:
                    plots.append(None)
                else:
                    p = syn_histogram(df, cell_type, type_syn, roi)
                    plots.append(hv.render(p))

            # Scatter
            for type_syn, df in rdf.groupby('type_syn', sort=True, observed=False):
                if len(df) == 0:
                    plots.append(None)
                else:
                    p = plot_neuron_positions(df, cell_type, type_syn, roi, template_link)
                    plots.append(p)

        # Combine into a single layout for this cell type
        layout = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)], merge_tools=False)
        layouts.append(layout)

        if export:
            cell_type = cell_type.replace('/', '-')

            # Export the layout as one png
            path = f'{export_dir}/png/{cell_type}.png'
            rm_f(path)
            export_png(layout, filename=path)

            # Export as html
            path = f'{export_dir}/html/{cell_type}.html'
            rm_f(path)
            output_file(filename=path, title=f'{cell_type} report')
            bokeh_save(layout)

    if export:
        # Create an index page
        # Determine counts, tweak some names (avoid '/' in the name)
        cell_type_counts = stats.drop_duplicates('bodyId')['type'].value_counts()
        cell_types = [ct.replace('/', '-') for ct in cell_type_counts.index]
        cell_type_counts.index = cell_types

        write_onepage_png_report(export_dir, cell_type_counts)

    return layouts


def rm_f(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def parse_nglink(link):
    """
    Extract the json content from a neuroglancer URL.

    Args:
        link:
            A neuroglancer link, copied from your browser location bar.

    Returns:
        dict
    """
    url_base, pseudo_json = link.split('#!')
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return data


def format_nglink(ng_server, link_json_settings):
    """
    Construct a neuroglancer URL from a server name and neuroglancer JSON settings.

    Args:
        ng_server:
            A server that is hosting the static neuroglancer JSON bundle,
            e.g. 'https://clio-ng.janelia.org'

        link_json_settings:
            The JSON settings from a neuroglancer scene,
            as obtained via the neuroglancer UI or via parse_nglink().

    Returns:
        str
    """
    return ng_server + '/#!' + urllib.parse.quote(json.dumps(link_json_settings))
