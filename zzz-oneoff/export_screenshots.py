"""
This was a one-off script to automate the process of extracting a series of screenshots of specific coordinates in neuroglancer.
It relies on Chrome and the third-party 'screenshot' tool (MacOS only):
https://github.com/alexdelorenzo/screenshot

Maybe this code will be a useful example if I need to do this again someday.
"""
import copy
import json
import time
import subprocess

import pandas as pd

TAB = '29'
TABLE_CSV = 'DECISIONS-TAB29.csv'
CHROME_EXE = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

url_base = "https://neuroglancer-demo.appspot.com/#!"

default_url_data = \
{
    "layers": [
        {
            "source": "brainmaps://274750196357:hemibrain:raw_clahe_yz",
            "type": "image",
            "name": "raw_clahe_yz"
        },
        {
            "source": "brainmaps://274750196357:hemibrain:base20180227_8nm_watershed_fixed",
            "type": "segmentation",
            "skeletonRendering": {
                "mode2d": "lines_and_points",
                "mode3d": "lines"
            },
            "name": "base20180227_8nm_watershed_fixed",
            "visible": False
        },
        {
            "source": f"brainmaps://274750196357:hemibrain:seg_v2_tab{TAB}:rsg32_16_8_nom1247_iso",
            "type": "segmentation",
            "skeletonRendering": {
                "mode2d": "lines_and_points",
                "mode3d": "lines"
            },
            "name": "secgan_seg_8nm_tab29",
            "visible": False
        }
    ],
    "navigation": {
        "pose": {
            "position": {
                "voxelSize": [
                    8,
                    8,
                    8
                ],
                "voxelCoordinates": [
                    14494,
                    26967,
                    19343
                ]
            }
        },
        "zoomFactor": 3.501958434848348
    },
    "perspectiveZoom": 60.0,
    "showSlices": False,
    "layout": "4panel"
}

#
# Chrome needs to be open already BEFORE you run this script!
#
def neuroglancer_screenshot(url_data, name, delay=3.0, write_link=False):
    url = url_base + json.dumps(url_data).replace('\n', '').replace(' ', '')
    subprocess.run([CHROME_EXE, url], check=True)
    time.sleep(delay)

    # https://github.com/alexdelorenzo/screenshot
    subprocess.run(['screenshot', "Google Chrome", '-f', f'{name}.png'], check=True)
    print(f"Wrote {name}")
    if write_link:
        with open(f"{name}-link.txt", 'w') as f:
            f.write(url + '\n')


def main():
    decisions_df = pd.read_csv(TABLE_CSV)
    
    for row in decisions_df.itertuples(index=True):
        url_data = copy.copy(default_url_data)
        url_data['navigation']['pose']['position']['voxelCoordinates'] = [row.xa, row.ya, row.zb]

        # Grayscale only
        neuroglancer_screenshot(url_data, f"index-{row.index:02d}-raw")
        
        # Original supervoxels
        url_data['layers'][1]['segments'] = [str(row.sv_a), str(row.sv_b)]
        url_data['layers'][1]['visible'] = True
        neuroglancer_screenshot(url_data, f"index-{row.index:02d}-sv-original")
        url_data['layers'][1]['visible'] = False

        # New agglo
        url_data['layers'][2]['visible'] = True
        neuroglancer_screenshot(url_data, f"index-{row.index:02d}-sv-secgan")
        url_data['layers'][2]['visible'] = False

if __name__ == "__main__":
    main()
