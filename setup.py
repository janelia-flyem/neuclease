"""
neuclease setup.py
"""
from setuptools import find_packages, setup
import versioneer

# For now, requirements are only specified in the conda recipe, not here.
#
# TODO: Specify them here (or requirements.txt),
#       and have the conda recipe import them via:
#
#    run:
#      - python
#       {% for dep in data['install_requires'] %}
#      - {{ dep.lower() }}
#      {% endfor %}
#

setup(
    name='neuclease',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Tools for computing interactive "cleaves" of agglomerated neuron fragments from a DVID server.',
    url='https://github.com/janelia-flyem/neuclease',
    packages=find_packages(),
    package_data={'neuclease.misc': ['*.json']},
    entry_points={
        'console_scripts': [
            'neuclease_cleave_server = neuclease.bin.cleave_server_main:main',
            'adjust_focused_points = neuclease.bin.adjust_focused_points:main',
            'check_tarsupervoxels_status = neuclease.bin.check_tarsupervoxels_status:main',
            'ingest_synapses = neuclease.bin.ingest_synapses:main',
            'decimate_existing_mesh = neuclease.bin.decimate_existing_mesh:main',
            'export_sparsevol = neuclease.bin.export_sparsevol:main',
            'neuron_mito_stats = neuclease.misc.neuron_mito_stats:main',
            'copy_vnc_subvolume = neuclease.misc.copy_vnc_subvolume:main',
            'point_neighborhoods = neuclease.misc.point_neighborhoods:main',
            'vnc_group_analysis = neuclease.misc.vnc_group_analysis_main:main',
            'prepare_user_branches = neuclease.misc.prepare_user_branches:main',
            'supervoxel_meshes_for_body = neuclease.misc.supervoxel_meshes_for_body:main',
            'sparse_connectome = neuclease.misc.sparse_connectome:main',
            'make_bucket_public = neuclease.misc.neuroglancer:make_bucket_public',
            'derived-updates = neuclease.misc.derived_updates:main',
            'dvid_virtual_n5_server = neuclease.bin.virtual_n5_server_main:main',
            'precomputed_virtual_n5_server = neuclease.bin.virtual_n5_from_precomputed:main'
        ]
    }
)
