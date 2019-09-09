from setuptools import find_packages, setup

setup( name='neuclease',
       version='0.1',
       description='Tools for computing interactive "cleaves" of agglomerated neuron fragments from a DVID server.',
       url='https://github.com/janelia-flyem/neuclease',
       packages=find_packages(),
       package_data={},
       entry_points={
          'console_scripts': [
              'neuclease_cleave_server = neuclease.bin.cleave_server_main:main',
              'adjust_focused_points = neuclease.bin.adjust_focused_points:main',
              'check_tarsupervoxels_status = neuclease.bin.check_tarsupervoxels_status:main',
              'ingest_synapses = neuclease.bin.ingest_synapses:main',
              'decimate_existing_mesh = neuclease.bin.decimate_existing_mesh:main'
          ]
       }
     )
