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
              'adjust_focused_points = neuclease.bin.adjust_focused_points:main'
          ]
       }
     )
