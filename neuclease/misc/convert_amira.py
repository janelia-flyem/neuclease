import os
import h5py
import numpy as np


EXAMPLE_AMIRAMESH_BINARY_HEADER = """\
# AmiraMesh BINARY-LITTLE-ENDIAN 2.1


define Lattice 28146 38380 1

Parameters {
    Expression "A+B",
    Content "28146x38380x1 float, uniform coordinates",
    BoundingBox 0 28146 0 38380 0 1,
    CoordType "uniform"
}

Lattice { float Data } @1

# Data section follows
@1
"""


def load_amiramesh_binary(path):
    """
    Read an AmiraMesh Binary image.
    
    Notes:
        - Only supports "binary-little-endian 2.1"
        - For now, supports only float32
        - Loads the whole image into RAM

    The format is pretty much self-explanatory if you actually
    peek inside the file with a text editor, but this page
    also describes it:
    
        https://www.csc.kth.se/~weinkauf/notes/amiramesh.html
    
    Returns:
        np.ndarray
    """
    # Load the whole image into RAM
    data = open(path, 'rb').read()
    
    # Check the first line: We don't support ascii format.
    FORMAT_LINE = EXAMPLE_AMIRAMESH_BINARY_HEADER.split('\n')[0].encode('utf-8')
    if data[:len(FORMAT_LINE)] != FORMAT_LINE:
        raise RuntimeError(f"Expected file to start with {FORMAT_LINE}")
    
    # Search for the start of the image data
    MARKER = b'Data section follows\n@1\n'
    img_offset = data.index(MARKER) + len(MARKER)
    header_lines = data[:img_offset].decode('utf-8').split('\n')

    # Check dtype: require float32 for now
    if 'Lattice { float Data } @1' not in header_lines:
        raise RuntimeError("Expected Data section to be float")

    dtype = np.float32
    element_size = dtype().nbytes

    # Read shape (X,Y,Z)
    for line in header_lines:
        if line.startswith('define Lattice'):
            X, Y, Z = map(int, line.split()[-3:])
            break

    # Sanity chceck: file size
    img_end = img_offset + (element_size*X*Y)
    
    # file ends with an extra byte (a newline)
    if len(data) != 1+img_end:
        msg = f"File size ({len(data)}) doesn't match expected size ({img_end})"
        raise RuntimeError(msg)

    buf = data[img_offset:img_end]
    floats_1d = np.frombuffer(buf, dtype=np.float32)
    floats_3d = floats_1d.reshape((Z, Y, X))
    return floats_3d


def convert_amiramesh_to_h5(path, output_path=None, compress=False):
    """
    Read the given AmiraMesh file, and copy the data to an hdf5 file.
    
    Args:
        path:
            Filepath to the input .am AmiraMesh file.
            Must be in binary-little-endian format.

        output_path:
            Path to write to.  By default, writes
            to the same directory as the input.

        compress:
            If True, enable light gzip compression in the output file.
    """
    print(f"Loading {path}")
    vol = load_amiramesh_binary(path)
    
    if output_path is None:
        output_path = os.path.splitext(path)[0] + '.h5'

    opts = {}
    if compress:
        opts = { 'shuffle': True, 'compress': 'gzip' }

    print(f"Data has shape XY: ({vol.shape[::-1]})")
    print(f"Writing to: {output_path}")    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('volume', data=vol, **opts)


def main():
    from tqdm import tqdm
    from glob import glob
    from pathlib import Path
    
    src_dir = Path('/nrs/flyem/alignment')
    src_paths = glob(f'{src_dir}/Z1217-19m/VNC/Sec*/flatten/*.am')
    dest_dir = Path('/nrs/flyem/tmp/flattening-maps')
    
    for p in map(Path, tqdm(src_paths)):
        dest = dest_dir.joinpath(*p.parts[len(src_dir.parts):]).with_suffix('.h5')
        os.makedirs(dest.parent, exist_ok=True)
        convert_amiramesh_to_h5(str(p), str(dest))


if __name__ == "__main__":
    main()
