import sys
import neuclease.cleave_server

_debug_mode = False
def main():
    ## DEBUG
    if len(sys.argv) == 1:
        import os
        log_dir = os.path.dirname(neuclease.__file__) + '/../logs'
        sys.argv += ["--merge-table", "/tmp/old-table-reviewed-only.npy",
                     "--log-dir", log_dir]
        _debug_mode = True

    neuclease.cleave_server.main(_debug_mode)
    
if __name__ == "__main__":
    main()
