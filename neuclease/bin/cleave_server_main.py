import sys
import neuclease.cleave_server

if __name__ == "__main__":
    ## DEBUG
    debug_mode = False
    if len(sys.argv) == 1:
        import os
        log_dir = os.path.dirname(neuclease.__file__) + '/../logs'
        sys.argv += ["--merge-table", "/tmp/old-table-reviewed-only.npy",
                     "--log-dir", log_dir]
        debug_mode = True

    neuclease.cleave_server.main(debug_mode)
