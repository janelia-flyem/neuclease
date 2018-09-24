#!/usr/bin/env python3
import neuclease.cleave_server

# main function declared so this file can be used as console_script in setup.py
def main():
    neuclease.cleave_server.main(debug_mode=False)

if __name__ == "__main__":
    main()