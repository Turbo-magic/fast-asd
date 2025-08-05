#!/usr/bin/env python3
"""
Convenience script to run local active speaker detection
Automatically activates virtual environment
"""

import sys
import os
import subprocess

def main():
    # Check if virtual environment exists
    venv_path = os.path.join(os.path.dirname(__file__), 'venv')
    if not os.path.exists(venv_path):
        print("❌ Virtual environment not found!")
        print("Please run: ./setup_local.sh first")
        sys.exit(1)
    
    # Get Python executable from virtual environment
    venv_python = os.path.join(venv_path, 'bin', 'python')
    if not os.path.exists(venv_python):
        print("❌ Python not found in virtual environment!")
        print("Please run: ./setup_local.sh to reinstall")
        sys.exit(1)
    
    # Run local_main.py with all arguments passed through
    cmd = [venv_python, 'local_main.py'] + sys.argv[1:]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running local_main.py: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()