# Put this in: C:\Users\bbrel\agentic\tests\__init__.py
import os
import sys

# Ensure the parent directory (project root) is in the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
