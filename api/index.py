import sys
import os

# Add root directory to sys.path so app modules are discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api import app
