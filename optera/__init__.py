import logging

# NullHandler is the library-level default so importing optera never produces
# spurious "no handler" warnings.  The application entry point (run.py) sets up
# the actual handler and level.
logging.getLogger(__name__).addHandler(logging.NullHandler())
