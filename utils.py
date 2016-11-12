import contextlib
import sys
from io import StringIO


@contextlib.contextmanager
def no_stdout():
    save_stdout = sys.stdout
    sys.stdout = StringIO()
    yield
    sys.stdout = save_stdout
