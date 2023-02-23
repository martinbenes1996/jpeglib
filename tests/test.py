
import sys
import unittest
sys.path.append(".")

# logging
if __name__ == "__main__":
    import logging
    logging.basicConfig(filename="test.log", level=logging.INFO)
    import jpeglib
    logging.info(f"{jpeglib.__path__=}")

# === unit tests ===
from test_cstruct import TestCStruct  # noqa: F401,E402
from test_dct import TestDCT  # noqa: F401,E402
from test_interface import TestInterface  # noqa: F401,E402
from test_notation import TestNotation  # noqa: F401,E402
from test_performance import TestPerformance  # noqa: F401,E402
from test_spatial import TestSpatial  # noqa: F401,E402
from test_version import TestVersion  # noqa: F401,E402

# from test_flags import TestFlags  # noqa: F401,E402
# from test_marker import TestMarker  # noqa: F401,E402
# from test_progressive import TestProgressive  # noqa: F401,E402
# ==================

# run unittests
if __name__ == "__main__":
    unittest.main(verbosity=1)
