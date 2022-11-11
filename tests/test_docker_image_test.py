import unittest

from bin.test_docker_image import main


class TestBin(unittest.TestCase):
    def test_docker_image_bin(self):
        main()


if __name__ == "__main__":
    unittest.main()
