import unittest

from bin.server import limit_poem


class TestServer(unittest.TestCase):
    def test_limit_poem(self):
        text = "\n1\n\n2\n\n3\n4\n5\n6\n\n7\n8\n9\n1\n"
        res = limit_poem(text)
        self.assertEqual(res, "3\n4\n5\n6\n\n7\n8\n9\n1")


if __name__ == "__main__":
    unittest.main()
