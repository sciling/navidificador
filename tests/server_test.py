import unittest

from bin.server import limit_poem


class TestServer(unittest.TestCase):
    def test_limit_poem(self):
        tests = [
            (
                "\n1\n\n2\n\n3\n4\n5\n6\n\n7\n8\n9\n1\n\nb3\nb4\nb5\nb6\n\nb7\nb8\nb9\nb1\n\nc3\nc4\nc5\nc6\n\nc7\nc8\nc9\nc1\n",
                "3\n4\n5\n6\n\n7\n8\n9\n1\n\nb3\nb4\nb5\nb6\n\nb7\nb8\nb9\nb1",
            ),
            (
                "\n1\n\n2\n\n3\n4\n5\n6",
                "3\n4\n5\n6",
            ),
        ]

        for text, ref in tests:
            res = limit_poem(text)
            self.assertEqual(res, ref)


if __name__ == "__main__":
    unittest.main()
