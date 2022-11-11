import logging
import os
import sys
import unittest

from importlib import reload
from unittest import mock

import medp


class TestExample(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MODULE"] = "pytest"

        return super().setUp()

    def test_terminal(self):
        logging.shutdown()
        reload(logging)
        self.assertTrue(medp.logging.configure_logging())

        with self.assertLogs("foo") as context:
            logger = medp.logging.getLogger("foo")
            logger.info("test terminal")
            logger.alert("alert terminal")

            self.assertEqual(context.output, ["INFO:foo:test terminal", "WARNING:foo:alert terminal"])

            try:
                raise ValueError()
            except ValueError:
                logger.exception("exception terminal")

            self.assertRegex(context.output[2].replace("\n", " "), "ERROR:foo:exception terminal.*ValueError")

            try:
                raise ValueError()
            except ValueError as err:
                logger.critical(err, exc_info=True)

            self.assertRegex(context.output[3].replace("\n", " "), "CRITICAL:foo: Traceback.*ValueError")

    @mock.patch.dict(os.environ, {"KUBERNETES_PORT": "80"})
    def test_kubernetes(self):
        logging.shutdown()
        reload(logging)
        self.assertTrue(medp.logging.configure_logging())

        with self.assertLogs("foo") as context:
            logger = medp.logging.getLogger("foo")
            logger.info("test kubernetes")
            logger.alert("alert kubernetes")

            self.assertEqual(context.output, ["INFO:foo:test kubernetes", "WARNING:foo:alert kubernetes"])

            try:
                raise ValueError()
            except ValueError:
                logger.exception("exception kubernetes")

            self.assertRegex(context.output[2].replace("\n", " "), "ERROR:foo:exception kubernetes.*ValueError")

    @mock.patch.dict(os.environ, {"LOGGING_IGNORE_PACKAGES": '{"DISABLE": ["module_name"]}'})
    def test_disable(self):
        logging.shutdown()
        reload(logging)
        self.assertTrue(medp.logging.configure_logging())

        try:
            with self.assertLogs("module_name") as context:
                logger = medp.logging.getLogger("module_name")
                logger.info("test")

                self.assertEqual(context.output, [])
        except AssertionError:
            pass
        else:
            raise AssertionError("Logs should have been suppressed")

    @mock.patch.dict(os.environ, {"SENTRY_SECRET": "bad sentry secret (Unsupported scheme)"})  # pragma: allowlist secret
    def test_sentry(self):
        logging.shutdown()
        reload(logging)
        self.assertFalse(medp.logging.configure_logging())

    def test_handle_exception(self):
        with self.assertLogs("medp.logging") as context:
            try:
                raise ValueError
            except ValueError:
                medp.logging.handle_exception(*sys.exc_info())

            self.assertRegex(context.output[0].replace("\n", " "), "CRITICAL:.*ValueError")


if __name__ == "__main__":
    unittest.main()
