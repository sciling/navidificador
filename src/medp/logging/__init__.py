import datetime
import json
import logging
import os
import sys
import traceback

from logging import config

import sentry_sdk


getLogger = logging.getLogger  # noqa: N816

sentry_tags = (
    "env",
    "branch",
    "commit_id",
    "version",
    "device",
    "module",
    "workflow_id",
    "workflow_name",
    "codebuild_batch_build_identifier",
    "codebuild_build_arn",
    "codebuild_build_id",
    "codebuild_build_image",
    "codebuild_build_number",
    "codebuild_initiator",
    "codebuild_kms_key_id",
    "codebuild_log_path",
    "codebuild_public_build_url",
    "codebuild_resolved_source_version",
    "codebuild_source_repo_url",
    "codebuild_source_version",
    "codebuild_start_time",
    "codebuild_webhook_actor_account_id",
    "codebuild_webhook_base_ref",
    "codebuild_webhook_event",
    "codebuild_webhook_merge_commit",
    "codebuild_webhook_prev_commit",
    "codebuild_webhook_head_ref",
    "codebuild_webhook_trigger",
)


class ConsoleFormatter(logging.Formatter):
    # Use an appropriate date format for fluentd:
    # https://stackoverflow.com/questions/59220299/fluentd-time-field-handling-in-json-log-records.
    # Based on:
    # https://stackoverflow.com/a/6290946
    def formatTime(self, record, datefmt=None):  # noqa: N802
        if not datefmt:
            # Iso format compatible with fluentd current configuration
            datefmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        created_at = datetime.datetime.fromtimestamp(record.created)
        return created_at.strftime(datefmt)

    def formatException(self, ei):  # noqa: N802
        if ei:
            result = super().formatException(ei)
            return result  # or format into one line however you want to
        return None

    def format(self, record):
        if getattr(record, "alarm", False):
            record.__dict__["alarm"] = "*"
        else:
            record.__dict__["alarm"] = ""

        return super().format(record)


class FluentdFormatter(ConsoleFormatter):
    def format(self, record):
        try:
            log = {
                "time": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "caller": f"{record.pathname}:{record.lineno}",
                "pythonModule": record.module,
                "stack": self.formatException(record.exc_info),
                # 'asctime': record.created,
                # 'funcName': record.funcName,  # info, debug, exception... not really useful, similar to level.
                # 'msecs': record.msecs,
                # 'pathname': record.pathname,
                # 'filename': record.filename,
                # 'lineno': record.lineno,
                "threadName": record.threadName,
                "thread": record.thread,
                "processName": record.processName,
                "process": record.process,
                "levelno": record.levelno,
            }

            if getattr(record, "alarm", False):
                log["alarm"] = True

            # Force one line output
            return json.dumps(log, indent=None, ensure_ascii=False)
        except Exception:  # pylint: disable=broad-except
            # Make sure that logging does not break the app.
            # Instead, print the stacktrace
            traceback.print_exc()
            # and call the default formatter.
            return super().format(record)


def handle_exception(exc_type, exc_value, exc_traceback):
    logger = logging.getLogger(__name__)
    logger.critical(
        "Uncaught exception: %s: %s",
        exc_type.__name__,
        exc_value,
        exc_info=(exc_type, exc_value, exc_traceback),
    )


default_logging_packages = json.dumps(
    {
        "CRITICAL": ["boto3", "botocore", "protego", "readability", "scrapy", "snowflake", "urllib3"],
    }
)


def configure_logging():
    # We define the class AlarmLogger so that we can reset
    # the logging system and reconfigure it in the same way it
    # is done in the tests. Also: https://stackoverflow.com/a/12034791
    # This should not pose a security issue since
    # this logger only adds a fixed-value variable
    class AlarmLogger(logging.Logger):  # NOSONAR
        def alert(self, *args, **kwargs):
            extra = kwargs.pop("extra", {})
            extra["alarm"] = True
            return super().warning(*args, extra=extra, **kwargs)

    dirname = os.path.dirname(os.path.realpath(__file__))
    # This should not pose a security issue since
    # this logger only adds a fixed-value variable
    logging.setLoggerClass(AlarmLogger)  # NOSONAR

    logging_level_packages = json.loads(os.getenv("LOGGING_IGNORE_PACKAGES", default_logging_packages))
    for level, packages in logging_level_packages.items():
        for package in packages:
            if level.upper() == "DISABLE":
                logging.getLogger(package).propagate = False
            else:
                logging.getLogger(package).setLevel(level)

    # Configure the logging library in a general way.
    if os.getenv("KUBERNETES_PORT", None):
        config.fileConfig(os.path.join(dirname, "./kubernetes_logging_config.ini"))  # nosec # NOSONAR python:S4792
    else:
        config.fileConfig(os.path.join(dirname, "./logging_config.ini"))  # nosec # NOSONAR python:S4792

    logger = logging.getLogger(__name__)
    logger.info("Configured logging for medp common lib ...")

    sys.excepthook = handle_exception

    try:
        init_sentry(logger)

    except Exception:  # pylint: disable=broad-except
        logger.exception("Sentry could not be initialized")
        return False
    return True


def init_sentry(logger):
    if "SENTRY_ENVIRONMENT" not in os.environ:
        if "pytest" in sys.modules:
            sentry_env = "pytest"
        else:
            sentry_env = "dev"  # pragma: no cover

        os.environ["SENTRY_ENVIRONMENT"] = sentry_env

    ignore_error_list = getattr(sentry_sdk, "ignore_error_list", [])
    if "KeyboardInterrupt" not in ignore_error_list:
        ignore_error_list.append("KeyboardInterrupt")
    logger.info(f"Sentry ignoring: {ignore_error_list}")
    sentry_sdk.init(
        os.getenv("SENTRY_SECRET"),
        release=os.getenv("IMAGE_TAG"),
        ignore_errors=ignore_error_list,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
    )

    for tag in sentry_tags:
        if os.getenv(tag.upper()):
            logger.info(f"Setting sentry tag {tag} {os.getenv(tag.upper())}")
            sentry_sdk.set_tag(tag, os.getenv(tag.upper()))

    logger.info(f"Starting sentry with env tag '{os.getenv('SENTRY_ENVIRONMENT', None)}'")
