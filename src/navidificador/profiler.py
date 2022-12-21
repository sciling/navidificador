# Based on https://stackoverflow.com/a/3620972
import inspect
import logging
import time

from functools import wraps


PROF_DATA = {}

logger = logging.getLogger(__name__)


def profile(desc=None):
    def profile_wrap(function):
        def start_profiling(*args, **kwargs):
            name = function.__name__
            if isinstance(desc, int) and len(args) > desc:
                name = name + ":" + args[desc]
            elif isinstance(desc, str):
                if desc in kwargs:
                    name = name + ":" + kwargs["desc"]
                else:
                    name = name + ":" + desc

            logger.debug(f"Starting profiling of {name}")

            start_time = time.time()
            return name, start_time

        def end_profiling(name, start_time):
            elapsed_time = time.time() - start_time

            logger.debug(f"Ending profiling of {name} ({elapsed_time}s)")

            if name not in PROF_DATA:
                PROF_DATA[name] = [0, []]
            PROF_DATA[name][0] += 1
            PROF_DATA[name][1].append(elapsed_time)

        @wraps(function)
        async def with_profiling_async(*args, **kwargs):
            name, start_time = start_profiling(*args, **kwargs)

            try:
                ret = await function(*args, **kwargs)
                logger.debug(f"with_profiling:async:{name}:{type(ret)}")
            except Exception as err:
                logger.debug(f"Ending profiling of {name} with error. Reason: {err}")
                raise err

            end_profiling(name, start_time)
            return ret

        @wraps(function)
        def with_profiling_sync(*args, **kwargs):
            name, start_time = start_profiling(*args, **kwargs)

            try:
                ret = function(*args, **kwargs)
                logger.debug(f"with_profiling:sync:{name}:{type(ret)}")
            except Exception as err:
                logger.debug(f"Ending profiling of {name} with error. Reason: {err}")
                raise err

            end_profiling(name, start_time)
            return ret

        logger.debug(f"PROFILER: {inspect.iscoroutinefunction(function)} {function}")
        if inspect.iscoroutinefunction(function):
            return with_profiling_async
        return with_profiling_sync

    return profile_wrap


def get_profiling_data():
    prof = {}
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])

        prof[fname] = {
            "num_calls": data[0],
            "max_time": max_time,
            "avg_time": avg_time,
        }

    return prof


def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        logger.info("Function %s called %d times. ", fname, data[0])
        logger.info("Execution time max: %.3f, average: %.3f", max_time, avg_time)


def clear_prof_data():
    PROF_DATA.clear()
