# Based on https://stackoverflow.com/a/3620972
import time
import logging
from functools import wraps

PROF_DATA = {}

logger = logging.getLogger(__name__)


def profile(desc=None):
    def profile_wrap(function):
        @wraps(function)
        def with_profiling(*args, **kwargs):
            name = function.__name__
            if isinstance(desc, int) and len(args) > desc:
                name = name + ':' + args[desc]
            elif isinstance(desc, str):
                if desc in kwargs:
                    name = name + ':' + kwargs['desc']
                else:
                    name = name + ':' + desc

            logger.info(f"Starting profiling of {name}")

            start_time = time.time()
            ret = function(*args, **kwargs)
            elapsed_time = time.time() - start_time

            logger.info(f"Ending profiling of {name} ({elapsed_time}s)")

            if name not in PROF_DATA:
                PROF_DATA[name] = [0, []]
            PROF_DATA[name][0] += 1
            PROF_DATA[name][1].append(elapsed_time)

            return ret

        return with_profiling
    return profile_wrap


def get_profiling_data():
    prof = {}
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])

        prof[fname] = {
            'num_calls': data[0],
            'max_time': max_time,
            'avg_time': avg_time,
        }

    return prof


def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        logger.info("Function %s called %d times. ", fname, data[0])
        logger.info('Execution time max: %.3f, average: %.3f', max_time, avg_time)


def clear_prof_data():
    PROF_DATA.clear()
