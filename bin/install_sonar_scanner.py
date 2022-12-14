#! /usr/bin/env python

import os
import stat

from argparse import ArgumentParser
from io import BytesIO
from urllib.request import Request
from urllib.request import urlopen
from zipfile import ZipFile


parser = ArgumentParser()
parser.add_argument("-e", "--venv", help="Direction of poetry's virtual environment", default=".venv")
args = parser.parse_args()

venv = args.venv

SONAR_SCANNER_VERSION = "4.7.0.2747"
zip_fn = f"sonar-scanner-cli-{SONAR_SCANNER_VERSION}-linux.zip"
sonar_scanner_url = f"https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/{zip_fn}"

print(f"Installing sonar scanner version {SONAR_SCANNER_VERSION} in '{venv}'")

if os.path.isfile(zip_fn):
    archive = ZipFile(zip_fn)
else:
    # This audits open urls to prevent bandit B310 problems.
    # https://stackoverflow.com/a/53040523
    if sonar_scanner_url.lower().startswith("http"):
        req = Request(sonar_scanner_url)
    else:
        raise ValueError from None

    with urlopen(req) as resp:  # nosec B310
        archive = ZipFile(BytesIO(resp.read()))

for file in archive.filelist:
    dest = file.filename.split(os.path.dirname(os.sep))
    dest[0] = ""
    file.filename = os.sep.join(dest)
    archive.extract(file, path=venv)

for fn in (f"{venv}/bin/sonar-scanner", f"{venv}/bin/sonar-scanner-debug", f"{venv}/jre/bin/java"):
    st = os.stat(fn)
    os.chmod(fn, st.st_mode | stat.S_IEXEC)
