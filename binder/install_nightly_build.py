import glob
import json
import subprocess
import sys

# look for nightly build download url
release_curl_res = subprocess.run(
    [
        "curl",
        "-L",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
        "https://api.github.com/repos/airbus/scikit-decide/releases/tags/nightly",
    ],
    capture_output=True,
)
release_dict = json.loads(release_curl_res.stdout)
release_download_url = sorted(release_dict["assets"], key=lambda d: d["updated_at"])[
    -1
]["browser_download_url"]

# download and unzip
subprocess.run(["wget", "--output-document=release.zip", release_download_url])
subprocess.run(["unzip", "-o", "release.zip"])

# get proper wheel name according to python version used
wheel_pythonversion_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
wheel_path = glob.glob(f"dist/scikit_decide*{wheel_pythonversion_tag}*manylinux*.whl")[
    0
]

skdecide_pip_spec = f"{wheel_path}[all]"

subprocess.run(["pip", "install", skdecide_pip_spec])
