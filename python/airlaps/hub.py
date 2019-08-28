# Adapted from https://github.com/pytorch/pytorch/blob/master/torch/hub.py
from __future__ import annotations

import os
import inspect
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen


try:
    from tqdm import tqdm
except ImportError:
    # fake tqdm if it's not installed
    class tqdm(object):

        def __init__(self, total=None, disable=False, unit=None, unit_scale=None, unit_divisor=None):
            self.total = total
            self.disable = disable
            self.n = 0
            # ignore unit, unit_scale, unit_divisor; they're just for real tqdm

        def update(self, n):
            global verbosity

            if self.disable:
                return

            self.n += n
            if verbosity > 0:
                if self.total is None:
                    sys.stderr.write("\r{0:.1f} bytes".format(self.n))
                else:
                    sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
                sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            global verbosity

            if self.disable:
                return

            if verbosity > 0:
                sys.stderr.write('\n')

MASTER_BRANCH = 'master'
ENV_AIRLAPS_HOME = 'AIRLAPS_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
MODULE_HUBCONF = 'hubconf.py'
READ_DATA_CHUNK = 8192
hub_dir = None
verbosity = 1  # level between 0 (silent) and 2 (very verbose)


def _import_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _git_archive_link(repo_owner, repo_name, branch):
    return 'https://github.com/{}/{}/archive/{}.zip'.format(repo_owner, repo_name, branch)


def _download_archive_zip(url, filename):
    global verbosity

    if verbosity > 0:
        sys.stderr.write('Downloading: \"{}\" to {}\n'.format(url, filename))

    response = urlopen(url)
    with open(filename, 'wb') as f:
        while True:
            data = response.read(READ_DATA_CHUNK)
            if len(data) == 0:
                break
            f.write(data)


def _load_attr_from_module(module, func_name):
    # Check if callable is defined in the module
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)


def _get_airlaps_home():
    airlaps_home = os.path.expanduser(
        os.getenv(ENV_AIRLAPS_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'AIRLAPS')))
    return airlaps_home


def _setup_hubdir():
    global hub_dir

    if hub_dir is None:
        airlaps_home = _get_airlaps_home()
        hub_dir = os.path.join(airlaps_home, 'hub')

    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)


def _parse_repo_info(github):
    branch = MASTER_BRANCH
    if ':' in github:
        repo_info, branch = github.split(':')
    else:
        repo_info = github
    repo_owner, repo_name = repo_info.split('/')
    return repo_owner, repo_name, branch


def _get_cache_or_reload(github, force_reload):
    # Parse github repo information
    repo_owner, repo_name, branch = _parse_repo_info(github)

    # Github renames folder repo-v1.x.x to repo-1.x.x
    # We don't know the repo name before downloading the zip file
    # and inspect name from it.
    # To check if cached repo exists, we need to normalize folder names.
    repo_dir = os.path.join(hub_dir, '__'.join([repo_owner, repo_name, branch]))

    use_cache = (not force_reload) and os.path.exists(repo_dir)

    if use_cache:
        if verbosity > 1:
            sys.stderr.write('Using cache found in {}\n'.format(repo_dir))
    else:
        cached_file = os.path.join(hub_dir, branch + '.zip')
        _remove_if_exists(cached_file)

        url = _git_archive_link(repo_owner, repo_name, branch)
        _download_archive_zip(url, cached_file)

        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            # Unzip the code and rename the base folder
            cached_zipfile.extractall(hub_dir)

        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)  # rename the repo

    return repo_dir


def _load_entry_from_hubconf(m, entry):
    if not isinstance(entry, str):
        raise ValueError('Invalid input: entry should be a string of callable name')

    func = _load_attr_from_module(m, entry)

    if func is None or not callable(func):
        raise RuntimeError('Cannot find {} in hubconf'.format(entry))

    return func


def _import_local_hub_module(path: str):
    sys.path.insert(0, path)
    hub_module = _import_module(MODULE_HUBCONF, path + '/' + MODULE_HUBCONF)
    sys.path.remove(path)
    return hub_module


def _import_hub_module(github: str, folder: str, force_reload: bool):
    # Setup hub_dir to save downloaded files
    _setup_hubdir()

    repo_dir = os.path.join(_get_cache_or_reload(github, force_reload), *folder.split('/'))

    hub_module = _import_local_hub_module(repo_dir)

    return hub_module



def set_dir(d):
    """Optionally set hub_dir to a local dir to save downloaded repositories.

    If ``set_dir`` is not called, default path is ``$AIRLAPS_HOME/hub`` where
    environment variable ``$AIRLAPS_HOME`` defaults to ``$XDG_CACHE_HOME/AIRLAPS``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if the environment
    variable is not set.

    # Parameters
    d: path to a local folder to save downloaded repositories.
    """
    global hub_dir
    hub_dir = d


def set_verbosity(v: int):
    """Optionally set verbosity to a desired level.

    If ``set_verbosity`` is not called, default verbosity is 1 (verbose).

    # Parameters
    v: verbosity level between 0 (silent) and 2 (very verbose).
    """
    global verbosity
    verbosity = v


def list(github='Airbus-AI-Research/AIRLAPS', folder='', force_reload=False):
    """List all entrypoints available in a `github` hubconf.

    # Parameters
    github: Optional, a string with format "repo_owner/repo_name[:tag_name]" with an optional
        tag/branch. The default branch is `master` if not specified. Default is `Airbus-AI-Research/AIRLAPS`.
        Example: 'Airbus-AI-Research/AIRLAPS[:hub]'
    folder: Optional, the repo folder containing the hubconf to consider. Default is '' (root).
        Example: 'hub/solver/lazy_astar'
    force_reload: Optional, whether to discard the existing cache and force a fresh download.
        Default is `False`.

    # Returns
    A list of available entrypoint names.

    # Example
    ```python
    entrypoints = airlaps.hub.list(folder='hub/domain/gym', force_reload=True)
    ```
    """
    hub_module = _import_hub_module(github, folder, force_reload)

    # We take functions starting with '_' as internal helper functions
    entrypoints = [f for f in dir(hub_module) if callable(getattr(hub_module, f)) and not f.startswith('_')]

    return entrypoints


def help(entry, github='Airbus-AI-Research/AIRLAPS', folder='', force_reload=False):
    """Show the docstring of entrypoint `entry`.

    # Parameters
    entry: Required, a string of entrypoint name defined in hubconf.py.
    github: Optional, a string with format "repo_owner/repo_name[:tag_name]" with an optional
        tag/branch. The default branch is `master` if not specified. Default is `Airbus-AI-Research/AIRLAPS`.
        Example: 'Airbus-AI-Research/AIRLAPS[:hub]'
    folder: Optional, the repo folder containing the hubconf to consider. Default is '' (root).
        Example: 'hub/solver/lazy_astar'
    force_reload: Optional, whether to discard the existing cache and force a fresh download.
        Default is `False`.

    # Example
    ```python
    print(airlaps.hub.help('GymDomain', folder='hub/domain/gym', force_reload=True))
    ```
    """
    hub_module = _import_hub_module(github, folder, force_reload)

    loaded_entry = _load_entry_from_hubconf(hub_module, entry)

    return loaded_entry.__doc__


def load(entry, github='Airbus-AI-Research/AIRLAPS', folder='', force_reload=False):
    """Load an entry from a github repo, such as a Domain, Solver or Space class.

    # Parameters
    entry: Required, a string of entrypoint name defined in hubconf.py.
    github: Optional, a string with format "repo_owner/repo_name[:tag_name]" with an optional
        tag/branch. The default branch is `master` if not specified. Default is `Airbus-AI-Research/AIRLAPS`.
        Example: 'Airbus-AI-Research/AIRLAPS[:hub]'
    folder: Optional, the repo folder containing the hubconf to consider. Default is '' (root).
        Example: 'hub/solver/lazy_astar'
    force_reload: Optional, whether to force a fresh download of github repo unconditionally.
        Default is `False`.

    # Returns
    A single entry.

    # Example
    ```python
    GymDomain = airlaps.hub.load('GymDomain', folder='hub/domain/gym', force_reload=True)
    ```
    """
    hub_module = _import_hub_module(github, folder, force_reload)

    loaded_entry = _load_entry_from_hubconf(hub_module, entry)

    return loaded_entry


def local_search(obj_type):
    """Search for all entries downloaded from hub of a certain type, such as Domain, Solver or Space class.

    # Parameters
    obj_type: Required, the object type to look for.

    # Returns
    A list of matching entries.

    # Example
    ```python
    downloaded_domains = airlaps.hub.local_search(Domain)
    ```
    """
    global hub_dir

    _setup_hubdir()
    results = []
    for file in Path(hub_dir).glob('**/hubconf.py'):
        hub_module = _import_local_hub_module(str(file.parent))

        # We take functions starting with '_' as internal helper functions
        entryclasses = [f for f in dir(hub_module) if inspect.isclass(getattr(hub_module, f)) and not f.startswith('_')]

        results += [getattr(hub_module, e) for e in entryclasses if issubclass(getattr(hub_module, e), obj_type)]

    return results

# TODO: add 'install' & 'uninstall' functions to add content from hub (without loading necessarily)
