# hub

[[toc]]

## set\_dir

<airlaps-signature name= "set_dir" :sig="{'params': [{'name': 'd'}]}"></airlaps-signature>

Optionally set hub_dir to a local dir to save downloaded repositories.

If ``set_dir`` is not called, default path is ``$AIRLAPS_HOME/hub`` where
environment variable ``$AIRLAPS_HOME`` defaults to ``$XDG_CACHE_HOME/AIRLAPS``.
``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
filesytem layout, with a default value ``~/.cache`` if the environment
variable is not set.

#### Parameters
- **d**: path to a local folder to save downloaded repositories.

## set\_verbosity

<airlaps-signature name= "set_verbosity" :sig="{'params': [{'name': 'v', 'annotation': 'int'}]}"></airlaps-signature>

Optionally set verbosity to a desired level.

If ``set_verbosity`` is not called, default verbosity is 1 (verbose).

#### Parameters
- **v**: verbosity level between 0 (silent) and 2 (very verbose).

## list

<airlaps-signature name= "list" :sig="{'params': [{'name': 'github', 'default': 'Airbus-AI-Research/AIRLAPS'}, {'name': 'folder', 'default': ''}, {'name': 'force_reload', 'default': 'False'}]}"></airlaps-signature>

List all entrypoints available in a `github` hubconf.

#### Parameters
- **github**: Optional, a string with format "repo_owner/repo_name[:tag_name]" with an optional
    tag/branch. The default branch is `master` if not specified. Default is `Airbus-AI-Research/AIRLAPS`.
    Example: 'Airbus-AI-Research/AIRLAPS[:hub]'
- **folder**: Optional, the repo folder containing the hubconf to consider. Default is '' (root).
    Example: 'hub/solver/lazy_astar'
- **force_reload**: Optional, whether to discard the existing cache and force a fresh download.
    Default is `False`.

#### Returns
A list of available entrypoint names.

#### Example
```python
entrypoints = airlaps.hub.list(folder='hub/domain/gym', force_reload=True)
```

## help

<airlaps-signature name= "help" :sig="{'params': [{'name': 'entry'}, {'name': 'github', 'default': 'Airbus-AI-Research/AIRLAPS'}, {'name': 'folder', 'default': ''}, {'name': 'force_reload', 'default': 'False'}]}"></airlaps-signature>

Show the docstring of entrypoint `entry`.

#### Parameters
- **entry**: Required, a string of entrypoint name defined in hubconf.py.
- **github**: Optional, a string with format "repo_owner/repo_name[:tag_name]" with an optional
    tag/branch. The default branch is `master` if not specified. Default is `Airbus-AI-Research/AIRLAPS`.
    Example: 'Airbus-AI-Research/AIRLAPS[:hub]'
- **folder**: Optional, the repo folder containing the hubconf to consider. Default is '' (root).
    Example: 'hub/solver/lazy_astar'
- **force_reload**: Optional, whether to discard the existing cache and force a fresh download.
    Default is `False`.

#### Example
```python
print(airlaps.hub.help('GymDomain', folder='hub/domain/gym', force_reload=True))
```

## load

<airlaps-signature name= "load" :sig="{'params': [{'name': 'entry'}, {'name': 'github', 'default': 'Airbus-AI-Research/AIRLAPS'}, {'name': 'folder', 'default': ''}, {'name': 'force_reload', 'default': 'False'}]}"></airlaps-signature>

Load an entry from a github repo, such as a Domain, Solver or Space class.

#### Parameters
- **entry**: Required, a string of entrypoint name defined in hubconf.py.
- **github**: Optional, a string with format "repo_owner/repo_name[:tag_name]" with an optional
    tag/branch. The default branch is `master` if not specified. Default is `Airbus-AI-Research/AIRLAPS`.
    Example: 'Airbus-AI-Research/AIRLAPS[:hub]'
- **folder**: Optional, the repo folder containing the hubconf to consider. Default is '' (root).
    Example: 'hub/solver/lazy_astar'
- **force_reload**: Optional, whether to force a fresh download of github repo unconditionally.
    Default is `False`.

#### Returns
A single entry.

#### Example
```python
GymDomain = airlaps.hub.load('GymDomain', folder='hub/domain/gym', force_reload=True)
```

## local\_search

<airlaps-signature name= "local_search" :sig="{'params': [{'name': 'obj_type'}]}"></airlaps-signature>

Search for all entries downloaded from hub of a certain type, such as Domain, Solver or Space class.

#### Parameters
- **obj_type**: Required, the object type to look for.

#### Returns
A list of matching entries.

#### Example
```python
downloaded_domains = airlaps.hub.local_search(Domain)
```

