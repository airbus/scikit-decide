# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import ast
import importlib
import inspect
import json
import logging
import os
import pkgutil
import re
import sys
import urllib
from enum import Enum
from functools import lru_cache
from glob import glob
from typing import Any, List, Tuple

import skdecide

NOTEBOOKS_LIST_PLACEHOLDER = "[[notebooks-list]]"
NOTEBOOKS_PAGE_TEMPLATE_RELATIVE_PATH = "notebooks/README.template.md"
NOTEBOOKS_PAGE_RELATIVE_PATH = "notebooks/README.md"
NOTEBOOKS_SECTION_KEY_VAR_SEP = "_"
NOTEBOOKS_DIRECTORY_NAME = "notebooks"

DEFAULT_REPO_NAME = "airbus/scikit-decide"

docdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(f"{docdir}/..")
notebooksdir = f"{rootdir}/{NOTEBOOKS_DIRECTORY_NAME}"

logger = logging.getLogger(__name__)
refs = set()


# https://stackoverflow.com/questions/48879353/how-do-you-recursively-get-all-submodules-in-a-python-package
def find_abs_modules(package):
    path_list = []
    spec_list = []
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__):
        if modname == "hub.__skdecide_hub_cpp":
            continue
        import_path = f"{package.__name__}.{modname}"
        if ispkg:
            spec = pkgutil._get_spec(importer, modname)
            try:
                importlib._bootstrap._load(spec)
                spec_list.append(spec)
            except Exception as e:
                print(f"Could not load package {modname}, so it will be ignored ({e}).")
        else:
            path_list.append(import_path)
    for spec in spec_list:
        del sys.modules[spec.name]
    return path_list


def py_parse(filepath):  # avoid using ast package just for extracting file doc string?
    with open(filepath) as fd:
        file_contents = fd.read()
    module = ast.parse(file_contents)
    docstring = ast.get_docstring(module)
    docstring = "" if docstring is None else docstring.strip()
    name = os.path.splitext(os.path.basename(filepath))[0]
    return docstring, name, file_contents


@lru_cache(maxsize=1000)
def get_ref(object):
    name = getattr(object, "__qualname__", None)
    if name is None:
        name = getattr(object, "__name__", None)
        if isinstance(object, Enum):
            name = getattr(object, "name", None)
        if name is None:
            name = object._name
    reflist = [name]
    if hasattr(object, "__module__"):
        reflist.insert(0, object.__module__)
    ref = ".".join(reflist)
    refs.add(ref)
    return ref


def format_doc(doc):

    # Replace references like "#obj.func()" by "`obj.func()`" for Markdown code highlighting - TODO: replace in doc?
    doc = re.sub(r"#(?P<ref>[\w\.,()]*[\w()])", lambda m: f'`{m.group("ref")}`', doc)

    # Replace content of "# Parameters" by list of parameters
    def list_content(content):
        content = re.sub(
            r"^(?P<param>\w+)",
            lambda m: f'- **{m.group("param")}**',
            content,
            flags=re.MULTILINE,
        )
        return content.strip()

    doc = re.sub(
        r"^(?<=# Parameters\n)(?P<content>(?:\n?\s*\w.*)+)",
        lambda m: list_content(m.group("content")),
        doc,
        flags=re.MULTILINE,
    )

    doc = re.sub(
        r"^(?<=# Attributes\n)(?P<content>(?:\n?\s*\w.*)+)",
        lambda m: list_content(m.group("content")),
        doc,
        flags=re.MULTILINE,
    )

    # Replace "# Title" (e.g. "# Parameters") by "#### Title"
    doc = re.sub(
        r"^# (?=Parameters|Attributes|Returns|Example)",
        "#### ",
        doc,
        flags=re.MULTILINE,
    )

    # Replace "!!! container" (e.g. "!!! tip") by "::: container [...] :::"
    def strip_content(content):
        content = re.sub(r"^ {4}", "", content, flags=re.MULTILINE)
        return content.rstrip()

    doc = re.sub(
        r"!!! (?=tip|warning|danger)(?P<head>.*)\n(?P<content>(?:\n? {4,}.*)+)",
        lambda m: f'::: {m.group("head")}\n{strip_content(m.group("content"))}\n:::',
        doc,
    )

    return doc


def add_func_method_infos(func_method, autodoc):
    if inspect.isfunction(func_method):
        autodoc["type"] = "function"
    elif inspect.ismethod(func_method):
        autodoc["type"] = "method"

    # Get signature
    signature = inspect.signature(func_method)
    parameters = signature.parameters
    params = []
    for k, v in parameters.items():
        if not (k == "self" and func_method.__name__ == "__init__"):
            parameter = parameters[k]
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                param = {"name": "*" + k}
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                param = {"name": "**" + k}
            else:
                param = {"name": k}
            if parameter.default != signature.empty:
                param["default"] = str(parameter.default)
                if "lambda" in param["default"]:
                    param["default"] = "<lambda function>"  # TODO: improve?
            if parameter.annotation != signature.empty:
                param["annotation"] = parameter.annotation
            params.append(param)
    autodoc["signature"] = {"params": params}
    if signature.return_annotation != signature.empty:
        autodoc["signature"]["return"] = signature.return_annotation


def add_basic_member_infos(member, autodoc):
    try:
        autodoc["ref"] = get_ref(member)
        if isinstance(member, Enum):
            doc = inspect.getdoc(member)
            if doc == inspect.getdoc(type(member)):
                # same doc as the enum class (if enum member doc have not been specified): do not repeat it
                doc = ""
        else:
            source, line = inspect.getsourcelines(member)
            autodoc["source"] = "".join(source)
            autodoc["line"] = line
            doc = inspect.getdoc(member)
        if doc is not None:
            autodoc["doc"] = format_doc(doc)
    except Exception:  # can happen e.g. when member is TypeVar
        pass


class AnnotationEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        return str(o)


def json_escape(obj):
    return json.dumps(obj, cls=AnnotationEncoder).replace("'", r"\'").replace('"', "'")


def md_escape(md):
    return re.sub(r"[_<>]", lambda m: f"\\{m.group()}", md)


def doc_escape(md):
    return re.sub(r"[<]", lambda m: f"\\{m.group()}", md)


def write_signature(md, member):
    if "signature" in member:
        escape_json_sig = json_escape(member["signature"])
        md += f'<skdecide-signature name= "{member["name"]}" :sig="{escape_json_sig}"></skdecide-signature>\n\n'
    return md


def is_implemented(func_code):
    return not func_code.strip().endswith("raise NotImplementedError")


def get_binder_link(
    binder_env_repo_name: str,
    binder_env_branch: str,
    notebooks_repo_url: str,
    notebooks_branch: str,
    notebook_relative_path: str,
    notebooks_repo_name: str,
    use_nbgitpuller: bool = False,
) -> str:
    # binder hub url
    jupyterhub = urllib.parse.urlsplit("https://mybinder.org")

    if use_nbgitpuller:
        # path to the binder env
        binder_path = f"v2/gh/{binder_env_repo_name}/{binder_env_branch}"

        # nbgitpuller query
        notebooks_repo_basename = os.path.basename(notebooks_repo_url)
        urlpath = f"tree/{notebooks_repo_basename}/{notebook_relative_path}"
        next_url_params = urllib.parse.urlencode(
            {
                "repo": notebooks_repo_url,
                "urlpath": urlpath,
                "branch": notebooks_branch,
            }
        )
        next_url = f"git-pull?{next_url_params}"
        query = urllib.parse.urlencode({"urlpath": next_url})

        # full link
        link = urllib.parse.urlunsplit(
            urllib.parse.SplitResult(
                scheme=jupyterhub.scheme,
                netloc=jupyterhub.netloc,
                path=binder_path,
                query=query,
                fragment="",
            )
        )
    else:
        if notebooks_repo_name:
            # path to the binder env
            binder_path = f"v2/gh/{notebooks_repo_name}/{notebooks_branch}"

            # query to open proper notebook
            query = urllib.parse.urlencode({"labpath": notebook_relative_path})

            # full link
            link = urllib.parse.urlunsplit(
                urllib.parse.SplitResult(
                    scheme=jupyterhub.scheme,
                    netloc=jupyterhub.netloc,
                    path=binder_path,
                    query=query,
                    fragment="",
                )
            )
        else:
            link = ""

    return link


def get_colab_link(
    notebooks_repo_name: str,
    notebooks_branch: str,
    notebook_relative_path: str,
) -> str:
    if notebooks_repo_name:
        return f"https://colab.research.google.com/github/{notebooks_repo_name}/blob/{notebooks_branch}/{notebook_relative_path}"
    else:
        return ""


def get_github_link(
    notebooks_repo_url: str,
    notebooks_branch: str,
    notebook_relative_path: str,
) -> str:
    return f"{notebooks_repo_url}/blob/{notebooks_branch}/{notebook_relative_path}"


def get_repo_n_branches_for_binder_n_github_links() -> Tuple[
    bool, str, str, str, str, str, bool
]:
    # repos + branches to use for binder environment and notebooks content.
    creating_links = True
    use_nbgitpuller = False
    try:
        use_nbgitpuller_str = os.environ["AUTODOC_BINDER_NBGITPULLER"]
        try:
            use_nbgitpuller_int = int(use_nbgitpuller_str)
        except ValueError:
            use_nbgitpuller_int = 1
        if (use_nbgitpuller_str.lower() != "false") and (use_nbgitpuller_int != 0):
            use_nbgitpuller = True
    except KeyError:
        pass
    try:
        binder_env_repo_name = os.environ["AUTODOC_BINDER_ENV_GH_REPO_NAME"]
    except KeyError:
        binder_env_repo_name = DEFAULT_REPO_NAME
    try:
        binder_env_branch = os.environ["AUTODOC_BINDER_ENV_GH_BRANCH"]
    except KeyError:
        binder_env_branch = "binder"
    try:
        notebooks_repo_url = os.environ["AUTODOC_NOTEBOOKS_REPO_URL"]
        notebooks_branch = os.environ["AUTODOC_NOTEBOOKS_BRANCH"]
    except KeyError:
        # missing environment variables => no github and binder links creation
        notebooks_repo_url = ""
        notebooks_branch = ""
        creating_links = False
        logger.warning(
            "Missing environment variables AUTODOC_NOTEBOOKS_REPO_URL "
            "or AUTODOC_NOTEBOOKS_BRANCH to create github and binder links for notebooks."
        )
    try:
        notebooks_repo_name = os.environ["AUTODOC_NOTEBOOKS_REPO_NAME"]
    except KeyError:
        notebooks_repo_name = ""
        match = re.match(".*/(.*/.*)/?", notebooks_repo_url)
        if match:
            notebooks_repo_name = match.group(1)
    return (
        creating_links,
        notebooks_repo_url,
        notebooks_repo_name,
        notebooks_branch,
        binder_env_repo_name,
        binder_env_branch,
        use_nbgitpuller,
    )


def extract_notebook_title_n_description(
    notebook_filepath: str,
) -> Tuple[str, List[str]]:
    # load notebook
    with open(notebook_filepath, "rt", encoding="utf-8") as f:
        notebook = json.load(f)

    # find title + description: from first cell,  h1 title + remaining text.
    # or title from filename else
    title = ""
    description_lines: List[str] = []
    cell = notebook["cells"][0]
    if cell["cell_type"] == "markdown":
        firstline = cell["source"][0].strip()
        if firstline.startswith("# "):
            title = cell["source"][0][2:].strip()
            description_lines = cell["source"][1:]
        else:
            description_lines = cell["source"]
    if not title:
        title = os.path.splitext(os.path.basename(notebook_filepath))[0]

    return title, description_lines


if __name__ == "__main__":
    # ========== GATHER AUTODOC INFOS ==========

    # Get all scikit-decide (sub)modules
    modules = []
    for m in find_abs_modules(skdecide):
        try:
            module = importlib.import_module(m)
            modules.append(module)
        except Exception as e:
            print(f"Could not load module {m}, so it will be ignored ({e}).")

    autodocs = []
    for module in modules:
        autodoc = {}

        # Get module-level infos
        autodoc["ref"] = get_ref(module)
        doc = inspect.getdoc(module)
        if doc is not None:
            autodoc["doc"] = format_doc(doc)
        members = inspect.getmembers(module, lambda x: inspect.getmodule(x) == module)
        autodoc_members = []
        for member_name, member in members:
            member = inspect.unwrap(member)

            # Get member-level infos
            if getattr(member, "__doc__", None) is not None:
                autodoc_member = {}
                autodoc_member["name"] = member_name
                add_basic_member_infos(member, autodoc_member)

                if inspect.isfunction(member):
                    add_func_method_infos(member, autodoc_member)

                elif inspect.isclass(member):
                    autodoc_member["type"] = "class"
                    autodoc_member["bases"] = (
                        list(map(get_ref, member.__bases__))
                        if member.__bases__ != (object,)
                        else None
                    )
                    autodoc_member["inheritance"] = list(
                        map(get_ref, inspect.getmro(member)[:-1])
                    )
                    submembers = inspect.getmembers(member)
                    autodoc_submembers = []
                    for submember_name, submember in submembers:
                        submember = inspect.unwrap(submember)

                        # Get class member-level infos
                        if (
                            submember_name == "__init__"
                            or not submember_name.startswith("__")
                        ):
                            autodoc_submember = {}
                            autodoc_submember["name"] = (
                                submember_name
                                if submember_name != "__init__"
                                else member_name
                            )
                            add_basic_member_infos(submember, autodoc_submember)

                            # Find original owner class of this member (in class inheritance)
                            if submember_name == "__init__":
                                autodoc_submember["owner"] = member_name
                            else:
                                for cls in inspect.getmro(member):
                                    if hasattr(cls, submember_name):
                                        autodoc_submember["owner"] = cls.__name__

                            if (
                                inspect.isfunction(submember)
                                or inspect.ismethod(submember)
                                or submember_name == "__init__"
                            ):
                                add_func_method_infos(submember, autodoc_submember)
                            elif isinstance(submember, Enum):
                                autodoc_submember["type"] = "enum"
                            else:
                                # Class variables (e.g. T_memory, T_agent...)
                                autodoc_submember["type"] = "variable"

                            if (
                                "doc" in autodoc_submember
                                or autodoc_submember.get("type") == "variable"
                            ):
                                autodoc_submembers.append(autodoc_submember)

                    autodoc_member["members"] = sorted(
                        autodoc_submembers,
                        key=lambda x: x["line"] if "line" in x else 0,
                    )

                if "doc" in autodoc_member:
                    autodoc_members.append(autodoc_member)

        autodoc["members"] = sorted(
            autodoc_members, key=lambda x: x["line"] if "line" in x else 0
        )
        autodocs.append(autodoc)

    # ========== GENERATE MARKDOWN FILES ==========

    # Remove all previously auto-generated files
    for oldpath in (
        glob(f"{docdir}/reference/_*.md")
        + glob(f"{docdir}/guide/_*.md")
        + glob(f"{docdir}/.vuepress/public/notebooks/*.ipynb")
    ):
        os.remove(oldpath)

    # Generate Reference Markdown files (reference/_skdecide.*.md)
    os.makedirs(f"{docdir}/reference", exist_ok=True)
    for module in autodocs:

        # Initiate Markdown
        md = ""

        # Write module title
        md += f'# {module["ref"].split(".", 1)[-1]}\n\n'

        # Write module doc (if any)
        if "doc" in module:
            md += f'{module["doc"]}\n\n'

        # Write domain spec summary
        md += "::: tip Domain specification\n<skdecide-summary></skdecide-summary>\n:::\n\n"

        # Write members
        for member in module["members"]:

            # Write member title
            md += f'## {md_escape(member["name"])}\n\n'

            # Write member signature (if any)
            md = write_signature(md, member)

            # Write member doc (if any)
            if "doc" in member:
                md += f'{doc_escape(member["doc"])}\n\n'

            # Write submembers (if any)
            if "members" in member:
                for submember in sorted(
                    member["members"],
                    key=lambda x: (x["name"].startswith("_"), x["name"]),
                ):
                    if submember["type"] != "variable":

                        # Write submember title
                        md += (
                            f'### {md_escape(submember["name"]) if submember["name"] != member["name"] else "Constructor"}'
                            f' <Badge text="{submember["owner"]}" type="{"tip" if submember["owner"] == member["name"] else "warn"}"/>\n\n'
                        )

                        # Write submember signature (if any)
                        md = write_signature(md, submember)

                        # Write submember doc (if any)
                        if "doc" in submember:
                            md += f'{doc_escape(submember["doc"])}\n\n'

        with open(f'{docdir}/reference/_{module["ref"]}.md', "w") as f:
            f.write(md)

    # Write Reference index (reference/README.md)
    REF_INDEX_MAXDEPTH = 5
    ref_entries = sorted(
        [tuple(m["ref"].split(".")) for m in autodocs],
        key=lambda x: [
            x[i] if i < len(x) - 1 else "" for i in range(REF_INDEX_MAXDEPTH)
        ],
    )  # tree-sorted entries
    ref_entries = filter(
        lambda e: len(e) <= REF_INDEX_MAXDEPTH, ref_entries
    )  # filter out by max depth
    ref_entries = [
        {"text": e[-1], "link": ".".join(e), "section": e[:-1]} for e in ref_entries
    ]  # organize entries

    reference = ""
    sections = set()
    for e in ref_entries:
        for i in range(1, len(e["section"]) + 1):
            section = e["section"][:i]
            if section not in sections:
                title = "Reference"
                if section[-1] != "skdecide":
                    title = section[-1]
                    reference += "\n"
                reference += f'{"".join(["#"]*i)} {title}\n\n'
                sections.add(section)
        reference += f'- <router-link to="_{e["link"]}">{e["text"]}</router-link>\n'

    with open(f"{docdir}/reference/README.md", "w") as f:
        f.write(reference)

    # Write Domain/Solver Specification pages (guide/_domainspec.md & guide/_solverspec.md)
    state = {
        "selection": {},
        "templates": {},
        "characteristics": {},
        "methods": {},
        "types": {},
        "signatures": {},
        "objects": {},
    }
    for element in ["domain", "solver"]:
        spec = ""
        characteristics = [
            module
            for module in autodocs
            if module["ref"].startswith(f"skdecide.builders.{element}.")
            and ".scheduling." not in module["ref"]
        ]  # TODO: add separate scheduling domain/solver generator?
        default_characteristics = {
            c["ref"].split(".")[-1].capitalize(): "(none)" for c in characteristics
        }

        tmp_templates = []
        for template in [
            member
            for module in autodocs
            if module["ref"] == f"skdecide.{element}s"
            for member in module["members"]
        ]:
            if template["name"] == element.capitalize():
                mandatory_characteristics = [
                    base.split(".")[-2].capitalize() for base in template["bases"] or []
                ]
            tmp_templates.append(
                {
                    "name": template["name"],
                    "characteristics": dict(
                        default_characteristics,
                        **{
                            base.split(".")[-2].capitalize(): base.split(".")[-1]
                            for base in template["bases"] or []
                            if base.split(".")[-1] != element.capitalize()
                        },
                    ),
                }
            )
            spec += f'<template v-slot:{template["name"]}>\n\n'
            if "doc" in template:
                spec += f'{doc_escape(template["doc"])}\n\n'
            spec += "</template>\n\n"

        tmp_characteristics = []
        for characteristic in characteristics:
            characteristic_name = characteristic["ref"].split(".")[-1].capitalize()
            tmp_characteristics.append({"name": characteristic_name, "levels": []})
            if characteristic_name not in mandatory_characteristics:
                tmp_characteristics[-1]["levels"].append("(none)")
            for level in characteristic["members"]:
                tmp_characteristics[-1]["levels"].append(level["name"])
                spec += f'<template v-slot:{level["name"]}>\n\n'
                if "doc" in level:
                    spec += f'{doc_escape(level["doc"])}\n\n'
                spec += "</template>\n\n"

        state["selection"][element] = {
            "template": tmp_templates[0]["name"],
            "characteristics": tmp_templates[0]["characteristics"],
            "showFinetunedOnly": True,
        }
        if element == "domain":
            state["selection"][element]["simplifySignatures"] = True
        state["templates"][element] = tmp_templates
        state["characteristics"][element] = tmp_characteristics

        spec = (
            "---\n"
            "navbar: false\n"
            "sidebar: false\n"
            "---\n\n"
            f'<skdecide-spec{" isSolver" if element == "solver" else ""}>\n\n' + spec
        )
        spec += "</skdecide-spec>\n\n"

        with open(f"{docdir}/codegen/_{element}spec.md", "w") as f:
            f.write(spec)

    # Write Json state (.vuepress/_state.json)
    state["objects"] = {
        member["name"]: f'/reference/_skdecide.core.html#{member["name"].lower()}'
        for module in autodocs
        if module["ref"] == "skdecide.core"
        for member in module["members"]
    }
    for element in ["domain", "solver"]:
        tmp_methods = (
            {}
        )  # TODO: detect classmethods/staticmethods to add decorator in code generator (only necessary if there was any NotImplemented classmethod/staticmethod in base template or any characteristic level)
        tmp_types = {}
        tmp_signatures = {}
        for module in autodocs:
            if (
                module["ref"].startswith(f"skdecide.builders.{element}.")
                and ".scheduling." not in module["ref"]
            ):  # TODO: also store scheduling domain/state state (separately?) once scheduling domain/solver generator implemented
                not_implemented = set()
                for level in module.get("members", []):
                    level_name = level["name"]
                    types_dict = {}
                    for member in level.get("members", []):
                        member_name = member["name"]
                        if member["type"] == "function":
                            tmp_signatures[member_name] = member["signature"]
                            if is_implemented(member["source"]):
                                not_implemented.discard(member_name)
                            else:
                                not_implemented.add(member_name)
                        elif member["type"] == "variable":
                            types_dict[member_name] = member["ref"]
                    tmp_methods[level_name] = list(not_implemented)
                    tmp_types[level_name] = types_dict
            elif module["ref"] == f"skdecide.{element}s":
                for template in module["members"]:
                    if template["name"] == element.capitalize():
                        tmp_methods[element] = []
                        for member in template.get("members", []):
                            if (
                                member["type"] == "function"
                                and member["owner"] == element.capitalize()
                                and not is_implemented(member["source"])
                            ):
                                member_name = member["name"]
                                tmp_signatures[member_name] = member["signature"]
                                tmp_methods[element].append(member_name)

        state["methods"][element] = tmp_methods
        state["types"][element] = tmp_types
        state["signatures"][element] = tmp_signatures

    with open(f"{docdir}/.vuepress/_state.json", "w") as f:
        json.dump(state, f)

    # List existing notebooks and write Notebooks page
    notebook_filepaths = sorted(glob(f"{notebooksdir}/**/*.ipynb", recursive=True))
    notebooks_list_text = ""
    notebooksdir_prefixlen = len(notebooksdir) + 1
    sections_baselevel = 2
    current_sections = []
    (
        creating_links,
        notebooks_repo_url,
        notebooks_repo_name,
        notebooks_branch,
        binder_env_repo_name,
        binder_env_branch,
        use_nbgitpuller,
    ) = get_repo_n_branches_for_binder_n_github_links()
    # loop on notebooks sorted alphabetically by filenames
    for notebook_filepath in notebook_filepaths:
        # get subsections arborescence
        notebook_relpath = notebook_filepath[notebooksdir_prefixlen:]
        notebook_arbo = notebook_relpath.split(os.path.sep)
        notebook_sections = notebook_arbo[:-1]
        # write missing sections
        for i_section, section in enumerate(notebook_sections):
            if (
                i_section >= len(current_sections)
                or section != current_sections[i_section]
            ):
                section_prefix = (sections_baselevel + i_section) * "#"
                section_name = section.split(NOTEBOOKS_SECTION_KEY_VAR_SEP)[-1]
                notebooks_list_text += f"{section_prefix} {section_name}\n\n"
        current_sections = notebook_sections
        # extract title and description
        title, description_lines = extract_notebook_title_n_description(
            notebook_filepath
        )
        # write title
        title_prefix = (sections_baselevel + len(notebook_sections)) * "#"
        notebooks_list_text += f"{title_prefix} {title}\n\n"
        # links
        if creating_links:
            notebook_path_prefix_len = len(f"{rootdir}/")
            notebook_relative_path = notebook_filepath[notebook_path_prefix_len:]
            binder_link = get_binder_link(
                binder_env_repo_name=binder_env_repo_name,
                binder_env_branch=binder_env_branch,
                notebooks_repo_url=notebooks_repo_url,
                notebooks_branch=notebooks_branch,
                notebook_relative_path=notebook_relative_path,
                notebooks_repo_name=notebooks_repo_name,
                use_nbgitpuller=use_nbgitpuller,
            )
            if binder_link:
                binder_badge = (
                    f"[![Binder](https://mybinder.org/badge_logo.svg)]({binder_link})"
                )
            else:
                binder_badge = ""
            github_link = get_github_link(
                notebooks_repo_url=notebooks_repo_url,
                notebooks_branch=notebooks_branch,
                notebook_relative_path=notebook_relative_path,
            )
            github_badge = f"[![Github](https://img.shields.io/badge/see-Github-579aca?logo=github)]({github_link})"
            colab_link = get_colab_link(
                notebooks_repo_name=notebooks_repo_name,
                notebooks_branch=notebooks_branch,
                notebook_relative_path=notebook_relative_path,
            )
            if colab_link:
                colab_badge = f"[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})"
            else:
                colab_badge = ""
            # markdown item
            # notebooks_list_text += f"{github_badge}\n{binder_badge}\n\n"
            notebooks_list_text += f"{github_badge}\n"
            if colab_badge:
                notebooks_list_text += f"{colab_badge}\n"
            if binder_badge:
                notebooks_list_text += f"{binder_badge}\n"
            notebooks_list_text += "\n"

        # description
        notebooks_list_text += "".join(description_lines)
        notebooks_list_text += "\n\n"

    with open(f"{docdir}/{NOTEBOOKS_PAGE_TEMPLATE_RELATIVE_PATH}", "rt") as f:
        readme_template_text = f.read()

    readme_text = readme_template_text.replace(
        NOTEBOOKS_LIST_PLACEHOLDER, notebooks_list_text
    )

    with open(f"{docdir}/{NOTEBOOKS_PAGE_RELATIVE_PATH}", "wt") as f:
        f.write(readme_text)
