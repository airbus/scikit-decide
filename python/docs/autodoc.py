import importlib
import inspect
import json
# from pathlib import Path
import pkgutil
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pprint import pprint  # TODO: remove

# airlaps_dir = Path(__file__).parent.parent
# print(str(airlaps_dir))
# sys.path.insert(0, str(airlaps_dir))
import airlaps
# print(airlaps.__file__)

refs = set()

# https://stackoverflow.com/questions/48879353/how-do-you-recursively-get-all-submodules-in-a-python-package
def find_abs_modules(package):
    path_list = []
    spec_list = []
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__):
        import_path = f'{package.__name__}.{modname}'
        if ispkg:
            spec = pkgutil._get_spec(importer, modname)
            importlib._bootstrap._load(spec)
            spec_list.append(spec)
        else:
            path_list.append(import_path)
    for spec in spec_list:
        del sys.modules[spec.name]
    return path_list


@lru_cache(maxsize=1000)
def get_ref(object):
    name = getattr(object, '__qualname__', '__name__')
    if name == '__name__':
        name = object.__name__
    reflist = [name]
    if hasattr(object, '__module__'):
        reflist.insert(0, object.__module__)
    ref = '.'.join(reflist)
    refs.add(ref)
    return ref


def format_doc(doc):

    # Replace references like "#obj.func()" by "`obj.func()`" for Markdown code highlighting - TODO: replace in doc?
    doc = re.sub(r'#(?P<ref>[\w\.,()]*[\w()])', lambda m: f'`{m.group("ref")}`', doc)

    # Replace content of "# Parameters" by list of parameters
    def list_content(content):
        content = re.sub(r'^(?P<param>\w+)', lambda m: f'- **{m.group("param")}**', content, flags=re.MULTILINE)
        return content.strip()
    doc = re.sub(r'^(?<=# Parameters\n)(?P<content>(?:\n?\s*\w.*)+)',
                 lambda m: list_content(m.group("content")), doc, flags=re.MULTILINE)

    # Replace "# Title" (e.g. "# Parameters") by "#### Title"
    doc = re.sub(r'^# (?=Parameters|Returns|Example)', '#### ', doc, flags=re.MULTILINE)

    # Replace "!!! container" (e.g. "!!! tip") by "::: container [...] :::"
    def strip_content(content):
        content = re.sub(r'^ {4}', '', content, flags=re.MULTILINE)
        return content.rstrip()
    doc = re.sub(r'!!! (?=tip|warning|danger)(?P<head>.*)\n(?P<content>(?:\n? {4,}.*)+)',
                 lambda m: f'::: {m.group("head")}\n{strip_content(m.group("content"))}\n:::', doc)

    return doc


def add_func_method_infos(func_method, autodoc):
    if inspect.isfunction(func_method):
        autodoc['type'] = 'function'
    elif inspect.ismethod(func_method):
        autodoc['type'] = 'method'

    # Get signature
    signature = inspect.signature(func_method)
    parameters = signature.parameters
    params = []
    for k, v in parameters.items():
        if not (k == 'self' and func_method.__name__ == '__init__'):
            parameter = parameters[k]
            param = {'name': k}
            if parameter.default != signature.empty:
                param['default'] = str(parameter.default)
                if 'lambda' in param['default']:
                    param['default'] = '<lambda function>'  # TODO: improve?
            if parameter.annotation != signature.empty:
                param['annotation'] = parameter.annotation
            params.append(param)
    autodoc['signature'] = {'params': params}
    if signature.return_annotation != signature.empty:
        autodoc['signature']['return'] = signature.return_annotation


def add_basic_member_infos(member, autodoc):
    # if getattr(member, '__module__', '').startswith('airlaps.'):
    try:
        source, line = inspect.getsourcelines(member)
        autodoc['source'] = ''.join(source)  # TODO: keep?
        autodoc['line'] = line
        autodoc['ref'] = get_ref(member)
        doc = inspect.getdoc(member)
        if doc is not None:
            autodoc['doc'] = format_doc(doc)
    except Exception:  # can happen e.g. when member is TypeVar
        pass


def write_signature(md, member):
    if 'signature' in member:
        # sig = member['signature']
        # formatted_params = [f'{p["name"]}{": " + p["annotation"] if "annotation" in p else ""}{" = " + p["default"] if "default" in p else ""}' for p in sig['params']]
        # md += f'```python\n{member["name"]}({", ".join(formatted_params)})\n```\n\n'
        escape_json_sig = json.dumps(member['signature']).replace('"', "'")
        md += f'<airlaps-signature name= "{member["name"]}" :sig="{escape_json_sig}"></airlaps-signature>\n\n'
    return md

def md_escape(md):
    return re.sub(r'[_]', lambda m: f'\\{m.group()}', md)


if __name__ == '__main__':

    # Get all AIRLAPS (sub)modules
    modules = [importlib.import_module(m) for m in find_abs_modules(airlaps)]
    autodocs = []
    for module in modules:
        autodoc = {}

        # Get module-level infos
        autodoc['ref'] = get_ref(module)
        doc = inspect.getdoc(module)
        if doc is not None:
            autodoc['doc'] = format_doc(doc)
        members = inspect.getmembers(module, lambda x: inspect.getmodule(x) == module)
        autodoc_members = []
        for member_name, member in members:
            member = inspect.unwrap(member)

            # Get member-level infos
            if getattr(member, '__doc__', None) is not None:
                autodoc_member = {}
                autodoc_member['name'] = member_name
                add_basic_member_infos(member, autodoc_member)

                if inspect.isfunction(member):
                    add_func_method_infos(member, autodoc_member)

                elif inspect.isclass(member):
                    autodoc_member['type'] = 'class'
                    autodoc_member['bases'] = list(map(get_ref, member.__bases__)) if member.__bases__ != (object,) else None
                    autodoc_member['inheritance'] = list(map(get_ref, reversed(inspect.getmro(member)[:-1])))
                    submembers = inspect.getmembers(member)
                    autodoc_submembers = []
                    for submember_name, submember in submembers:
                        submember = inspect.unwrap(submember)

                        # Get class member-level infos
                        if submember_name == '__init__' or not submember_name.startswith('__'):
                            autodoc_submember = {}
                            autodoc_submember['name'] = submember_name if submember_name != '__init__' else member_name
                            add_basic_member_infos(submember, autodoc_submember)

                            # Find original owner class of this member (in class inheritance)
                            if submember_name == '__init__':
                                autodoc_submember['owner'] = member_name
                            else:
                                for cls in inspect.getmro(member):
                                    if hasattr(cls, submember_name):
                                        autodoc_submember['owner'] = cls.__name__
                                    # else:
                                    #     break

                            if inspect.isfunction(submember) or inspect.ismethod(submember):
                                add_func_method_infos(submember, autodoc_submember)

                            else:
                                # Class variables (e.g. T_memory, T_agent...)
                                autodoc_submember['type'] = 'variable'

                            if 'doc' in autodoc_submember:
                                autodoc_submembers.append(autodoc_submember)

                        # elif submember_name == '__init__':
                        #     autodoc_submember = {}
                        #     autodoc_submember['name'] = submember_name
                        #     doc = inspect.getdoc(submember)
                        #     if doc is not None:
                        #         autodoc['doc'] = format_doc(doc)
                        #     add_func_method_infos(submember, autodoc_submember)

                    autodoc_member['members'] = sorted(autodoc_submembers, key=lambda x: x['line'] if 'line' in x else 0)

                if 'doc' in autodoc_member:
                    autodoc_members.append(autodoc_member)

        autodoc['members'] = sorted(autodoc_members, key=lambda x: x['line'] if 'line' in x else 0)
        autodocs.append(autodoc)

    # pprint(autodocs)
    # print(sorted(refs))

    # # Export JSON file with all autodoc infos
    # with open('.vuepress/public/autodoc.json', 'w') as f:
    #     json.dump({'autodocs': autodocs, 'refs': list(refs)}, f)  # TODO: keep refs? Add shortrefs?

    # Generate Markdown files  # TODO: remove all previously generated files first (_*.md)
    tree = lambda: defaultdict(tree)
    ref_index = tree()
    ref_titles = {}
    ref_links = defaultdict(list)
    for module in autodocs:

        # Prepare link in Reference index
        ref_key = ref_index
        for i, pkg in enumerate(module['ref'].split('.')[:-1], 1):
            pkg = 'Reference' if pkg == 'airlaps' else pkg
            ref_key = ref_key[pkg]
            title = ''.join(['#']*i) + ' ' + pkg
            ref_titles[title] = id(ref_key)
        ref_links[id(ref_key)].append(module['ref'])

        # Initiate Markdown
        md = ''

        # Write module title
        md += f'# {module["ref"].split(".", 1)[-1]}\n\n'

        # Write module doc (if any)
        if 'doc' in module:
            md += f'{module["doc"]}\n\n'

        # Write Table Of Content
        md += '[[toc]]\n\n'

        # Write members
        for member in module['members']:

            # Write member title
            md += f'## {md_escape(member["name"])}\n\n'

            # Write member signature (if any)
            md = write_signature(md, member)

            # Write member doc (if any)
            if 'doc' in member:
                md += f'{member["doc"]}\n\n'

            # Write submembers (if any)
            if 'members' in member:
                for submember in sorted(member['members'], key=lambda x: (x['name'].startswith('_'), x['name'])):
                    if submember['type'] != 'variable':

                        # Write submember title
                        md += f'### {md_escape(submember["name"]) if submember["name"] != member["name"] else "Constructor"}' \
                              f' <Badge text="{submember["owner"]}" type="{"tip" if submember["owner"] == member["name"] else "warn"}"/>\n\n'

                        # Write submember signature (if any)
                        md = write_signature(md, submember)

                        # Write submember doc (if any)
                        if 'doc' in submember:
                            md += f'{submember["doc"]}\n\n'

        with open(f'reference/_{module["ref"]}.md', 'w') as f:
            f.write(md)

    # Write Reference index
    reference = ''
    for k, v in ref_titles.items():
        reference += f'{k}\n\n'
        for ref_link in sorted(ref_links[v]):
            reference += f'[{ref_link.rsplit(".", 1)[-1]}](_{ref_link})\n\n'
    with open(f'reference/README.md', 'w') as f:
        f.write(reference)
