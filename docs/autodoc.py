# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import importlib
import inspect
import json
import os
import pkgutil
import re
import sys
from functools import lru_cache
from glob import glob

import skdecide

refs = set()
header_comment = '# %%\n'


# https://github.com/kiwi0fruit/ipynb-py-convert/blob/master/ipynb_py_convert/__main__.py
def py2nb(py_str):
    cells = []
    chunks = py_str.split(f'\n\n{header_comment}')[1:]

    for chunk in chunks:
        cell_type = 'code'
        chunk = chunk.strip()
        if chunk.startswith("'''"):
            chunk = chunk.strip("'\n")
            cell_type = 'markdown'

        cell = {
            'cell_type': cell_type,
            'metadata': {},
            'source': chunk.splitlines(True),
        }

        if cell_type == 'code':
            cell.update({'outputs': [], 'execution_count': None})

        cells.append(cell)

    notebook = {
        'cells': cells,
        'metadata': {
            'anaconda-cloud': {},
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'},
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.6.1'}},
        'nbformat': 4,
        'nbformat_minor': 1
    }

    return notebook


# https://stackoverflow.com/questions/48879353/how-do-you-recursively-get-all-submodules-in-a-python-package
def find_abs_modules(package):
    path_list = []
    spec_list = []
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__):
        import_path = f'{package.__name__}.{modname}'
        if ispkg:
            spec = pkgutil._get_spec(importer, modname)
            try:
                importlib._bootstrap._load(spec)
                spec_list.append(spec)
            except Exception as e:
                print(f'Could not load package {modname}, so it will be ignored ({e}).')
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
    docstring = '' if docstring is None else docstring.strip()
    name = os.path.splitext(os.path.basename(filepath))[0]
    return docstring, name, file_contents


@lru_cache(maxsize=1000)
def get_ref(object):
    name = getattr(object, '__qualname__', None)
    if name is None:
        name = getattr(object, '__name__', None)
        if name is None:
            name = object._name
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
    try:
        autodoc['ref'] = get_ref(member)
        source, line = inspect.getsourcelines(member)
        autodoc['source'] = ''.join(source)  # TODO: keep?
        autodoc['line'] = line
        doc = inspect.getdoc(member)
        if doc is not None:
            autodoc['doc'] = format_doc(doc)
    except Exception:  # can happen e.g. when member is TypeVar
        pass


def json_escape(obj):
    return json.dumps(obj).replace("'", r"\'").replace('"', "'")


def md_escape(md):
    return re.sub(r'[_<>]', lambda m: f'\\{m.group()}', md)


def doc_escape(md):
    return re.sub(r'[<]', lambda m: f'\\{m.group()}', md)


def write_signature(md, member):
    if 'signature' in member:
        escape_json_sig = json_escape(member['signature'])
        md += f'<skdecide-signature name= "{member["name"]}" :sig="{escape_json_sig}"></skdecide-signature>\n\n'
    return md


def is_implemented(func_code):
    return not func_code.strip().endswith('raise NotImplementedError')


if __name__ == '__main__':

    # ========== GATHER AUTODOC INFOS ==========

    # Get all scikit-decide (sub)modules
    modules = []
    for m in find_abs_modules(skdecide):
        try:
            module = importlib.import_module(m)
            modules.append(module)
        except Exception as e:
            print(f'Could not load module {m}, so it will be ignored ({e}).')

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
                    autodoc_member['inheritance'] = list(map(get_ref, inspect.getmro(member)[:-1]))
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

                            if inspect.isfunction(submember) or inspect.ismethod(submember) or submember_name == '__init__':
                                add_func_method_infos(submember, autodoc_submember)

                            else:
                                # Class variables (e.g. T_memory, T_agent...)
                                autodoc_submember['type'] = 'variable'

                            if 'doc' in autodoc_submember or autodoc_submember.get('type') == 'variable':
                                autodoc_submembers.append(autodoc_submember)

                    autodoc_member['members'] = sorted(autodoc_submembers, key=lambda x: x['line'] if 'line' in x else 0)

                if 'doc' in autodoc_member:
                    autodoc_members.append(autodoc_member)

        autodoc['members'] = sorted(autodoc_members, key=lambda x: x['line'] if 'line' in x else 0)
        autodocs.append(autodoc)

    # ========== GENERATE MARKDOWN FILES ==========

    # Remove all previously auto-generated files
    for oldpath in glob('reference/_*.md') + glob('guide/_*.md') + glob('.vuepress/public/notebooks/*.ipynb'):
        os.remove(oldpath)

    # Generate Reference Markdown files (reference/_skdecide.*.md)
    for module in autodocs:

        # Initiate Markdown
        md = ''

        # Write module title
        md += f'# {module["ref"].split(".", 1)[-1]}\n\n'

        # Write module doc (if any)
        if 'doc' in module:
            md += f'{module["doc"]}\n\n'

        # Write Table Of Content
        md += '[[toc]]\n\n'

        # Write domain spec summary
        md += '::: tip\n<skdecide-summary></skdecide-summary>\n:::\n\n'

        # Write members
        for member in module['members']:

            # Write member title
            md += f'## {md_escape(member["name"])}\n\n'

            # Write member signature (if any)
            md = write_signature(md, member)

            # Write member doc (if any)
            if 'doc' in member:
                md += f'{doc_escape(member["doc"])}\n\n'

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
                            md += f'{doc_escape(submember["doc"])}\n\n'

        with open(f'reference/_{module["ref"]}.md', 'w') as f:
            f.write(md)

    # Write Reference index (reference/README.md)
    REF_INDEX_MAXDEPTH = 5
    ref_entries = sorted([tuple(m['ref'].split('.')) for m in autodocs], key=lambda x: (len(x), x))  # sorted entries
    ref_entries = filter(lambda e: len(e) <= REF_INDEX_MAXDEPTH, ref_entries)  # filter out by max depth
    ref_entries = [{'text': e[-1], 'link': '.'.join(e), 'section': e[:-1]} for e in ref_entries]  # organize entries

    reference = ''
    sections = set()
    for e in ref_entries:
        for i in range(1, len(e['section']) + 1):
            section = e['section'][:i]
            if section not in sections:
                title = 'Reference'
                if section[-1] != 'skdecide':
                    title = section[-1]
                    reference += '\n'
                reference += f'{"".join(["#"]*i)} {title}\n\n'
                sections.add(section)
        reference += f'- <router-link to="_{e["link"]}">{e["text"]}</router-link>\n'

    with open(f'reference/README.md', 'w') as f:
        f.write(reference)

    # Write Domain/Solver Specification pages (guide/_domainspec.md & guide/_solverspec.md)
    state = {
        'selection': {},
        'templates': {},
        'characteristics': {},
        'methods': {},
        'types': {},
        'signatures': {},
        'objects': {}
    }
    for element in ['domain', 'solver']:
        spec = ''
        characteristics = [module for module in autodocs if module['ref'].startswith(f'skdecide.builders.{element}.')]
        default_characteristics = {c['ref'].split('.')[-1].capitalize(): '(none)' for c in characteristics}
    
        tmp_templates = []
        for template in [member for module in autodocs if module['ref'] == f'skdecide.{element}s' for member in module['members']]:
            if template['name'] == element.capitalize():
                mandatory_characteristics = [base.split('.')[-2].capitalize() for base in template['bases'] or []]
            tmp_templates.append({'name': template['name'], 'characteristics': dict(default_characteristics, **{base.split('.')[-2].capitalize(): base.split('.')[-1] for base in template['bases'] or [] if base.split('.')[-1] != element.capitalize()})})
            spec += f'<template v-slot:{template["name"]}>\n\n'
            if 'doc' in template:
                spec += f'{doc_escape(template["doc"])}\n\n'
            spec += '</template>\n\n'
    
        tmp_characteristics = []
        for characteristic in characteristics:
            characteristic_name = characteristic['ref'].split('.')[-1].capitalize()
            tmp_characteristics.append({'name': characteristic_name, 'levels': []})
            if characteristic_name not in mandatory_characteristics:
                tmp_characteristics[-1]['levels'].append('(none)')
            for level in characteristic['members']:
                tmp_characteristics[-1]['levels'].append(level['name'])
                spec += f'<template v-slot:{level["name"]}>\n\n'
                if 'doc' in level:
                    spec += f'{doc_escape(level["doc"])}\n\n'
                spec += '</template>\n\n'

        state['selection'][element] = {
                'template': tmp_templates[0]['name'],
                'characteristics': tmp_templates[0]['characteristics'],
                'showFinetunedOnly': True
            }
        if element == 'domain':
            state['selection'][element]['simplifySignatures'] = True
        state['templates'][element] = tmp_templates
        state['characteristics'][element] = tmp_characteristics

        spec = '---\n' \
                     'navbar: false\n' \
                     'sidebar: false\n' \
                     '---\n\n' \
                     f'<skdecide-spec{" isSolver" if element == "solver" else ""}>\n\n' + spec
        spec += '</skdecide-spec>\n\n'

        with open(f'guide/_{element}spec.md', 'w') as f:
            f.write(spec)

    # Write Json state (.vuepress/_state.json)
    state['objects'] = {member['name']: f'/reference/_skdecide.core.html#{member["name"].lower()}' for module in autodocs if module['ref'] == 'skdecide.core' for member in module['members']}
    for element in ['domain', 'solver']:
        tmp_methods = {}  # TODO: detect classmethods/staticmethods to add decorator in code generator (only necessary if there was any NotImplemented classmethod/staticmethod in base template or any characteristic level)
        tmp_types = {}
        tmp_signatures = {}
        for module in autodocs:
            if module['ref'].startswith(f'skdecide.builders.{element}.'):
                not_implemented = set()
                for level in module.get('members', []):
                    level_name = level['name']
                    types_dict = {}
                    for member in level.get('members', []):
                        member_name = member['name']
                        if member['type'] == 'function':
                            tmp_signatures[member_name] = member['signature']
                            if is_implemented(member['source']):
                                not_implemented.discard(member_name)
                            else:
                                not_implemented.add(member_name)
                        elif member['type'] == 'variable':
                            types_dict[member_name] = member['ref']
                    tmp_methods[level_name] = list(not_implemented)
                    tmp_types[level_name] = types_dict
            elif module['ref'] == f'skdecide.{element}s':
                for template in module['members']:
                    if template['name'] == element.capitalize():
                        tmp_methods[element] = []
                        for member in template.get('members', []):
                            if member['type'] == 'function' and member['owner'] == element.capitalize() and not is_implemented(member['source']):
                                member_name = member['name']
                                tmp_signatures[member_name] = member['signature']
                                tmp_methods[element].append(member_name)

        state['methods'][element] = tmp_methods
        state['types'][element] = tmp_types
        state['signatures'][element] = tmp_signatures

    with open('.vuepress/_state.json', 'w') as f:
        json.dump(state, f)

    # Convert selected examples to notebooks & write Examples page (guide/_examples.md)
    examples = '# Examples\n\n'

    selected_examples = []
    for example in glob('../examples/*.py'):
        docstr, name, code = py_parse(example)
        if docstr.startswith('Example '):
            selected_examples.append((docstr, name, code))

    sorted_examples = sorted(selected_examples)
    for docstr, name, code in sorted_examples:
        examples += f'## {docstr[docstr.index(":")+1:]}\n\n'
        examples += f'<el-link type="primary" icon="el-icon-bottom" :underline="false" style="margin: 10px" href="/notebooks/{name}.ipynb">Download Notebook</el-link>\n'
        examples += f'<el-link type="warning" icon="el-icon-cloudy" :underline="false" style="margin: 10px" href="https://colab.research.google.com/github/airbus/scikit-decide/blob/gh-pages/notebooks/{name}.ipynb">Run in Google Colab</el-link>\n\n'
        notebook = py2nb(code)

        for cell in notebook.get('cells', []):
            cell_type = cell['cell_type']
            cell_source = ''.join(cell['source'])
            if cell_type == 'markdown':
                examples += f'{cell_source}\n\n'
            elif cell_type == 'code':
                examples += f'``` py\n{cell_source}\n```\n\n'

        with open(f'.vuepress/public/notebooks/{name}.ipynb', 'w') as f:
            json.dump(notebook, f, indent=2)

    with open('guide/_examples.md', 'w') as f:
        f.write(examples)
