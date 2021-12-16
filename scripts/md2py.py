"""Extract python code blocks from a markdown file to generate a python script

The goal is to be able to test python code examples shown in a markdown file.

Usage:
    md2py.py <md_inputfile> <py_outputfile>

Arguments:
    <md_inputfile>   path of the markdownfile from which extract python code blocks
    <py_outputfile>  path of the resulting python script

"""

import commonmark
from docopt import docopt

PYTHON_FLAVORS = ["python", "py3", "python3"]


def extract_pythoncode_from_markdown(markdown_inputfile, python_outputfile):
    with open(markdown_inputfile, "rt") as fp:
        doc = fp.read()
    python_nodes = []
    parser = commonmark.Parser()
    ast = parser.parse(doc)
    walker = ast.walker()
    for node, entering in walker:
        if node.t == "code_block" and node.is_fenced and node.info in PYTHON_FLAVORS:
            python_nodes.append(node)
    python_string = "\n".join([node.literal for node in python_nodes])
    with open(python_outputfile, "wt") as fp:
        fp.write(python_string)


if __name__ == "__main__":
    #  arguments
    arguments = docopt(__doc__)
    markdown_inputfile = arguments["<md_inputfile>"]
    python_outputfile = arguments["<py_outputfile>"]

    # process
    extract_pythoncode_from_markdown(
        markdown_inputfile=markdown_inputfile, python_outputfile=python_outputfile
    )
