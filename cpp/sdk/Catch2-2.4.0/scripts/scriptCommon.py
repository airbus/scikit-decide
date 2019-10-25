# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import subprocess


catchPath = os.path.dirname(os.path.realpath( os.path.dirname(sys.argv[0])))

def getBuildExecutable():
    dir = os.environ.get('CATCH_DEV_OUT_DIR', "cmake-build-debug/projects/SelfTest")
    return dir

def runAndCapture( args ):
    child = subprocess.Popen(" ".join( args ), shell=True, stdout=subprocess.PIPE)
    lines = []
    line = ""
    while True:
        out = child.stdout.read(1)
        if out == '' and child.poll():
            break
        if out != '':
            if out == '\n':
                lines.append( line )
                line = ""
            else:
                line = line + out
    return lines
