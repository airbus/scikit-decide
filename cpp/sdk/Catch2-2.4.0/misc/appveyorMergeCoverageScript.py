# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

import glob
import subprocess

if __name__ == '__main__':
    cov_files = list(glob.glob('cov-report*.bin'))
    base_cmd = ['OpenCppCoverage', '--quiet', '--export_type=cobertura:cobertura.xml'] + ['--input_coverage={}'.format(f) for f in cov_files]
    subprocess.call(base_cmd)
