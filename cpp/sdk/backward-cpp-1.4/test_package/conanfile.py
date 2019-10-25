# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from conans import ConanFile, CMake
import os

class TestBackward(ConanFile):
    settings = 'os', 'compiler', 'build_type', 'arch'
    generators = 'cmake'

    def build(self):
        cmake = CMake(self)
        cmake.configure(defs={'CMAKE_VERBOSE_MAKEFILE': 'ON'})
        cmake.build()

    def test(self):
        self.run(os.path.join('.', 'bin', 'example'))
