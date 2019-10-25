# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
from os import getenv
from os import path
from conans import ConanFile
from conans import CMake


class CatchConanTest(ConanFile):
    generators = "cmake"
    settings = "os", "compiler", "arch", "build_type"
    username = getenv("CONAN_USERNAME", "philsquared")
    channel = getenv("CONAN_CHANNEL", "testing")
    requires = "Catch/2.4.0@%s/%s" % (username, channel)

    def build(self):
        cmake = CMake(self)
        cmake.configure(build_dir="./")
        cmake.build()

    def test(self):
        self.run(path.join("bin", "CatchTest"))
