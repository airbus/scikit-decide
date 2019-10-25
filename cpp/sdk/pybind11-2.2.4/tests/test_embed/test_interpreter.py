# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from widget_module import Widget


class DerivedWidget(Widget):
    def __init__(self, message):
        super(DerivedWidget, self).__init__(message)

    def the_answer(self):
        return 42
