/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "catch_tag_alias.h"

namespace Catch {
    TagAlias::TagAlias(std::string const & _tag, SourceLineInfo _lineInfo): tag(_tag), lineInfo(_lineInfo) {}    
}
