/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ff.hh"
#include "impl/ff_impl.hh"

template class skdecide::pddl::FFSolver<skdecide::SequentialExecution>;
template class skdecide::pddl::FFSolver<skdecide::ParallelExecution>;
