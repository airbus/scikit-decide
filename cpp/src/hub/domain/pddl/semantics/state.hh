/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEMANTICS_STATE_HH
#define SKDECIDE_PDDL_SEMANTICS_STATE_HH

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace skdecide {

namespace pddl {

using GroundTuple = std::vector<int>;

struct GroundTupleHash {
  std::size_t operator()(const GroundTuple &t) const {
    std::size_t seed = t.size();
    for (auto v : t) {
      seed ^= std::hash<int>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

using TupleSet = std::unordered_set<GroundTuple, GroundTupleHash>;
using FluentMap = std::unordered_map<GroundTuple, double, GroundTupleHash>;
using Binding = std::unordered_map<std::string, int>;

struct ActiveDurativeAction {
  int action_id;
  Binding binding;
  double start_time;
  double end_time;

  bool operator==(const ActiveDurativeAction &other) const {
    return action_id == other.action_id && binding == other.binding &&
           start_time == other.start_time && end_time == other.end_time;
  }
};

struct State {
  std::vector<TupleSet> atoms;
  std::vector<FluentMap> fluents;
  double time = 0.0;
  double dt = 0.0;
  double duration = 0.0;
  std::vector<ActiveDurativeAction> active_durative_actions;

  State copy() const { return *this; }

  bool operator==(const State &other) const {
    return atoms == other.atoms && fluents == other.fluents &&
           time == other.time &&
           active_durative_actions == other.active_durative_actions;
  }

  std::size_t hash() const {
    std::size_t seed = 0;
    GroundTupleHash th;
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      seed ^=
          std::hash<std::size_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      // Sort element hashes for order-independence: TupleSet is an
      // unordered_set whose iteration order may differ across copies.
      std::vector<std::size_t> hashes;
      hashes.reserve(atoms[i].size());
      for (auto &t : atoms[i]) {
        hashes.push_back(th(t));
      }
      std::sort(hashes.begin(), hashes.end());
      for (auto h : hashes) {
        seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
    }
    for (std::size_t i = 0; i < fluents.size(); ++i) {
      seed ^=
          std::hash<std::size_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      // Sort entry hashes for order-independence: FluentMap is an
      // unordered_map whose iteration order may differ across copies.
      std::vector<std::size_t> hashes;
      hashes.reserve(fluents[i].size());
      for (auto &[k, v] : fluents[i]) {
        std::size_t entry_hash = th(k);
        entry_hash ^= std::hash<double>{}(v) + 0x9e3779b9 + (entry_hash << 6) +
                      (entry_hash >> 2);
        hashes.push_back(entry_hash);
      }
      std::sort(hashes.begin(), hashes.end());
      for (auto h : hashes) {
        seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
    }
    seed ^= std::hash<double>{}(time) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    for (auto &ada : active_durative_actions) {
      seed ^= std::hash<int>{}(ada.action_id) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
      seed ^= std::hash<double>{}(ada.start_time) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
      seed ^= std::hash<double>{}(ada.end_time) + 0x9e3779b9 + (seed << 6) +
              (seed >> 2);
    }
    return seed;
  }
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEMANTICS_STATE_HH
