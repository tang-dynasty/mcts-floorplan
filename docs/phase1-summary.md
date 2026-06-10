# Phase 1 Implementation Summary

**Date**: 2026-06-10
**Status**: Complete, tested, pushed
**Commit**: `8e1bfe4`

---

## What Was Built

```
mcts-floorplan/
├── CMakeLists.txt                    # Root build, links parallel-mcts submodule
├── .gitmodules                       # Submodule: parallel-mcts
├── third_party/parallel-mcts         # Git submodule (P3-ES-WS-MCTS backbone)
├── include/floorplan/
│   ├── floorplan_state.hpp           # FloorplanState (8 GameState interfaces)
│   ├── floorplan_reward.hpp          # Area / HPWL / aspect evaluators
│   ├── floorplan_rollout.hpp         # Greedy / random / ε-greedy rollouts
│   └── floorplan_bookshelf.hpp       # MCNC/GSRC .aux/.nodes/.nets parser
├── examples/
│   ├── floorplan.cpp                 # Demo binary (uses P3ESWSMCTS)
│   └── CMakeLists.txt
├── tests/
│   ├── test_floorplan_state.cpp      # 10 unit tests (Catch2 v3)
│   └── CMakeLists.txt
└── benchmarks/CMakeLists.txt         # Placeholder for Phase 2
```

---

## Build & Test Results

```bash
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ cmake --build .
$ ./tests/test_floorplan_state
===============================================================================
All tests passed (227 assertions in 10 test cases)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Action encoding | `(module_id << 4) \| slot_id` | 10 bits, fits `uint32_t`, atomic-CAS ready |
| HPWL update | Incremental delta | O(\|pins\|) per `apply()` vs full recompute |
| Canvas boundary | Dynamic | Rollouts adapt; fixed boundary limits search |
| Reward normalization | Linear to [0,1] | Matches MCTS reward assumption |
| Hash | FNV-1a with lazy cache | O(N) per recompute, not O(N²) |
| Max corners | 16 (4 bits) | UCB convergence sweet spot per literature |

---

## Fixes vs. Original Spec

| Issue | Spec (buggy) | Implementation (fixed) |
|-------|-------------|------------------------|
| **HPWL incremental update** | `compute_hpwl_after()` **replaced** `current_hpwl_` with only nets touching the new module — discarding all previous contributions | Computes **delta** (`new_hpwl_net - old_hpwl_net`) and adds to `current_hpwl_` |
| **`P3ESWSMCTS` template args** | Passed 3 args `<State, Action, greedy_rollout>` | Correct 2 args `<State, Action>` (matches actual API) |
| **`MCTSConfig` field name** | `cfg.exploration_constant` | `cfg.c_puct` (matches actual struct) |
| **`from_bookshelf` declaration** | Defined in `.hpp` without class declaration | Added `static FloorplanState from_bookshelf(...)` to class |
| **Catch2 `REQUIRE_FALSE` with `&&`** | `REQUIRE_FALSE(x_overlap && y_overlap)` | Extracted to bool variable first (Catch2 v3 limitation) |

---

## What's Still TODO (Phase 1.5 / Phase 2)

1. **Test datasets** — No `.aux`/`.nodes`/`.nets` files yet; `from_bookshelf` is unexercised
2. **`compute_w_max()` is loose** — Uses theoretical upper bound; should be replaced with greedy baseline run
3. **Module rotation** — `Placement::rotated` exists but is always `false`
4. **Pin-aware HPWL** — Currently uses module-center proxy; real pin offsets from `.nets` are parsed but unused
5. **Rollout engine integration** — `RolloutEngine::simulate()` in `parallel-mcts` does pure random; greedy/ε-greedy provided but not wired into MCTS template

---

## Dependencies

- [parallel-mcts](https://github.com/tang-dynasty/parallel-mcts) — P3-ES-WS-MCTS engine (git submodule)
- C++20
- CMake ≥ 3.20
- Catch2 v3.4.0 (fetched automatically by CMake)
