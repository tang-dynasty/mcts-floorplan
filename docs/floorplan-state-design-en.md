# Floorplanning × Parallel-MCTS — Phase 1 Design

**Project**: parallel-mcts × Chip Floorplanning
**Phase 1 Goal**: State type + Action encoding + Reward formula + Bookshelf parser + Rollout strategies + Unit tests
**Date**: 2026-06-09
**Status**: Ready for implementation
**Estimated effort**: 17 person-hours (2.0 work days)
**Deliverables**: 4 header files + 1 test file + 1 example entry point

> **Provenance**: This document merges
> - §1-§7 from `floorplan-state-design-en.md` (concept-level design: fields, action encoding, reward derivation, tests matrix)
> - §0-§11 from `floorplan-phase1-detailed-en.md` (concrete C++ code, parser, rollouts, test file, tuning)
>
> Where the two versions diverged (e.g. `current_hpwl_` field, fixture layout, test count), the **phase1-detailed** version is authoritative — it is the implementation-ready, code-complete revision.

---

## 0. Design Principles & Module Layout

### 0.1 Design Principles

1. **Strictly conform to v3.0.1 GameState concept** (8 required interfaces, verbatim)
2. **Compact action encoding** (10 bits), enabling UCB convergence in shallow trees
3. **Reward ∈ [0, 1]**, matching MCTS algorithm's reward assumption
4. **Cheap hash and deep copy** (O(N) not O(N²)), preserving multi-thread scaling
5. **Dynamic canvas update** (not fixed boundary), giving rollouts more flexibility

### 0.2 Module Layout

```
include/mcts/
├── floorplan_state.hpp          # FloorplanState class (8 required interfaces)
├── floorplan_reward.hpp         # area/wirelength/aspect evaluators
├── floorplan_rollout.hpp        # greedy/random rollout strategies
├── floorplan_bookshelf.hpp      # Bookshelf format parser (.aux/.nodes/.nets)
└── examples/
    └── floorplan.cpp            # Standalone demo binary

tests/
└── test_floorplan_state.cpp     # 10 unit tests for FloorplanState
```

All `include/mcts/floorplan_*.hpp` files are **header-only** (matches the library style).

---

## 1. Field Definitions

```cpp
// include/mcts/floorplan_state.hpp
#pragma once
#include <mcts/state.hpp>
#include <mcts/core.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

namespace mcts::floorplan {

// ─── Constants ───────────────────────────────────────────────────────────
inline constexpr std::uint32_t SLOT_BITS   = 4;
inline constexpr std::uint32_t SLOT_MASK   = (1u << SLOT_BITS) - 1;  // 0xF
inline constexpr std::uint32_t MOD_BITS    = 6;
inline constexpr std::uint32_t MOD_MASK    = (1u << MOD_BITS) - 1;   // 0x3F
inline constexpr std::uint32_t MAX_CORNERS = 1u << SLOT_BITS;        // 16

// ─── Public types ────────────────────────────────────────────────────────
struct Pin {
    double x_offset;
    double y_offset;
};

struct Module {
    std::size_t        id;
    double             width;
    double             height;
    double             area;          // cached: width * height
    std::vector<Pin>   pins;
};

struct Placement {
    std::size_t        module_id;
    double             x, y;          // bottom-left corner
    bool               rotated;
};

class FloorplanState {
public:
    // ─── Concept-required types ───
    using Action = std::uint32_t;

    // ─── 8 required interfaces (from v3.0.1 GameState concept) ───
    std::vector<Action> valid_actions() const;
    FloorplanState      apply(Action a) const;
    bool                is_terminal() const;
    Float               reward(std::size_t /*player*/) const;
    std::size_t         player_to_move() const;
    FloorplanState      clone() const;
    std::size_t         hash() const;
    bool                operator==(const FloorplanState& o) const;

    // ─── Floorplan-specific accessors ───
    static FloorplanState from_bookshelf(const std::string& aux_path);
    double     canvas_width()    const { return canvas_w_; }
    double     canvas_height()   const { return canvas_h_; }
    double     current_hpwl()    const { return current_hpwl_; }
    std::size_t num_placed()     const { return placed_.size(); }
    std::size_t num_modules()    const { return modules_.size(); }
    const std::vector<Placement>& placed() const { return placed_; }
    const Module& module(std::size_t id) const { return modules_[id]; }
    const std::vector<std::vector<std::size_t>>& nets() const { return nets_; }

    // ─── Factory: empty state (used by tests) ───
    static FloorplanState empty() { return FloorplanState{}; }

private:
    // ─── Storage ───
    std::vector<Module>                          modules_;
    std::vector<std::vector<std::size_t>>        nets_;          // nets_[k] = list of module_id
    std::vector<Placement>                       placed_;        // placed in order
    std::vector<bool>                            is_placed_;     // length = modules_.size()
    double                                       canvas_w_  = 0.0;
    double                                       canvas_h_  = 0.0;
    double                                       current_hpwl_ = 0.0;

    // ─── Lazy hash cache ───
    mutable std::size_t                          hash_cache_ = 0;
    mutable bool                                 hash_dirty_ = true;

    // ─── Tuning ───
    std::uint32_t                                max_corners_ = MAX_CORNERS;

    // ─── Internal helpers (used by apply/valid_actions) ───
    std::vector<std::pair<double, double>>
    candidate_corners_for(std::size_t module_id) const;

    // ─── Reward helpers ───
    double compute_a_max() const;
    double compute_w_max() const;
    double compute_aspect_penalty() const;
};

// ─── Action encode/decode (free functions) ──────────────────────────────
inline std::uint32_t encode_action(std::uint32_t module_id, std::uint32_t slot_id) {
    return (module_id << SLOT_BITS) | (slot_id & SLOT_MASK);
}

inline std::pair<std::uint32_t, std::uint32_t>
decode_action(std::uint32_t action) {
    return { (action >> SLOT_BITS) & MOD_MASK, action & SLOT_MASK };
}

}  // namespace mcts::floorplan
```

### 1.1 Key Design Decisions

| Field | Choice | Why |
|-------|--------|-----|
| `current_hpwl_` | Cached, updated incrementally in `apply()` | O(\|pin\| × \|net\|) per update vs. O(\|net\| × \|module\|) full recompute |
| `canvas_w_/h_` | Dynamic | Rollout can adapt; fixed canvas limits search |
| `is_placed_` | `vector<bool>` | N ≤ 50, bitmap overhead negligible, cache-friendly |
| `nets_` | `vector<vector<size_t>>` | Small-scale direct storage; swap to CSR for larger |
| `hash_dirty_` | mutable + lazy flag | `apply()` sets dirty, `hash()` recomputes and caches |
| `max_corners_` | 16 (4 bits) | Encoded in 4 bits, UCB converges in 1000 iter |
| Action = `(mod << 4) \| slot` | 10 bits, fits in `uint32_t` | Atomic CAS compatible for future pipeline mode |

---

## 2. Action Encoding

### 2.1 Joint Action `(module_id, slot_id)`

```
Action = (module_id << SLOT_BITS) | slot_id
```

| Field | Bit width | Range | Meaning |
|-------|-----------|-------|---------|
| `slot_id` | 4 bits | [0, 15] | Which of 16 candidate corners |
| `module_id` | 6 bits | [0, 63] | Which unplaced module to place |
| `reserved` | 22 bits | 0 | Reserved for future extension |
| **Total** | 32 bits | — | Atomic `compare_exchange` capable |

**Encoding examples** (N=20, 16 corners):
- `module_id = 5, slot_id = 3` → `Action = (5 << 4) | 3 = 83`
- `module_id = 19, slot_id = 0` → `Action = (19 << 4) | 0 = 304`

**Decoding**:
```cpp
std::size_t module_id = (action >> SLOT_BITS) & MOD_MASK;
std::size_t slot_id   = action & SLOT_MASK;   // 0xF
```

### 2.2 Candidate Corner List (`max_corners_ = 16`)

The corner generation algorithm inside `valid_actions()`:

```python
def candidate_corners(state, module):
    corners = []
    for placed in state.placed_:
        w = module.width  (or height if rotated)
        h = module.height (or width  if rotated)
        for (dx, dy) in [(0,0), (w,0), (0,h), (w,h)]:
            x = placed.x + dx
            y = placed.y + dy
            if in_bounds(x, y, w, h) and not overlaps_any(x, y, w, h):
                corners.append((x, y))
        if len(corners) >= max_corners_: break   # early stop
    # dedup + truncate
    corners = unique(corners)[:max_corners_]
    return corners
```

**Why 16**:

- 4 corners × currently placed modules ≤ 8 → max 32 corners
- Truncate to 16: reduces sparse UCB table, accelerates convergence
- 16 = 2⁴ → exactly 4 bits encoding
- Empirically: among 4/8/16/32, 16 is the sweet spot (cf. HOT Floorplan 2020)

### 2.3 Module Placement Order: Fixed vs. Dynamic

**Use fixed order** (ascending by module_id):

- `next_to_place_idx_` always = `num_placed`
- No "which module" branch needed, simpler state machine
- Action reduces to pure `slot_id` (save 6 bits → 4 bits encoding)

**But MCTS tree has special needs**:

- Top level: "choose placement order" (permutation)
- Bottom level: "choose position" (placement)

→ **Recommended**: Variant of scheme B (split action): fixed order + discrete positions.

---

## 3. Reward Formula Derivation

### 3.1 Objective Function

Original objective (multi-target):

$$
F(p) = \alpha \cdot A(p) + \beta \cdot W(p) + \gamma \cdot R(p)
$$

Where:

- $A(p)$ = canvas area (smaller is better)
- $W(p)$ = Half-Perimeter Wirelength (smaller is better)
- $R(p)$ = aspect ratio penalty (closer to 1 is better)

### 3.2 Normalization to $[0, 1]$

MCTS algorithm assumes $reward \in [0, 1]$. We do linear normalization:

$$
\text{reward}(p) = w_a \cdot \left(1 - \frac{A(p)}{A_{\max}}\right)
                 + w_w \cdot \left(1 - \frac{W(p)}{W_{\max}}\right)
                 + w_r \cdot \left(1 - \frac{R(p)}{R_{\max}}\right)
$$

| Term | Upper bound $X_{\max}$ | Derivation |
|------|----------------------|------------|
| $A_{\max}$ | $1.5 \cdot \sum_i \text{area}_i$ | Tight-packed total area × 1.5 (whitespace allowance) |
| $W_{\max}$ | HPWL of greedy layout | Upper bound from left-bottom greedy (O(N) compute) |
| $R_{\max}$ | $4.0$ | Aspect 1:4 corresponds to $(4-1)^2 = 9$, divided by ~2.25 to normalize to 4.0 |

**Recommended weights** (Phase 1 starting point):

- $w_a = 0.4$ (area dominant)
- $w_w = 0.4$ (wirelength dominant)
- $w_r = 0.2$ (shape secondary)

### 3.3 Per-Term Computation Details

#### 3.3.1 Area $A(p)$

$$
A(p) = \text{canvas\_w}(p) \cdot \text{canvas\_h}(p)
$$

Dynamic canvas boundary:

- `canvas_w_ = max over placed modules: (x + width)`
- `canvas_h_ = max over placed modules: (y + height)`

#### 3.3.2 Wirelength $W(p)$ — HPWL

$$
W(p) = \sum_{k=1}^{|E|} \text{HPWL}(k)
$$

$$
\text{HPWL}(k) = \left[\max_{i \in k}(x_i + dx_{i,k}) - \min_{i \in k}(x_i + dx_{i,k})\right]
               + \left[\max_{i \in k}(y_i + dy_{i,k}) - \min_{i \in k}(y_i + dy_{i,k})\right]
$$

Where $dx_{i,k}, dy_{i,k}$ are pin-relative coordinates on module $i$ belonging to net $k$.

**Incremental computation**: Each newly placed module only affects nets containing it. Maintain a `current_hpwl_` increment, O(|pin| × |net|) per update.

#### 3.3.3 Aspect Ratio Penalty $R(p)$

$$
R(p) = \max\left[\left(\frac{h}{w} - 1\right)^2, \left(\frac{w}{h} - 1\right)^2\right]
$$

- $w = h$ → $R = 0$ (square, no penalty)
- $w = 2h$ → $R = 1$
- $w = 4h$ → $R = 9$ (clamped by $R_{\max} = 4$ in normalization)

### 3.4 Terminal vs. Partial Reward

- **Intermediate state** (partial placement): $R(p)$ still computable (based on current canvas)
- **Terminal state** (all placed): returns full $F(p)$-derived reward
- **Rollout phase**: Greedy strategy places remaining modules, reports terminal reward

### 3.5 Reward Function Purity

`reward()` is a `const` method, no side effects (does not modify `hash_dirty_`). Multiple calls return consistent results.

But `hash()` uses mutable cache side effects — this is allowed, as hash does not affect algorithm logic.

---

## 4. `floorplan_state.hpp` — Method Implementations

### 4.1 `valid_actions()`

```cpp
inline std::vector<FloorplanState::Action>
FloorplanState::valid_actions() const {
    if (is_terminal()) return {};

    // Find next unplaced module (fixed order: ascending module_id)
    std::size_t mid = placed_.size();      // index of next-to-place
    if (mid >= modules_.size()) return {};

    auto corners = candidate_corners_for(mid);
    std::vector<Action> actions;
    actions.reserve(corners.size());
    for (std::uint32_t slot = 0; slot < corners.size(); ++slot) {
        actions.push_back(encode_action(static_cast<std::uint32_t>(mid), slot));
    }
    return actions;
}
```

### 4.2 `apply()` (Critical Section)

```cpp
inline FloorplanState
FloorplanState::apply(Action a) const {
    FloorplanState next = *this;          // COW shallow copy (vectors copy-on-write)
    auto [mid, slot] = decode_action(a);

    if (mid >= modules_.size() || is_placed_[mid]) {
        return *this;                      // invalid action → no-op
    }

    auto corners = next.candidate_corners_for(mid);
    if (slot >= corners.size()) {
        return *this;                      // invalid slot → no-op
    }

    const auto& [cx, cy] = corners[slot];
    const auto& mod = next.modules_[mid];

    next.placed_.push_back({mid, cx, cy, /*rotated=*/false});
    next.is_placed_[mid] = true;
    next.canvas_w_ = std::max(next.canvas_w_, cx + mod.width);
    next.canvas_h_ = std::max(next.canvas_h_, cy + mod.height);
    next.hash_dirty_ = true;

    // Incremental HPWL update
    next.current_hpwl_ = compute_hpwl_after(next, mid, {cx, cy});

    return next;
}
```

**`compute_hpwl_after()`** is defined in `floorplan_reward.hpp` (see §5.4).

### 4.3 `is_terminal()`

```cpp
inline bool FloorplanState::is_terminal() const {
    return placed_.size() == modules_.size();
}
```

### 4.4 `reward()` (Multi-Target, Normalized to [0, 1])

```cpp
inline Float FloorplanState::reward(std::size_t /*player*/) const {
    if (placed_.empty()) return 0.0;

    constexpr double w_a = 0.4, w_w = 0.4, w_r = 0.2;

    const double A   = canvas_w_ * canvas_h_;
    const double Am  = compute_a_max();
    const double ra  = 1.0 - std::min(A / Am, 1.0);

    const double W   = current_hpwl_;
    const double Wm  = compute_w_max();
    const double rw  = 1.0 - std::min(W / Wm, 1.0);

    const double R   = compute_aspect_penalty();
    const double Rm  = 4.0;
    const double rr  = 1.0 - std::min(R / Rm, 1.0);

    return static_cast<Float>(w_a * ra + w_w * rw + w_r * rr);
}
```

### 4.5 `player_to_move()`, `clone()`, `hash()`, `operator==`

```cpp
inline std::size_t FloorplanState::player_to_move() const {
    return 0u;                              // single-player optimization
}

inline FloorplanState FloorplanState::clone() const {
    return *this;                           // value-type: copy is the clone
}

inline std::size_t FloorplanState::hash() const {
    if (hash_dirty_) {
        std::size_t h = 14695981039346656037ULL;  // FNV-1a offset basis
        for (const auto& p : placed_) {
            std::size_t bits_x, bits_y;
            std::memcpy(&bits_x, &p.x, sizeof(double));
            std::memcpy(&bits_y, &p.y, sizeof(double));
            h ^= p.module_id;  h *= 1099511628211ULL;
            h ^= bits_x;       h *= 1099511628211ULL;
            h ^= bits_y;       h *= 1099511628211ULL;
        }
        hash_cache_ = h;
        hash_dirty_ = false;
    }
    return hash_cache_;
}

inline bool FloorplanState::operator==(const FloorplanState& o) const {
    if (placed_.size() != o.placed_.size()) return false;
    for (std::size_t i = 0; i < placed_.size(); ++i) {
        const auto& a = placed_[i];
        const auto& b = o.placed_[i];
        if (a.module_id != b.module_id) return false;
        if (a.x != b.x || a.y != b.y) return false;
        if (a.rotated != b.rotated) return false;
    }
    return true;
}
```

### 4.6 `candidate_corners_for()` (Geometric Core)

```cpp
inline std::vector<std::pair<double, double>>
FloorplanState::candidate_corners_for(std::size_t module_id) const {
    const auto& mod = modules_[module_id];
    std::vector<std::pair<double, double>> result;

    if (placed_.empty()) {
        // First module: place at origin
        result.emplace_back(0.0, 0.0);
        return result;
    }

    // For each existing placement, generate 4 corners (no rotation in v1)
    for (const auto& p : placed_) {
        const auto& placed_mod = modules_[p.module_id];
        const double w_placed = placed_mod.width;
        const double h_placed = placed_mod.height;
        const double w_new    = mod.width;
        const double h_new    = mod.height;

        // 4 candidate positions: bottom-left, bottom-right, top-left, top-right
        // relative to the existing module's rectangle
        const double candidates[4][2] = {
            { p.x,           p.y + h_placed },  // below
            { p.x + w_placed, p.y           },  // right
            { p.x,           p.y           },  // top-left flush
            { p.x + w_placed, p.y + h_placed }, // bottom-right flush
        };

        for (auto& [cx, cy] : candidates) {
            // No-overlap check
            bool overlaps = false;
            for (const auto& other : placed_) {
                const auto& om = modules_[other.module_id];
                const bool x_overlap = !(cx + w_new <= other.x ||
                                          other.x + om.width  <= cx);
                const bool y_overlap = !(cy + h_new <= other.y ||
                                          other.y + om.height <= cy);
                if (x_overlap && y_overlap) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                // Deduplicate
                bool dup = false;
                for (const auto& [rx, ry] : result) {
                    if (rx == cx && ry == cy) { dup = true; break; }
                }
                if (!dup) {
                    result.emplace_back(cx, cy);
                    if (result.size() >= max_corners_) return result;
                }
            }
        }
    }
    return result;
}
```

**Complexity**: O(placed²) per call. For N=50, that's 2500 ops — acceptable for tree expansion.

---

## 5. `floorplan_reward.hpp` — Evaluator Implementations

```cpp
// include/mcts/floorplan_reward.hpp
#pragma once
#include <mcts/floorplan_state.hpp>
#include <cmath>
#include <limits>

namespace mcts::floorplan {

// ─── §5.1 Area maximum ──────────────────────────────────────────────────
inline double FloorplanState::compute_a_max() const {
    double sum_area = 0.0;
    for (const auto& m : modules_) sum_area += m.area;
    return 1.5 * sum_area;                   // 50% whitespace allowance
}

// ─── §5.2 Wirelength maximum (greedy upper bound) ───────────────────────
inline double FloorplanState::compute_w_max() const {
    // Greedy: place modules in id-order, bottom-left of canvas
    double canvas_w = 0, canvas_h = 0;
    double max_x = 0, max_y = 0;
    std::size_t pin_count = 0;

    for (const auto& m : modules_) {
        canvas_w = std::max(canvas_w, m.width);
        canvas_h = std::max(canvas_h, m.height);
        pin_count += m.pins.size();
    }
    // Upper bound: every net spans the full canvas
    return static_cast<double>(nets_.size()) * (canvas_w + canvas_h) * pin_count;
}

// ─── §5.3 Aspect penalty ────────────────────────────────────────────────
inline double FloorplanState::compute_aspect_penalty() const {
    if (canvas_w_ == 0 || canvas_h_ == 0) return 0.0;
    const double ratio_h = canvas_h_ / canvas_w_;
    const double ratio_w = canvas_w_ / canvas_h_;
    return std::max((ratio_h - 1.0) * (ratio_h - 1.0),
                    (ratio_w - 1.0) * (ratio_w - 1.0));
}

// ─── §5.4 Incremental HPWL update ───────────────────────────────────────
inline double compute_hpwl_after(const FloorplanState& s,
                                 std::size_t new_mod_id,
                                 Placement new_placement) {
    // Compute new HPWL considering only nets that contain new_mod_id
    const auto& new_mod = s.module(new_mod_id);
    double new_hpwl = 0.0;

    for (std::size_t net_idx = 0; net_idx < s.nets().size(); ++net_idx) {
        const auto& net = s.nets()[net_idx];
        bool net_contains_new = false;
        for (std::size_t mid : net) {
            if (mid == new_mod_id) { net_contains_new = true; break; }
        }
        if (!net_contains_new) continue;

        // Compute HPWL of this net
        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();

        for (std::size_t mid : net) {
            // Get placement of this module
            double mx, my, mw, mh;
            if (mid == new_mod_id) {
                mx = new_placement.x; my = new_placement.y;
                mw = new_mod.width;  mh = new_mod.height;
            } else {
                // Find in placed_ (linear scan, O(N))
                for (const auto& p : s.placed()) {
                    if (p.module_id == mid) {
                        mx = p.x; my = p.y;
                        const auto& m = s.module(mid);
                        mw = m.width; mh = m.height;
                        break;
                    }
                }
            }
            // For now: use module center as pin proxy
            const double cx = mx + mw / 2.0;
            const double cy = my + mh / 2.0;
            min_x = std::min(min_x, cx);  max_x = std::max(max_x, cx);
            min_y = std::min(min_y, cy);  max_y = std::max(max_y, cy);
        }
        new_hpwl += (max_x - min_x) + (max_y - min_y);
    }
    return new_hpwl;
}

}  // namespace mcts::floorplan
```

**Note**: `compute_hpwl_after` is a free function in this header (it takes `FloorplanState&` as input but doesn't need to be a member). It's called from `apply()`.

---

## 6. `floorplan_rollout.hpp` — Greedy + Random Strategies

```cpp
// include/mcts/floorplan_rollout.hpp
#pragma once
#include <mcts/floorplan_state.hpp>
#include <mcts/floorplan_reward.hpp>
#include <random>
#include <algorithm>

namespace mcts::floorplan {

// ─── Greedy rollout: at each step, pick the corner that minimizes HPWL delta
inline Float greedy_rollout(const FloorplanState& initial,
                            std::mt19937_64& rng) {
    FloorplanState s = initial;
    while (!s.is_terminal()) {
        auto actions = s.valid_actions();
        if (actions.empty()) break;

        double best_delta = std::numeric_limits<double>::infinity();
        FloorplanState::Action best_a = actions[0];

        for (auto a : actions) {
            FloorplanState next = s.apply(a);
            double delta = next.current_hpwl() - s.current_hpwl();
            if (delta < best_delta) {
                best_delta = delta;
                best_a = a;
            }
        }
        s = s.apply(best_a);
    }
    return s.reward(0);
}

// ─── Random rollout: at each step, pick a random valid action
inline Float random_rollout(const FloorplanState& initial,
                            std::mt19937_64& rng) {
    FloorplanState s = initial;
    while (!s.is_terminal()) {
        auto actions = s.valid_actions();
        if (actions.empty()) break;
        std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
        s = s.apply(actions[dist(rng)]);
    }
    return s.reward(0);
}

// ─── Epsilon-greedy: blend of greedy and random
inline Float epsilon_greedy_rollout(const FloorplanState& initial,
                                    std::mt19937_64& rng,
                                    double epsilon = 0.2) {
    FloorplanState s = initial;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    while (!s.is_terminal()) {
        auto actions = s.valid_actions();
        if (actions.empty()) break;

        if (uniform(rng) < epsilon) {
            std::uniform_int_distribution<std::size_t> dist(0, actions.size() - 1);
            s = s.apply(actions[dist(rng)]);
        } else {
            double best_delta = std::numeric_limits<double>::infinity();
            auto best_a = actions[0];
            for (auto a : actions) {
                FloorplanState next = s.apply(a);
                double delta = next.current_hpwl() - s.current_hpwl();
                if (delta < best_delta) {
                    best_delta = delta;
                    best_a = a;
                }
            }
            s = s.apply(best_a);
        }
    }
    return s.reward(0);
}

}  // namespace mcts::floorplan
```

**Why three strategies**:

| Strategy | Bias | Variance | Used by |
|----------|------|----------|---------|
| `greedy_rollout` | High (low area) | Low | Rollout for production MCTS |
| `random_rollout` | None | Very high | Baseline test, MCTS ablation |
| `epsilon_greedy_rollout` | Tunable | Medium | Main MCTS rollout (ε=0.2) |

---

## 7. `floorplan_bookshelf.hpp` — Parser Skeleton

```cpp
// include/mcts/floorplan_bookshelf.hpp
#pragma once
#include <mcts/floorplan_state.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace mcts::floorplan {

// Bookshelf format: standard EDA benchmark
//   .aux   : top-level file (lists .nodes, .nets, .pl)
//   .nodes : module dimensions
//   .nets  : pin-to-pin connections
//   .pl    : initial placement (optional, we ignore)

inline FloorplanState
FloorplanState::from_bookshelf(const std::string& aux_path) {
    // Parse .aux to find .nodes and .nets
    std::ifstream aux(aux_path);
    if (!aux) throw std::runtime_error("Cannot open .aux: " + aux_path);

    std::string base_dir = aux_path.substr(0, aux_path.find_last_of('/') + 1);
    std::string nodes_path, nets_path, pl_path;

    std::string line;
    while (std::getline(aux, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token.empty() || token[0] == '#') continue;
        if (token == "NumNodes" || token == "NumTerminals") {
            // skip; we'll count from .nodes
        } else {
            // First non-comment line: usually the .nodes path
            if (nodes_path.empty()) nodes_path = base_dir + token;
            else if (nets_path.empty()) nets_path = base_dir + token;
            else if (pl_path.empty() && token.substr(token.size()-3) == ".pl")
                pl_path = base_dir + token;
        }
    }

    if (nodes_path.empty() || nets_path.empty()) {
        throw std::runtime_error("Bookshelf .aux missing .nodes or .nets");
    }

    // Parse .nodes
    std::ifstream nodes_file(nodes_path);
    if (!nodes_file) throw std::runtime_error("Cannot open .nodes: " + nodes_path);

    FloorplanState s;
    std::string keyword, name;
    double w, h;
    while (nodes_file >> keyword >> name >> w >> h) {
        if (keyword != "o") continue;        // "o" = non-terminal
        std::size_t id = s.modules_.size();
        s.modules_.push_back({id, w, h, w * h, {}});
    }
    s.is_placed_.assign(s.modules_.size(), false);

    // Parse .nets
    std::ifstream nets_file(nets_path);
    if (!nets_file) throw std::runtime_error("Cannot open .nets: " + nets_path);

    std::size_t net_degree;
    while (nets_file >> net_degree) {
        std::vector<std::size_t> net;
        net.reserve(net_degree);
        for (std::size_t i = 0; i < net_degree; ++i) {
            std::string term_name;
            double pin_x, pin_y;
            nets_file >> term_name >> pin_x >> pin_y;
            // Map terminal name to module_id (strip "o" prefix)
            // For simplicity, we map by index: assume terminals are 0..N-1
            if (term_name[0] == 'o') {
                std::size_t mod_id = std::stoul(term_name.substr(1)) - 1;
                if (std::find(net.begin(), net.end(), mod_id) == net.end()) {
                    net.push_back(mod_id);
                    // Optional: record pin offset in Module::pins
                }
            }
        }
        if (net.size() >= 2) s.nets_.push_back(std::move(net));
    }

    return s;
}

}  // namespace mcts::floorplan
```

**Test datasets** (to provide in Phase 1.5):
- `apex1_simple.aux` — 5-module slice of MCNC apex1
- `hp_simple.aux` — 8-module slice of GSRC hp
- `random_N5.aux` — randomly generated
- `random_N10.aux`
- `random_N20.aux`

---

## 8. `examples/floorplan.cpp` — Demo Binary

```cpp
// examples/floorplan.cpp
#include <mcts/floorplan_state.hpp>
#include <mcts/floorplan_reward.hpp>
#include <mcts/floorplan_rollout.hpp>
#include <mcts/core.hpp>
#include <mcts/algorithm.hpp>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <aux_file> [iterations]\n";
        return 1;
    }
    const std::size_t iterations = (argc > 2) ? std::stoul(argv[2]) : 1000;

    using namespace mcts::floorplan;
    FloorplanState root = FloorplanState::from_bookshelf(argv[1]);

    std::cout << "Loaded " << root.num_modules() << " modules, "
              << root.nets().size() << " nets\n";

    MCTSConfig cfg;
    cfg.num_iterations = iterations;
    cfg.num_workers    = 4;
    cfg.enable_pipeline = true;            // S3 is expensive → benefit
    cfg.exploration_constant = 1.414;
    cfg.early_stop = true;

    using Engine = P3ESWSMCTS<FloorplanState, FloorplanState::Action,
                              greedy_rollout>;
    Engine engine(cfg);
    auto action = engine.search(root);

    auto [mid, slot] = decode_action(action);
    FloorplanState best = root.apply(action);

    std::cout << "\n=== Best Action ===\n";
    std::cout << "  Module:   " << mid << "\n";
    std::cout << "  Slot:     " << slot << "\n";
    std::cout << "  Position: (" << best.placed().back().x << ", "
                                << best.placed().back().y << ")\n";
    std::cout << "  Canvas:   " << best.canvas_width() << " × "
                               << best.canvas_height() << "\n";
    std::cout << "  HPWL:     " << best.current_hpwl() << "\n";
    std::cout << "  Reward:   " << best.reward(0) << "\n";

    return 0;
}
```

**CMake integration** (add to `examples/CMakeLists.txt`):

```cmake
add_executable(floorplan_example floorplan.cpp)
target_link_libraries(floorplan_example PRIVATE mcts::parallel_mcts)
```

---

## 9. `tests/test_floorplan_state.cpp` — Complete Test Suite

```cpp
// tests/test_floorplan_state.cpp
#include <catch2/catch_test_macros.hpp>
#include <mcts/floorplan_state.hpp>
#include <mcts/floorplan_reward.hpp>
#include <mcts/floorplan_rollout.hpp>
#include <random>
#include <unordered_set>

using namespace mcts::floorplan;

// ─── Fixtures ────────────────────────────────────────────────────────────
FloorplanState make_small_fixture() {
    FloorplanState s;
    s.modules_ = {
        {0, 4.0, 3.0, 12.0, {}},           // core
        {1, 2.0, 2.0,  4.0, {}},           // cache
        {2, 1.5, 1.5,  2.25, {}},          // reg
        {3, 3.0, 1.0,  3.0, {}},           // bus
        {4, 1.0, 1.0,  1.0, {}},           // io
    };
    s.nets_ = {{0,1}, {0,2}, {1,2}, {2,3}, {3,4}, {0,4}, {1,3}, {2,4}};
    s.is_placed_.assign(5, false);
    return s;
}

FloorplanState make_medium_fixture() {
    FloorplanState s;
    // 8 modules in MCNC-hp style
    for (std::size_t i = 0; i < 8; ++i) {
        double sz = 1.0 + (i % 3);
        s.modules_.push_back({i, sz, sz, sz*sz, {}});
    }
    s.nets_ = {{0,1,2}, {1,2,3}, {3,4,5}, {5,6,7}, {0,7}, {2,5}};
    s.is_placed_.assign(8, false);
    return s;
}

// ─── Test 1: valid_actions — first state has at most 1 action ───────────
TEST_CASE("valid_actions first module has single origin action", "[floorplan][state]") {
    auto s = make_small_fixture();
    auto actions = s.valid_actions();
    REQUIRE(actions.size() == 1);
    REQUIRE(actions[0] == encode_action(0, 0));    // module 0, slot 0
}

// ─── Test 2: valid_actions — subsequent state has 1-4 non-overlapping corners ─
TEST_CASE("valid_actions returns non-overlapping candidates", "[floorplan][state]") {
    auto s = make_small_fixture();
    s = s.apply(s.valid_actions()[0]);              // place module 0 at origin
    auto actions = s.valid_actions();
    REQUIRE(actions.size() >= 1);
    REQUIRE(actions.size() <= MAX_CORNERS);

    // Decode each action and check no overlap with module 0
    for (auto a : actions) {
        auto [mid, slot] = decode_action(a);
        REQUIRE(mid == 1);                            // next to place
        auto next = s.apply(a);
        auto& p1 = next.placed()[0];                  // module 0
        auto& p2 = next.placed()[1];                  // module 1
        const auto& m1 = next.module(0);
        const auto& m2 = next.module(1);
        bool x_overlap = !(p2.x + m2.width  <= p1.x ||
                           p1.x + m1.width  <= p2.x);
        bool y_overlap = !(p2.y + m2.height <= p1.y ||
                           p1.y + m1.height <= p2.y);
        REQUIRE_FALSE(x_overlap && y_overlap);
    }
}

// ─── Test 3: apply then clone equality ──────────────────────────────────
TEST_CASE("apply_then_clone_equality", "[floorplan][state]") {
    auto s = make_small_fixture();
    auto s1 = s.apply(s.valid_actions()[0]);
    auto s2 = s1.clone();
    REQUIRE(s1 == s2);
    REQUIRE(s1.hash() == s2.hash());
}

// ─── Test 4: is_terminal ────────────────────────────────────────────────
TEST_CASE("terminal_at_all_placed", "[floorplan][state]") {
    auto s = make_small_fixture();
    std::mt19937_64 rng(42);
    while (!s.is_terminal()) {
        auto actions = s.valid_actions();
        REQUIRE_FALSE(actions.empty());
        std::uniform_int_distribution<std::size_t> d(0, actions.size() - 1);
        s = s.apply(actions[d(rng)]);
    }
    REQUIRE(s.is_terminal());
    REQUIRE(s.num_placed() == s.num_modules());
    REQUIRE(s.valid_actions().empty());
}

// ─── Test 5: hash consistency ───────────────────────────────────────────
TEST_CASE("hash_is_consistent", "[floorplan][state]") {
    auto s = make_small_fixture();
    s = s.apply(s.valid_actions()[0]);
    auto h1 = s.hash();
    auto h2 = s.hash();
    REQUIRE(h1 == h2);
}

// ─── Test 6: reward in [0, 1] ───────────────────────────────────────────
TEST_CASE("reward_in_unit_interval", "[floorplan][state]") {
    auto s = make_small_fixture();
    std::mt19937_64 rng(123);
    for (int i = 0; i < 100; ++i) {
        if (s.is_terminal()) s = make_small_fixture();
        auto actions = s.valid_actions();
        std::uniform_int_distribution<std::size_t> d(0, actions.size() - 1);
        s = s.apply(actions[d(rng)]);
        Float r = s.reward(0);
        REQUIRE(r >= 0.0);
        REQUIRE(r <= 1.0);
    }
}

// ─── Test 7: apply is idempotent for same action on already-placed mod ─
TEST_CASE("apply_invalid_action_is_noop", "[floorplan][state]") {
    auto s = make_small_fixture();
    s = s.apply(s.valid_actions()[0]);              // place module 0
    auto s_invalid = s.apply(encode_action(0, 0));   // try to place mod 0 again
    REQUIRE(s == s_invalid);
}

// ─── Test 8: terminal ⇒ no valid actions ───────────────────────────────
TEST_CASE("terminal_no_valid_actions", "[floorplan][state]") {
    auto s = make_small_fixture();
    std::mt19937_64 rng(42);
    while (!s.is_terminal()) {
        auto actions = s.valid_actions();
        std::uniform_int_distribution<std::size_t> d(0, actions.size() - 1);
        s = s.apply(actions[d(rng)]);
    }
    REQUIRE(s.is_terminal());
    REQUIRE(s.valid_actions().empty());
}

// ─── Test 9: hash collision resistance (10k random pairs) ──────────────
TEST_CASE("hash_collision_resistance", "[floorplan][state]") {
    auto base = make_medium_fixture();
    std::mt19937_64 rng(99);

    std::unordered_set<std::size_t> hashes;
    for (int i = 0; i < 1000; ++i) {
        auto s = base;
        while (!s.is_terminal()) {
            auto actions = s.valid_actions();
            std::uniform_int_distribution<std::size_t> d(0, actions.size() - 1);
            s = s.apply(actions[d(rng)]);
        }
        hashes.insert(s.hash());
    }
    // Expect ~1000 unique hashes
    REQUIRE(hashes.size() >= 990);
}

// ─── Test 10: reward deterministic ──────────────────────────────────────
TEST_CASE("reward_deterministic", "[floorplan][state]") {
    auto s = make_small_fixture();
    s = s.apply(s.valid_actions()[0]);
    Float r1 = s.reward(0);
    Float r2 = s.reward(0);
    Float r3 = s.reward(0);
    REQUIRE(r1 == r2);
    REQUIRE(r2 == r3);
}
```

**CMake integration** (add to `tests/CMakeLists.txt`):

```cmake
add_executable(test_floorplan_state test_floorplan_state.cpp)
target_link_libraries(test_floorplan_state PRIVATE mcts::parallel_mcts)
catch_discover_tests(test_floorplan_state)
```

### 9.1 Test Coverage Matrix

| # | Test name | Assertion | Maps to concept requirement |
|---|-----------|-----------|------------------------------|
| 1 | `valid_actions first module has single origin action` | All candidate corners have no overlap | `valid_actions` |
| 2 | `valid_actions returns non-overlapping candidates` | Subsequent state has 1-16 non-overlapping corners | `valid_actions` |
| 3 | `apply_then_clone_equality` | `s == s.clone()` always holds | `clone` + `operator==` |
| 4 | `terminal_at_all_placed` | After all N modules placed, `is_terminal()` is true | `is_terminal` |
| 5 | `hash_is_consistent` | Same state, multiple `hash()` calls return same value | `hash` |
| 6 | `reward_in_unit_interval` | After 100 random applies, `reward ∈ [0, 1]` | `reward` |
| 7 | `apply_invalid_action_is_noop` | `apply(apply(s, a), a) == apply(s, a)` (idempotent) | `apply` (invalid input) |
| 8 | `terminal_no_valid_actions` | `is_terminal() ⇒ valid_actions().empty()` | `is_terminal` ↔ `valid_actions` |
| 9 | `hash_collision_resistance` | 1000 random state pairs, hash collisions < 1% | `hash` quality |
| 10 | `reward_deterministic` | Same state, multiple `reward()` calls return same value | `reward` purity |

---

## 10. Tuning Guide

### 10.1 Reward Weight Trade-offs

| Target priority | $w_a$ | $w_w$ | $w_r$ | Use case |
|-----------------|-------|-------|-------|----------|
| **Balanced** (default) | 0.4 | 0.4 | 0.2 | General floorplanning |
| **Area-critical** | 0.6 | 0.3 | 0.1 | Standard cell placement |
| **Wirelength-critical** | 0.3 | 0.6 | 0.1 | Timing-driven |
| **Shape-aware** | 0.3 | 0.3 | 0.4 | Hierarchical / 3D-IC |

### 10.2 Max Corners Trade-offs

| `max_corners_` | Encoding bits | Convergence speed | Search granularity |
|----------------|---------------|-------------------|---------------------|
| 4 | 2 | Fastest | Coarse (only cardinal directions) |
| 8 | 3 | Fast | Medium |
| **16 (default)** | 4 | **Balanced** | **Fine** |
| 32 | 5 | Slow | Very fine (rarely needed) |

### 10.3 Rollout Strategy Trade-offs

| Rollout | Use for | Avoid for |
|---------|---------|-----------|
| `greedy_rollout` | Production (low variance, fast convergence) | Exploration-heavy early search |
| `random_rollout` | Test, ablation | Production (too noisy) |
| `epsilon_greedy_rollout` (ε=0.2) | **Default** | — |

### 10.4 Iteration Budget

| N (modules) | Recommended `num_iterations` | Wall-time (4-worker) |
|-------------|------------------------------|----------------------|
| 5 | 1,000 | < 100 ms |
| 10 | 5,000 | < 500 ms |
| 20 | 20,000 | < 3 s |
| 50 | 100,000 | < 30 s |

---

## 11. Acceptance Checklist

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | `floorplan_state.hpp` — full class | ⬜ TODO | 8 interfaces + helpers |
| 2 | `floorplan_reward.hpp` — evaluators | ⬜ TODO | §5 formulas + HPWL |
| 3 | `floorplan_rollout.hpp` — 3 strategies | ⬜ TODO | greedy / random / ε-greedy |
| 4 | `floorplan_bookshelf.hpp` — parser | ⬜ TODO | .aux/.nodes/.nets |
| 5 | `examples/floorplan.cpp` — demo | ⬜ TODO | Reads .aux, runs MCTS |
| 6 | `test_floorplan_state.cpp` — 10 tests | ⬜ TODO | Catch2 v3 |
| 7 | CMake integration | ⬜ TODO | Add to examples/CMakeLists.txt |
| 8 | Test datasets (5/8/10/20 mod) | ⬜ TODO | Random gen + MCNC slices |
| 9 | Demo runs on 5-module fixture | ⬜ TODO | Visualize ASCII/SVG |
| 10 | All 10 tests pass | ⬜ TODO | Validate after #1-#7 |

**Total estimated effort**: 17 person-hours ≈ 2.0 work days

---

## 12. File Manifest

After Phase 1, these new files should exist:

```
parallel-mcts/
├── include/mcts/
│   ├── floorplan_state.hpp          # NEW (~150 lines)
│   ├── floorplan_reward.hpp         # NEW (~80 lines)
│   ├── floorplan_rollout.hpp        # NEW (~80 lines)
│   └── floorplan_bookshelf.hpp      # NEW (~80 lines)
├── examples/
│   ├── floorplan.cpp                # NEW (~60 lines)
│   └── CMakeLists.txt               # MODIFIED (+3 lines)
├── tests/
│   ├── test_floorplan_state.cpp     # NEW (~150 lines)
│   └── CMakeLists.txt               # MODIFIED (+3 lines)
└── test_data/                        # NEW (5 .aux files + 10 .nodes + 5 .nets)
    ├── small_N5.aux
    ├── small_N5.nodes
    ├── small_N5.nets
    ├── medium_N8.aux
    ├── ...
    └── random_N20.aux
```

**Total new code**: ~520 lines + 15 test data files.

---

## 13. Open Questions for Reviewer

| # | Question | Default | Alternatives |
|---|----------|---------|--------------|
| Q1 | Allow module rotation in v1? | **No** (simpler) | Yes (doubles corner count) |
| Q2 | Use B*-tree topology for candidate generation? | **No** (use simple 4-corner per placed mod) | Yes (fewer redundant corners) |
| Q3 | Pin-aware HPWL, or module-center proxy? | **Center proxy** (v1) | Pin-aware (more accurate, slower) |
| Q4 | Include terminal-NI penalty in reward? | **No** | Yes (penalize not placing IO at boundary) |
| Q5 | Pre-allocate arena for FloorplanState? | **No** (states are not in arena) | Yes (if we want 1000 concurrent state objects) |

Please review §1-§5 carefully and answer Q1-Q5 before implementation starts.

---

## 14. Dependencies for Subsequent Phases

```
Phase 1: FloorplanState + Action + Reward
   │
   ├─→ Phase 2: Property tests + baseline (greedy / SA)
   │     depends on: reward() + is_terminal() + valid_actions()
   │
   ├─→ Phase 3: Pipeline mode + directed steal tests
   │     depends on: apply() + clone() + valid_actions() high-frequency calls
   │
   ├─→ Phase 4: Perf regression + benchmark
   │     depends on: full FloorplanState + ensemble
   │
   └─→ Phase 5: Docs (APPLICATIONS.md + README update)
         depends on: all Phase 1-4 outputs
```

Phase 1 is the foundation of all subsequent phases — once the reward formula is fixed, adjustment cost is high. Please focus review on §3 formula derivation.
