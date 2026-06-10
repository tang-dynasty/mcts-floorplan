#pragma once

#include "floorplan/floorplan_state.hpp"

#include <cmath>
#include <limits>

namespace floorplan {

// ─── §5.1 Area maximum ──────────────────────────────────────────────────
inline double FloorplanState::compute_a_max() const {
    double sum_area = 0.0;
    for (const auto& m : modules_) sum_area += m.area;
    return 1.5 * sum_area;                   // 50% whitespace allowance
}

// ─── §5.2 Wirelength maximum (greedy upper bound) ───────────────────────
inline double FloorplanState::compute_w_max() const {
    double max_w = 0.0, max_h = 0.0;
    for (const auto& m : modules_) {
        max_w = std::max(max_w, m.width);
        max_h = std::max(max_h, m.height);
    }
    // Upper bound: every net spans the full canvas (loose but safe)
    return static_cast<double>(nets_.size()) * (max_w + max_h) * modules_.size();
}

// ─── §5.3 Aspect penalty ────────────────────────────────────────────────
inline double FloorplanState::compute_aspect_penalty() const {
    if (canvas_w_ == 0.0 || canvas_h_ == 0.0) return 0.0;
    const double ratio_h = canvas_h_ / canvas_w_;
    const double ratio_w = canvas_w_ / canvas_h_;
    return std::max((ratio_h - 1.0) * (ratio_h - 1.0),
                    (ratio_w - 1.0) * (ratio_w - 1.0));
}

// ─── §5.4 Incremental HPWL update ───────────────────────────────────────
//
// CRITICAL FIX vs the other agent's spec:
//   The spec's compute_hpwl_after() REPLACED current_hpwl_ with only the
//   nets touching the new module. This is wrong — it discards all previous
//   nets' contributions.
//
// CORRECT approach:
//   new_hpwl = old_hpwl + delta_for_affected_nets
//   where delta = new_hpwl(net) - old_hpwl(net)
//
// For nets that do NOT contain the new module, their contribution is unchanged.
// For nets that DO contain the new module, we recompute their bounding box.
//
inline double compute_hpwl_after(const FloorplanState& s,
                                 std::size_t new_mod_id,
                                 const Placement& new_placement) {
    const auto& new_mod = s.module(new_mod_id);
    const auto& nets    = s.nets();
    const auto& placed  = s.placed();

    // Build a quick lookup: module_id -> Placement (for already-placed modules)
    // N <= 50, so a linear scan per lookup is fine.
    auto find_placement = [&](std::size_t mid) -> const Placement* {
        for (const auto& p : placed) {
            if (p.module_id == mid) return &p;
        }
        return nullptr;
    };

    double hpwl_delta = 0.0;

    for (const auto& net : nets) {
        bool net_contains_new = false;
        for (std::size_t mid : net) {
            if (mid == new_mod_id) { net_contains_new = true; break; }
        }
        if (!net_contains_new) continue;  // unchanged contribution

        // ── Compute OLD HPWL for this net (before placing new_mod) ──
        double old_min_x =  std::numeric_limits<double>::infinity();
        double old_max_x = -std::numeric_limits<double>::infinity();
        double old_min_y =  std::numeric_limits<double>::infinity();
        double old_max_y = -std::numeric_limits<double>::infinity();

        bool any_placed_old = false;
        for (std::size_t mid : net) {
            if (mid == new_mod_id) continue;  // not placed yet in old state
            const auto* pp = find_placement(mid);
            if (!pp) continue;                // should not happen
            const auto& m = s.module(mid);
            // Use module center as pin proxy (Phase 1 simplification)
            double cx = pp->x + m.width  / 2.0;
            double cy = pp->y + m.height / 2.0;
            old_min_x = std::min(old_min_x, cx);  old_max_x = std::max(old_max_x, cx);
            old_min_y = std::min(old_min_y, cy);  old_max_y = std::max(old_max_y, cy);
            any_placed_old = true;
        }
        double old_hpwl_net = any_placed_old
            ? (old_max_x - old_min_x) + (old_max_y - old_min_y)
            : 0.0;

        // ── Compute NEW HPWL for this net (after placing new_mod) ──
        double new_min_x =  std::numeric_limits<double>::infinity();
        double new_max_x = -std::numeric_limits<double>::infinity();
        double new_min_y =  std::numeric_limits<double>::infinity();
        double new_max_y = -std::numeric_limits<double>::infinity();

        bool any_placed_new = false;
        for (std::size_t mid : net) {
            double cx, cy;
            if (mid == new_mod_id) {
                cx = new_placement.x + new_mod.width  / 2.0;
                cy = new_placement.y + new_mod.height / 2.0;
            } else {
                const auto* pp = find_placement(mid);
                if (!pp) continue;
                const auto& m = s.module(mid);
                cx = pp->x + m.width  / 2.0;
                cy = pp->y + m.height / 2.0;
            }
            new_min_x = std::min(new_min_x, cx);  new_max_x = std::max(new_max_x, cx);
            new_min_y = std::min(new_min_y, cy);  new_max_y = std::max(new_max_y, cy);
            any_placed_new = true;
        }
        double new_hpwl_net = any_placed_new
            ? (new_max_x - new_min_x) + (new_max_y - new_min_y)
            : 0.0;

        hpwl_delta += (new_hpwl_net - old_hpwl_net);
    }

    return s.current_hpwl() + hpwl_delta;
}

} // namespace floorplan
