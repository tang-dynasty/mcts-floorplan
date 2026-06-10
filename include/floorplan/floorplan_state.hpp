#pragma once

#include <mcts/core.hpp>
#include <mcts/state.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace floorplan {

// ─── Constants ───────────────────────────────────────────────────────────
inline constexpr std::uint32_t SLOT_BITS   = 4;
inline constexpr std::uint32_t SLOT_MASK   = (1u << SLOT_BITS) - 1;  // 0xF
inline constexpr std::uint32_t MOD_BITS    = 6;
inline constexpr std::uint32_t MOD_MASK    = (1u << MOD_BITS) - 1;   // 0x3F
inline constexpr std::uint32_t MAX_CORNERS = 1u << SLOT_BITS;        // 16

// ─── Public types ────────────────────────────────────────────────────────
struct Pin {
    double x_offset = 0.0;
    double y_offset = 0.0;
};

struct Module {
    std::size_t      id = 0;
    double           width = 0.0;
    double           height = 0.0;
    double           area = 0.0;          // cached: width * height
    std::vector<Pin> pins;
};

struct Placement {
    std::size_t module_id = 0;
    double      x = 0.0;
    double      y = 0.0;          // bottom-left corner
    bool        rotated = false;
};

// ─── Forward declarations ────────────────────────────────────────────────
class FloorplanState;

inline std::uint32_t encode_action(std::uint32_t module_id, std::uint32_t slot_id);
inline std::pair<std::uint32_t, std::uint32_t> decode_action(std::uint32_t action);

// ─── FloorplanState ──────────────────────────────────────────────────────
class FloorplanState {
public:
    // ─── Concept-required types ───
    using Action = std::uint32_t;

    // ─── 8 required interfaces (GameState concept) ───
    [[nodiscard]] std::vector<Action> valid_actions() const;
    [[nodiscard]] FloorplanState      apply(Action a) const;
    [[nodiscard]] bool                is_terminal() const;
    [[nodiscard]] mcts::Float         reward(std::size_t /*player*/) const;
    [[nodiscard]] std::size_t         player_to_move() const;
    [[nodiscard]] FloorplanState      clone() const;
    [[nodiscard]] std::size_t         hash() const;
    [[nodiscard]] bool                operator==(const FloorplanState& o) const;

    // ─── Floorplan-specific accessors ───
    [[nodiscard]] double      canvas_width()    const { return canvas_w_; }
    [[nodiscard]] double      canvas_height()   const { return canvas_h_; }
    [[nodiscard]] double      current_hpwl()    const { return current_hpwl_; }
    [[nodiscard]] std::size_t num_placed()      const { return placed_.size(); }
    [[nodiscard]] std::size_t num_modules()     const { return modules_.size(); }
    [[nodiscard]] const std::vector<Placement>& placed() const { return placed_; }
    [[nodiscard]] const Module& module(std::size_t id) const { return modules_[id]; }
    [[nodiscard]] const std::vector<std::vector<std::size_t>>& nets() const { return nets_; }

    // ─── Factory: empty state (used by tests) ───
    static FloorplanState empty() { return FloorplanState{}; }

    // ─── Factory: parse from Bookshelf format ───
    static FloorplanState from_bookshelf(const std::string& aux_path);

    // ─── Mutable builder interface (used by fixtures / parser) ───
    std::vector<Module>& mutable_modules() { return modules_; }
    std::vector<std::vector<std::size_t>>& mutable_nets() { return nets_; }
    std::vector<bool>& mutable_is_placed() { return is_placed_; }
    void set_placed(std::vector<Placement> p) { placed_ = std::move(p); }
    void set_canvas(double w, double h) { canvas_w_ = w; canvas_h_ = h; }
    void set_hpwl(double h) { current_hpwl_ = h; }
    void set_max_corners(std::uint32_t mc) { max_corners_ = mc; }

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

    // ─── Internal helpers ───
    [[nodiscard]] std::vector<std::pair<double, double>>
    candidate_corners_for(std::size_t module_id) const;

    // ─── Reward helpers (defined in floorplan_reward.hpp) ───
    [[nodiscard]] double compute_a_max() const;
    [[nodiscard]] double compute_w_max() const;
    [[nodiscard]] double compute_aspect_penalty() const;

    // Grant reward.hpp access to private helpers
    friend double compute_hpwl_after(const FloorplanState& s,
                                     std::size_t new_mod_id,
                                     const Placement& new_placement);
};

// ─── Action encode/decode (free functions) ──────────────────────────────
inline std::uint32_t encode_action(std::uint32_t module_id, std::uint32_t slot_id) {
    return (module_id << SLOT_BITS) | (slot_id & SLOT_MASK);
}

inline std::pair<std::uint32_t, std::uint32_t>
decode_action(std::uint32_t action) {
    return { (action >> SLOT_BITS) & MOD_MASK, action & SLOT_MASK };
}

// ─── Implementation ──────────────────────────────────────────────────────

inline std::vector<FloorplanState::Action>
FloorplanState::valid_actions() const {
    if (is_terminal()) return {};

    std::size_t mid = placed_.size();
    if (mid >= modules_.size()) return {};

    auto corners = candidate_corners_for(mid);
    std::vector<Action> actions;
    actions.reserve(corners.size());
    for (std::uint32_t slot = 0; slot < corners.size(); ++slot) {
        actions.push_back(encode_action(static_cast<std::uint32_t>(mid), slot));
    }
    return actions;
}

inline FloorplanState
FloorplanState::apply(Action a) const {
    FloorplanState next = *this;
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
    next.current_hpwl_ = compute_hpwl_after(next, mid, next.placed_.back());

    return next;
}

inline bool FloorplanState::is_terminal() const {
    return placed_.size() == modules_.size();
}

inline mcts::Float FloorplanState::reward(std::size_t /*player*/) const {
    if (placed_.empty()) return 0.0;

    constexpr double w_a = 0.4, w_w = 0.4, w_r = 0.2;

    const double A  = canvas_w_ * canvas_h_;
    const double Am = compute_a_max();
    const double ra = 1.0 - std::min(A / Am, 1.0);

    const double W  = current_hpwl_;
    const double Wm = compute_w_max();
    const double rw = 1.0 - std::min(W / Wm, 1.0);

    const double R  = compute_aspect_penalty();
    constexpr double Rm = 4.0;
    const double rr = 1.0 - std::min(R / Rm, 1.0);

    return static_cast<mcts::Float>(w_a * ra + w_w * rw + w_r * rr);
}

inline std::size_t FloorplanState::player_to_move() const {
    return 0u;
}

inline FloorplanState FloorplanState::clone() const {
    return *this;
}

inline std::size_t FloorplanState::hash() const {
    if (hash_dirty_) {
        std::size_t h = 14695981039346656037ULL;  // FNV-1a offset basis
        for (const auto& p : placed_) {
            std::size_t bits_x = 0, bits_y = 0;
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

inline std::vector<std::pair<double, double>>
FloorplanState::candidate_corners_for(std::size_t module_id) const {
    const auto& mod = modules_[module_id];
    std::vector<std::pair<double, double>> result;

    if (placed_.empty()) {
        result.emplace_back(0.0, 0.0);
        return result;
    }

    for (const auto& p : placed_) {
        const auto& placed_mod = modules_[p.module_id];
        const double w_placed = placed_mod.width;
        const double h_placed = placed_mod.height;
        const double w_new    = mod.width;
        const double h_new    = mod.height;

        const double candidates[4][2] = {
            { p.x,           p.y + h_placed },  // above
            { p.x + w_placed, p.y           },  // right
            { p.x,           p.y           },  // flush top-left
            { p.x + w_placed, p.y + h_placed }, // diagonal
        };

        for (const auto& c : candidates) {
            double cx = c[0];
            double cy = c[1];

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

} // namespace floorplan
