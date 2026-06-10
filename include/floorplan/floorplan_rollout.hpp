#pragma once

#include "floorplan/floorplan_state.hpp"
#include "floorplan/floorplan_reward.hpp"

#include <algorithm>
#include <limits>
#include <random>

namespace floorplan {

// ─── Greedy rollout: at each step, pick the corner that minimizes HPWL delta ─
inline mcts::Float greedy_rollout(const FloorplanState& initial,
                                  std::mt19937_64& /*rng*/) {
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

// ─── Random rollout: at each step, pick a random valid action ────────────
inline mcts::Float random_rollout(const FloorplanState& initial,
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

// ─── Epsilon-greedy: blend of greedy and random ──────────────────────────
inline mcts::Float epsilon_greedy_rollout(const FloorplanState& initial,
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

} // namespace floorplan
