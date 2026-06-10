/**
 * @file floorplan.cpp
 * @brief Example: VLSI Floorplanning with P3-ES-WS-MCTS v3.0.1
 *
 * WS = Work Stealing (Chase-Lev deques)
 */

#include <floorplan/floorplan_state.hpp>
#include <floorplan/floorplan_reward.hpp>
#include <floorplan/floorplan_rollout.hpp>
#include <floorplan/floorplan_bookshelf.hpp>

#include <mcts/mcts.hpp>

#include <chrono>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <aux_file> [iterations]\n";
        return 1;
    }
    const std::size_t iterations = (argc > 2) ? std::stoul(argv[2]) : 1000;

    using namespace floorplan;
    FloorplanState root = FloorplanState::from_bookshelf(argv[1]);

    std::cout << "Loaded " << root.num_modules() << " modules, "
              << root.nets().size() << " nets\n";

    mcts::MCTSConfig cfg;
    cfg.num_iterations = iterations;
    cfg.num_workers    = 4;
    cfg.c_puct = 1.414;

    // NOTE: The P3ESWSMCTS template takes <State, Action> only.
    // Rollout strategy is selected at compile time via a policy template param
    // or runtime via MCTSConfig. For this example we use the default random
    // rollout; greedy_rollout can be swapped in via a custom RolloutEngine.
    mcts::P3ESWSMCTS<FloorplanState, FloorplanState::Action> engine(cfg);
    auto action = engine.search(root);

    auto [mid, slot] = decode_action(action);
    FloorplanState best = root.apply(action);

    std::cout << "\n=== Best Action ===\n";
    std::cout << "  Module:   " << mid << "\n";
    std::cout << "  Slot:     " << slot << "\n";
    std::cout << "  Position: (" << best.placed().back().x << ", "
                                << best.placed().back().y << ")\n";
    std::cout << "  Canvas:   " << best.canvas_width() << " x "
                               << best.canvas_height() << "\n";
    std::cout << "  HPWL:     " << best.current_hpwl() << "\n";
    std::cout << "  Reward:   " << best.reward(0) << "\n";

    return 0;
}
