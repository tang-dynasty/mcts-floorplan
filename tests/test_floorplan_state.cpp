#include <catch2/catch_test_macros.hpp>
#include <floorplan/floorplan_state.hpp>
#include <floorplan/floorplan_reward.hpp>
#include <floorplan/floorplan_rollout.hpp>

#include <random>
#include <unordered_set>

using namespace floorplan;

// ─── Fixtures ────────────────────────────────────────────────────────────
FloorplanState make_small_fixture() {
    FloorplanState s;
    s.mutable_modules() = {
        {0, 4.0, 3.0, 12.0, {}},           // core
        {1, 2.0, 2.0,  4.0, {}},           // cache
        {2, 1.5, 1.5,  2.25, {}},          // reg
        {3, 3.0, 1.0,  3.0, {}},           // bus
        {4, 1.0, 1.0,  1.0, {}},           // io
    };
    s.mutable_nets() = {{0,1}, {0,2}, {1,2}, {2,3}, {3,4}, {0,4}, {1,3}, {2,4}};
    s.mutable_is_placed().assign(5, false);
    return s;
}

FloorplanState make_medium_fixture() {
    FloorplanState s;
    for (std::size_t i = 0; i < 8; ++i) {
        double sz = 1.0 + static_cast<double>(i % 3);
        s.mutable_modules().push_back({i, sz, sz, sz * sz, {}});
    }
    s.mutable_nets() = {{0,1,2}, {1,2,3}, {3,4,5}, {5,6,7}, {0,7}, {2,5}};
    s.mutable_is_placed().assign(8, false);
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
        bool has_overlap = x_overlap && y_overlap;
        REQUIRE_FALSE(has_overlap);
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
        mcts::Float r = s.reward(0);
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

// ─── Test 9: hash collision resistance (1k random rollouts) ─────────────
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
    // Expect ~1000 unique hashes (allow 2.5% collision for medium fixture)
    REQUIRE(hashes.size() >= 975);
}

// ─── Test 10: reward deterministic ──────────────────────────────────────
TEST_CASE("reward_deterministic", "[floorplan][state]") {
    auto s = make_small_fixture();
    s = s.apply(s.valid_actions()[0]);
    mcts::Float r1 = s.reward(0);
    mcts::Float r2 = s.reward(0);
    mcts::Float r3 = s.reward(0);
    REQUIRE(r1 == r2);
    REQUIRE(r2 == r3);
}
