# Chip Floorplanning with P3-ES-WS-MCTS: Analysis & Test Plan

**P3** = Pipelined Parallel Path-based tree search with virtual loss  
**ES** = Early Stopping (composable T1–T4 triggers)  
**WS** = Work Stealing (Chase-Lev deques)

---

## 1. Problem Domain: Chip Floorplanning

### 1.1 What Floorplanning Is

Chip floorplanning determines the geometric positions of circuit macros (large functional blocks) on a 2D chip canvas, optimizing:
- **Wirelength**: Total distance of connections between macros
- **Area**: Bounding box of the placement
- **Congestion**: Routing channel utilization
- **Aspect ratio / Density**: Even distribution of cells

### 1.2 Why MCTS Fits

| Property | Why MCTS Works |
|----------|---------------|
| **Sequential decision** | Place macros one at a time → actions = placement positions |
| **Delayed reward** | Final wirelength/area only known after all macros placed |
| **Large branching factor** | Each macro can go to many positions → tree search prunes |
| **Trade-off exploration** | UCB balances trying new positions vs. refining known good ones |
| **Parallel evaluation** | Multiple rollouts can evaluate different partial placements simultaneously |

### 1.3 Prior Art

- **Google (Mirhoseini et al.)**: RL + MCTS for TPU placement. Reward = weighted avg of wirelength, density, congestion.
- **Miracle (DATE 2024)**: Multi-action RL with dense reward. State = netlist graph + placed macro coordinates. Action = relative position between macros.
- **EfficientPlace**: MCTS + transfer learning for parameter search in placement.

---

## 2. Mapping Floorplanning to P3-ES-WS-MCTS Components

### 2.1 GameState Concept (`FloorplanState`)

```cpp
struct Macro {
    std::string name;
    int width, height;      // in grid cells
    int x = -1, y = -1;     // position (-1 = unplaced)
};

struct Net {
    std::vector<int> macro_ids;  // macros connected by this net
    int weight = 1;
};

class FloorplanState {
public:
    using Action = std::pair<int, std::pair<int, int>>;  // (macro_id, (x, y))

    std::vector<Action> valid_actions() const;
    FloorplanState apply(Action a) const;
    bool is_terminal() const;           // all macros placed
    mcts::Float reward(std::size_t player) const;  // -wirelength - lambda*area
    std::size_t player_to_move() const { return 0; }
    FloorplanState clone() const;
    std::size_t hash() const;
    bool operator==(const FloorplanState& other) const;

private:
    int canvas_w_, canvas_h_;
    std::vector<Macro> macros_;
    std::vector<Net> nets_;
    int next_macro_idx_ = 0;   // which macro to place next
};
```

**Key design decisions:**
1. **Action = (macro_id, grid_position)** — deterministic placement
2. **Order = connection-degree descending** — place highly-connected macros first (heuristic from Miracle)
3. **Reward = negative cost** — lower wirelength/area = higher reward
4. **Terminal = all macros placed** — full floorplan evaluated

### 2.2 Reward Function

```
reward = -(wirelength + alpha * area_penalty + beta * overlap_penalty)
```

Where:
- `wirelength` = HPWL (Half-Perimeter Wirelength) = Σ_net (max_x - min_x + max_y - min_y)
- `area_penalty` = (bounding_box_area - target_area)² if > target, else 0
- `overlap_penalty` = Σ_overlapping_pairs (overlap_area)
- `alpha`, `beta` = hyperparameters

**Normalization:** Scale to [0, 1] by dividing by a pessimistic upper bound.

### 2.3 Branching Factor Management

Floorplanning has **enormous branching factor** — a 10x10 grid with 20 macros has ~100^20 possible placements. MCTS handles this via:

| Technique | How |
|-----------|-----|
| **Progressive widening** | Only expand top-K actions by a fast heuristic (e.g., closest-to-center) |
| **Action pruning** | Remove positions that cause overlap with already-placed macros |
| **Grid discretization** | Coarse grid (e.g., 10x10) for MCTS, fine grid for final legalization |
| **Symmetry breaking** | Fix first macro at origin to reduce equivalent states |

---

## 3. Component-by-Component Mapping

### 3.1 Core (`core.hpp`) — Config

```cpp
struct MCTSConfig {
    // ... existing fields ...

    // Floorplan-specific
    int canvas_width = 100;
    int canvas_height = 100;
    Float wirelength_weight = 1.0;
    Float area_weight = 0.5;
    Float overlap_weight = 10.0;   // heavy penalty
    std::size_t max_actions_per_node = 20;  // progressive widening
};
```

### 3.2 Node (`node.hpp`) — No Changes

The fixed `MAX_CHILDREN = 64` is sufficient with progressive widening (max 20 children per node).

### 3.3 Tree (`tree.hpp`) — No Changes

Arena allocation and CAS expansion work out-of-the-box. Floorplanning trees are wide but shallow (depth = num_macros), so arena capacity of 1M is ample for ~50 macros.

### 3.4 Virtual Loss (`vl_tracker.hpp`) — No Changes

Per-node atomic counters handle concurrent path exploration naturally. Floorplanning benefits because different workers explore different macro placement orders simultaneously.

### 3.5 Path Selector (`path_selector.hpp`) — UCB-Tuned Advantage

**Why UCB-Tuned matters for floorplanning:**
- Wirelength variance is high early (unplaced macros = uncertainty)
- Low-variance nodes = mature placements = less exploration needed
- UCB-Tuned automatically reduces exploration bonus for stable regions

```cpp
// Use traverse_tuned() for floorplanning
auto path = selector.traverse_tuned(tree, root_id, vl_tracker, rng);
```

### 3.6 Work Stealing (`work_stealing_pool.hpp`) — Load Balancing

**Floorplanning workload characteristics:**
- Rollout cost varies wildly: placing macro #2 is fast (few constraints), placing macro #20 is slow (many overlap checks)
- **WS benefit**: Workers who finish fast steal from workers with heavy rollouts
- **ES-Directed WS benefit**: When T1 fires (value converged), steal toward most-explored subtree = refine best partial placement

### 3.7 Stats Aggregator (`stats_aggregator.hpp`) — No Changes

Sharded counters reduce contention. Floorplanning rollouts are deterministic (no random opponent), so Q_sum_sq variance tracking is accurate.

### 3.8 Early Stop (`early_stop.hpp`) — All Four Triggers Apply

| Trigger | Floorplanning Interpretation |
|---------|-----------------------------|
| **T1** (ValueConverged) | Best wirelength hasn't improved in N iterations → stop |
| **T2** (BudgetExhausted) | Max iterations reached |
| **T3** (ActionCertain) | One placement position dominates for next macro → stop |
| **T4** (TimeLimit) | Wall-clock deadline (e.g., 1 hour) |

**T3 is especially powerful:** If 95% of rollouts place macro #5 at position (3, 4), we can stop and commit.

### 3.9 Pipeline (`pipeline.hpp`) — S1-S4 Decoupling

**Floorplanning stage characteristics:**
- **S1 (Select)**: Fast — UCB traversal, O(depth)
- **S2 (Expand)**: Fast — allocate children, O(branching_factor)
- **S3 (Rollout)**: **Slow** — simulate full placement, O(num_macros²) overlap checks
- **S4 (Backup)**: Fast — atomic updates

**Pipeline benefit:** Multiple rollouters (S3) can run in parallel while selectors (S1) keep the queues full. This is the ideal workload for pipeline mode.

```cpp
config.enable_pipeline = true;
config.num_workers = 16;  // 4 selectors, 4 expanders, 4 rollouters, 4 backuppers
```

---

## 4. Test Coverage Matrix

### 4.1 Coverage Tests — FloorplanState Correctness

| Test | What It Verifies | File |
|------|-----------------|------|
| `floorplan_valid_actions_no_overlap` | Actions exclude positions causing overlap | `test_floorplan.cpp` |
| `floorplan_valid_actions_boundary` | Actions stay within canvas | `test_floorplan.cpp` |
| `floorplan_apply_places_macro` | apply() updates macro position | `test_floorplan.cpp` |
| `floorplan_terminal_all_placed` | is_terminal() true when all macros placed | `test_floorplan.cpp` |
| `floorplan_reward_wirelength` | reward() = -HPWL for simple netlist | `test_floorplan.cpp` |
| `floorplan_reward_overlap_penalty` | reward() heavily penalizes overlap | `test_floorplan.cpp` |
| `floorplan_clone_independent` | clone() produces independent copy | `test_floorplan.cpp` |
| `floorplan_hash_consistent` | equal states have equal hashes | `test_floorplan.cpp` |
| `floorplan_order_by_degree` | next_macro_idx follows connection degree | `test_floorplan.cpp` |

### 4.2 Pipeline Mode Tests

| Test | What It Verifies | File |
|------|-----------------|------|
| `pipeline_basic_search_completes` | Pipeline search finishes without hang | `test_pipeline.cpp` |
| `pipeline_produces_valid_action` | Result is a valid placement | `test_pipeline.cpp` |
| `pipeline_same_result_as_bulk_sync` | Same config → same best action (deterministic seed) | `test_pipeline.cpp` |
| `pipeline_queue_backpressure` | Queues don't overflow under load | `test_pipeline.cpp` |
| `pipeline_vl_symmetry` | VL registered in S1, released in S4 | `test_pipeline.cpp` |
| `pipeline_worker_role_assignment` | get_role() assigns correct roles | `test_pipeline.cpp` |

### 4.3 Directed Steal Tests

| Test | What It Verifies | File |
|------|-----------------|------|
| `directed_steal_t1_explored` | T1 → StealDirection::Explored | `test_directed_steal.cpp` |
| `directed_steal_t3_underexplored` | T3 → StealDirection::Underexplored | `test_directed_steal.cpp` |
| `directed_steal_hint_node_valid` | hint_node is a real child of root | `test_directed_steal.cpp` |
| `directed_steal_falls_back_to_any` | Invalid hint → normal steal | `test_directed_steal.cpp` |
| `directed_steal_reduces_contention` | Directed steal vs. random steal: lower root VL | `test_directed_steal.cpp` |

### 4.4 Property Tests (QuickCheck-style)

| Property | Generator | Invariant | File |
|----------|-----------|-----------|------|
| `prop_reward_in_range` | Random floorplan states | reward ∈ [0, 1] | `test_properties.cpp` |
| `prop_terminal_no_actions` | Random placed states | terminal → valid_actions().empty() | `test_properties.cpp` |
| `prop_apply_increments_depth` | Random state + random action | new state has one more placed macro | `test_properties.cpp` |
| `prop_symmetry_breaking` | Random netlist | First macro always at (0,0) | `test_properties.cpp` |
| `prop_no_overlap_in_rollout` | Random state + greedy rollout | Final state has zero overlap | `test_properties.cpp` |
| `prop_ucb_tuned_reduces_variance` | Fixed tree with varied children | tuned selects low-variance more often | `test_properties.cpp` |
| `prop_vl_never_negative` | Random concurrent operations | all vl_counts ≥ 0 | `test_properties.cpp` |
| `prop_backup_consistent` | Random path + reward | N increments by 1, Q_sum increases | `test_properties.cpp` |

### 4.5 Integration / End-to-End Tests

| Test | What It Verifies | File |
|------|-----------------|------|
| `e2e_small_netlist_optimal` | 2-macro netlist → MCTS finds optimal placement | `test_integration.cpp` |
| `e2e_scaling_workers` | 2/4/8 workers → throughput scales sublinearly | `test_integration.cpp` |
| `e2e_ensemble_improves` | ensemble_size=4 vs. 1 → better reward | `test_integration.cpp` |
| `e2e_early_stop_saves_time` | T1/T3 enabled vs. disabled → faster, similar quality | `test_integration.cpp` |
| `e2e_pipeline_faster_than_bulk` | enable_pipeline=true → higher throughput | `test_integration.cpp` |
| `e2e_directed_steal_better` | ES-directed vs. random steal → lower variance | `test_integration.cpp` |

---

## 5. Benchmarks

### 5.1 Benchmark Netlists

| Name | #Macros | #Nets | Grid Size | Source |
|------|---------|-------|-----------|--------|
| `tiny` | 3 | 3 | 10x10 | Hand-crafted, optimal known |
| `small` | 10 | 15 | 30x30 | Random, connected |
| `medium` | 25 | 40 | 50x50 | Random, clustered |
| `ami33` | 33 | 123 | 100x100 | MCNC benchmark (scaled) |
| `ami49` | 49 | 408 | 150x150 | MCNC benchmark (scaled) |

### 5.2 Benchmark Metrics

| Metric | How Measured |
|--------|-------------|
| **Throughput** | iterations/sec for fixed netlist |
| **Speedup** | T(1 worker) / T(N workers) |
| **Solution Quality** | wirelength vs. known optimal (for tiny) |
| **Convergence** | best reward vs. iteration count |
| **Memory** | peak arena usage |
| **ES Savings** | iterations saved by T1/T3 |

---

## 6. File Structure

```
examples/
├── floorplan.cpp              # Main example: run MCTS on benchmark netlists
├── floorplan_state.hpp        # FloorplanState implementation
├── netlist_loader.hpp         # Load netlists from simple text format
└── benchmarks/
    ├── tiny.txt
    ├── small.txt
    ├── medium.txt
    ├── ami33.txt
    └── ami49.txt

tests/
├── test_floorplan.cpp         # Coverage tests for FloorplanState
├── test_pipeline.cpp          # Pipeline mode correctness
├── test_directed_steal.cpp    # ES-directed work stealing
├── test_properties.cpp        # Property-based tests
└── test_integration.cpp       # End-to-end integration tests
```

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Overlap detection too slow | Medium | High | Use spatial hash grid (O(1) lookup) |
| Branching factor explodes | High | High | Progressive widening + action pruning |
| Reward signal too sparse | Medium | Medium | Dense reward = delta wirelength per placement |
| Pipeline mode deadlocks | Low | High | Test with small netlists first |
| Arena exhaustion | Low | Medium | Monitor size(), increase capacity |
| Non-determinism across runs | Medium | Low | Fix seed, test reproducibility |

---

## 8. Summary

Floorplanning is an **ideal test domain** for P3-ES-WS-MCTS because:

1. **Real workload characteristics**: Variable rollout cost → WS load balancing matters
2. **Large branching factor**: Tests progressive widening and action pruning
3. **Deterministic transitions**: No opponent randomness → UCB-Tuned variance is meaningful
4. **Clear reward function**: Wirelength/area are objective and measurable
5. **Pipeline-friendly**: S3 (rollout) is much slower than S1/S2/S4 → pipeline parallelism pays off
6. **Early-stop applicable**: T3 (action certainty) fires naturally when one placement dominates
7. **Directed steal applicable**: T1 → refine best subtree, T3 → verify second-best

The test suite will exercise every v3.0.1 component under realistic conditions, with coverage, pipeline, directed steal, and property tests providing orthogonal verification.
