# MCTS Floorplan

Circuit floorplanning framework using **P3-ES-WS-MCTS** as the optimization backbone.

## Overview

This repository implements VLSI floorplanning as a Monte Carlo Tree Search problem, leveraging the high-performance parallel MCTS engine from [parallel-mcts](https://github.com/tang-dynasty/parallel-mcts).

## Architecture

```
mcts-floorplan/
├── include/          # Public headers
│   ├── floorplan/    # Floorplanning domain model
│   └── mcts/         # MCTS adapter layer (wraps parallel-mcts)
├── src/              # Implementation
├── tests/            # Unit & integration tests
├── benchmarks/       # MCNC / GSRC benchmarks
└── docs/             # Design docs & API reference
```

## Dependencies

- [parallel-mcts](https://github.com/tang-dynasty/parallel-mcts) — P3-ES-WS-MCTS engine
- C++20
- CMake ≥ 3.20

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## License

MIT
