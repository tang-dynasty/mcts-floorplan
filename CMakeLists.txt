cmake_minimum_required(VERSION 3.20)
project(mcts-floorplan
    VERSION 0.1.0
    DESCRIPTION "VLSI Floorplanning via P3-ES-WS-MCTS"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Third-party: parallel-mcts (git submodule) ──────────────────────────
# The submodule provides header-only parallel-mcts under its own CMake.
# We add it as a subdirectory so its targets (parallel-mcts::parallel-mcts)
# are available for linking.
set(PARALLEL_MCTS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/parallel-mcts")
if(NOT EXISTS "${PARALLEL_MCTS_ROOT}/include/mcts/mcts.hpp")
    message(FATAL_ERROR
        "parallel-mcts submodule not found.\n"
        "Run: git submodule update --init --recursive")
endif()

add_subdirectory("${PARALLEL_MCTS_ROOT}" EXCLUDE_FROM_ALL)

# ─── Compiler warnings ───────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native -DNDEBUG)
    endif()
endif()

# ─── Main library target (header-only) ───────────────────────────────────
add_library(mcts-floorplan INTERFACE)
add_library(mcts-floorplan::mcts-floorplan ALIAS mcts-floorplan)

target_include_directories(mcts-floorplan INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(mcts-floorplan INTERFACE
    parallel-mcts::parallel-mcts
)

# ─── Examples ────────────────────────────────────────────────────────────
option(MCTS_FLOORPLAN_BUILD_EXAMPLES "Build examples" ON)
if(MCTS_FLOORPLAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ─── Tests ───────────────────────────────────────────────────────────────
option(MCTS_FLOORPLAN_BUILD_TESTS "Build tests" ON)
if(MCTS_FLOORPLAN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Benchmarks ──────────────────────────────────────────────────────────
option(MCTS_FLOORPLAN_BUILD_BENCHMARKS "Build benchmarks" ON)
if(MCTS_FLOORPLAN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
