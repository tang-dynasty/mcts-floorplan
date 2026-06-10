include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_floorplan_state test_floorplan_state.cpp)
target_link_libraries(test_floorplan_state PRIVATE mcts-floorplan::mcts-floorplan Catch2::Catch2WithMain)
