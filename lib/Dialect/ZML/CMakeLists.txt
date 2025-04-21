add_subdirectory(BuiltIns)
add_subdirectory(IR)
add_subdirectory(Typing)
add_subdirectory(Transforms)
add_subdirectory(ExtVal)

target_sources(ZIRC PRIVATE Helpers.cpp)
