target_sources(ZIRC PRIVATE 
  StripTests.cpp
  StripDirectives.cpp
)

add_subdirectory(ConvertZhlToZml)
add_subdirectory(ConvertZmlToLlzk)
