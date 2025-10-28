cmake_minimum_required(VERSION 3.15)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_vertexregen_tokenizer_pybind
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/simplify.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/visitor.cpp
)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(_vertexregen_tokenizer_pybind PRIVATE Eigen3::Eigen)

find_package(CGAL 5.6 CONFIG REQUIRED)
target_link_libraries(_vertexregen_tokenizer_pybind PRIVATE CGAL::CGAL)

target_compile_definitions(_vertexregen_tokenizer_pybind PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _vertexregen_tokenizer_pybind DESTINATION vertexregen_tokenizer)
