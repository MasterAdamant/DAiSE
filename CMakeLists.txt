cmake_minimum_required(VERSION 3.30...4.4)

project(DAiSE 
    VERSION 1.1 
    DESCRIPTION "Solution to labs of subject DAiSE (Data Analysis in Software Engineering)" 
    LANGUAGES CXX)

add_subdirectory(MathematicalAnalysis)

add_subdirectory(Lab1)
target_link_libraries(Lab1 PRIVATE MathAnLib)

add_subdirectory(Lab2)
target_link_libraries(Lab2 PRIVATE MathAnLib)
add_subdirectory(Lab3)
target_link_libraries(Lab3 PRIVATE MathAnLib)
