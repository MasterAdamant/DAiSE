#Declare library
add_library(MathAnLib SHARED src/formulas.cpp src/analysis.cpp)

#Make it composition-tied (self-incompilable)
target_include_directories(MathAnLib PUBLIC include)
target_compile_features(MathAnLib PUBLIC cxx_std_23)