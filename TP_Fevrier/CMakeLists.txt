cmake_minimum_required(VERSION 3.10)
project(QtOpencvExample)

set(CMAKE_CXX_STANDARD 17)

# Trouve Qt 5 ou 6 ; adapté selon vos versions
find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

# Trouve OpenCV
find_package(OpenCV REQUIRED)

# Inclure les répertoires Qt
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_executable(QtOpencvExample
    main.cpp
    mywindow.h
    mywindow.cpp
)

target_link_libraries(QtOpencvExample
    Qt5::Core
    Qt5::Widgets
    Qt5::Gui
    ${OpenCV_LIBS}
)
