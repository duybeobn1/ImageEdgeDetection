cmake_minimum_required(VERSION 3.10)
project(ImageProcessingSuite)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Activer le Position Independent Code pour Qt
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Inclure OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Inclure Qt5
find_package(Qt5 REQUIRED COMPONENTS Widgets)
include_directories(${Qt5Widgets_INCLUDE_DIRS})
add_definitions(${Qt5Widgets_DEFINITIONS})

# Fonction pour ajouter des sous-projets
function(add_subproject project_name source_path)
    add_executable(${project_name} ${source_path})
    target_link_libraries(${project_name} ${OpenCV_LIBS} Qt5::Widgets)
endfunction()

# TP1 - Convolution
add_subproject(convolution_compare "../Convolution/convolution_compare.cpp")

# TP1 - Filtrage median
add_subproject(filtrage_median "../Filtrage_median/filtrage_median.cpp")

# TP1 - Transformations géométriques
add_subproject(geometric_transformations "../Transformation/GeometricTransformations.cpp")

# TP1 - UI text
add_subproject(ui_module "../Interface/UI.cpp")

# TP1 - histogramProcessing
add_subproject(histogram_processing "../histogramProcessing/HistogramProcessing.cpp")

# Qt Directory
add_subproject(test_module "../test/main.cpp")

# Définir le dossier de sortie des binaires
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Copier les fichiers de données nécessaires dans le dossier binaire
file(COPY ${CMAKE_SOURCE_DIR}/donnee DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/donnee2 DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/resultat DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/output DESTINATION ${CMAKE_BINARY_DIR})
