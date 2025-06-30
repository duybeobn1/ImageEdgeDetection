g++ multi_directional_filters.cpp -o multi_directional_filters `pkg-config --cflags --libs opencv4`
./multi_directional_filters ./image/Cathedrale-Lyon.jpg


g++ gradient_detection.cpp -o gradient_detection `pkg-config --cflags --libs opencv4`
./gradient_detection ./image/Cathedrale-Lyon.jpg


g++ gradient_threshold.cpp -o gradient_threshold `pkg-config --cflags --libs opencv4`
./gradient_threshold ./image/lena2.png