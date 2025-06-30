#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Fonction pour appliquer un filtre médian
void applyMedianFilter(const cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize) {
    // Rayon du noyau
    int radius = kernelSize / 2;

    // Créer une copie de l'image d'entrée pour l'image de sortie
    outputImage = inputImage.clone();

    // Parcourir chaque pixel de l'image (en excluant les bords)
    for (int i = radius; i < inputImage.rows - radius; ++i) {
        for (int j = radius; j < inputImage.cols - radius; ++j) {
            std::vector<uchar> neighborhood;

            // Collecter les pixels voisins dans la fenêtre
            for (int ki = -radius; ki <= radius; ++ki) {
                for (int kj = -radius; kj <= radius; ++kj) {
                    neighborhood.push_back(inputImage.at<uchar>(i + ki, j + kj));
                }
            }

            // Trouver la médiane
            std::nth_element(neighborhood.begin(),
                             neighborhood.begin() + neighborhood.size() / 2,
                             neighborhood.end());
            uchar median = neighborhood[neighborhood.size() / 2];

            // Assigner la médiane au pixel central
            outputImage.at<uchar>(i, j) = median;
        }
    }
}

// Fonction principale
int main() {
    // Charger l'image d'entrée en niveaux de gris
    cv::Mat image = cv::imread("../donnee/Autre.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Erreur : Impossible de charger l'image !" << std::endl;
        return -1;
    }

    // Appliquer le filtre médian
    cv::Mat result;
    int kernelSize = 3; // Taille du voisinage (3x3 par défaut)
    applyMedianFilter(image, result, kernelSize);

    // Afficher les résultats
    cv::imshow("Image originale", image);
    cv::imshow("Image après filtrage médian", result);

    // Sauvegarder l'image résultante
    cv::imwrite("../resultat/resultat_filtrage_median.png", result);

    cv::waitKey(0);
    return 0;
}
