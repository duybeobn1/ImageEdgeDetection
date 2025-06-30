#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Fonction pour calculer l'histogramme de l'image
void calculateHistogram(const cv::Mat& inputImage, std::vector<int>& histogram) {
    histogram.assign(256, 0); // Initialiser l'histogramme avec 256 niveaux à 0
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            histogram[inputImage.at<uchar>(i, j)]++; // Incrémenter pour chaque intensité
        }
    }
}

// Fonction pour calculer l'histogramme cumulé
void calculateCumulativeHistogram(const std::vector<int>& histogram, std::vector<int>& cumulativeHistogram) {
    cumulativeHistogram.resize(histogram.size(), 0);
    cumulativeHistogram[0] = histogram[0];
    for (size_t i = 1; i < histogram.size(); ++i) {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i]; // Somme cumulée
    }
}

// Fonction pour égaliser l'histogramme
void equalizeHistogram(const cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& lut) {
    std::vector<int> histogram, cumulativeHistogram;
    calculateHistogram(inputImage, histogram); // Calculer l'histogramme
    calculateCumulativeHistogram(histogram, cumulativeHistogram); // Calculer l'histogramme cumulé

    int totalPixels = inputImage.rows * inputImage.cols;
    int cmin = *std::find_if(cumulativeHistogram.begin(), cumulativeHistogram.end(), [](int value) { return value > 0; });

    // Créer la LUT (Look-Up Table) pour l'égalisation
    lut = cv::Mat(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>((cumulativeHistogram[i] - cmin) * 255 / (totalPixels - cmin));
    }

    // Appliquer la LUT à l'image d'entrée
    cv::LUT(inputImage, lut, outputImage);
}

// Fonction pour étirer l'histogramme
void stretchHistogram(const cv::Mat& inputImage, cv::Mat& outputImage, cv::Mat& lut) {
    double minGray, maxGray;
    cv::minMaxLoc(inputImage, &minGray, &maxGray); // Trouver les valeurs min et max de l'intensité

    // Créer la LUT pour étirer l'histogramme
    lut = cv::Mat(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>((i - minGray) * 255 / (maxGray - minGray));
    }

    // Appliquer la LUT à l'image d'entrée
    cv::LUT(inputImage, lut, outputImage);
}

// Fonction pour tracer l'histogramme
void plotHistogram(const std::vector<int>& histogram, const std::string& windowName) {
    int histSize = histogram.size();
    int histHeight = 400;
    cv::Mat histImage(histHeight, histSize, CV_8UC1, cv::Scalar(255));

    int maxValue = *std::max_element(histogram.begin(), histogram.end());
    for (int i = 0; i < histSize; ++i) {
        int barHeight = cv::saturate_cast<int>((double)histogram[i] / maxValue * histHeight);
        cv::line(histImage, cv::Point(i, histHeight), cv::Point(i, histHeight - barHeight), cv::Scalar(0));
    }

    cv::imshow(windowName, histImage); // Afficher l'histogramme
}

// Fonction pour tracer la LUT (Look-Up Table)
void plotLUT(const cv::Mat& lut, const std::string& windowName) {
    int lutSize = lut.cols;
    int lutHeight = 256;
    cv::Mat lutImage(lutHeight, lutSize, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i < lutSize; ++i) {
        int value = lut.at<uchar>(i);
        cv::line(lutImage, cv::Point(i, lutHeight), cv::Point(i, lutHeight - value), cv::Scalar(0));
    }

    cv::imshow(windowName, lutImage); // Afficher la LUT
}

int main() {
    // Charger l'image en niveaux de gris
    cv::Mat inputImage = cv::imread("../donnee/Autre.png", cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Unable to load the image." << std::endl;
        return -1;
    }

    // Initialisation des images et LUTs
    cv::Mat equalizedImage, equalizationLUT;
    cv::Mat stretchedImage, stretchingLUT;

    // Appliquer l'égalisation d'histogramme
    equalizeHistogram(inputImage, equalizedImage, equalizationLUT);
    cv::imwrite("../resultat/equalized_image.png", equalizedImage);

    // Appliquer l'étirement d'histogramme
    stretchHistogram(inputImage, stretchedImage, stretchingLUT);
    cv::imwrite("../resultat/stretched_image.png", stretchedImage);

    // Calculer les histogrammes pour chaque image
    std::vector<int> originalHistogram, equalizedHistogramData, stretchedHistogramData;
    calculateHistogram(inputImage, originalHistogram);
    calculateHistogram(equalizedImage, equalizedHistogramData);
    calculateHistogram(stretchedImage, stretchedHistogramData);

    // Tracer les histogrammes
    plotHistogram(originalHistogram, "Original Histogram");
    plotHistogram(equalizedHistogramData, "Equalized Histogram");
    plotHistogram(stretchedHistogramData, "Stretched Histogram");

    // Tracer les LUTs
    plotLUT(equalizationLUT, "Equalization LUT");
    plotLUT(stretchingLUT, "Stretching LUT");

    // Afficher les images originales et transformées
    cv::imshow("Original Image", inputImage);
    cv::imshow("Equalized Image", equalizedImage);
    cv::imshow("Stretched Image", stretchedImage);

    // Sauvegarder les images finales
    cv::imwrite("../resultat/original_image.png", inputImage);
    cv::imwrite("../resultat/equalized_image.png", equalizedImage);
    cv::imwrite("../resultat/stretched_image.png", stretchedImage);

    cv::waitKey(0); // Attendre une touche pour fermer les fenêtres
    return 0;
}
