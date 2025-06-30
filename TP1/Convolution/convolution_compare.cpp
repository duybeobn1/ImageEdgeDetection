#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>


// Charger une image PNG en niveaux de gris
/**
 * @brief Charge une image PNG en niveaux de gris.
 * 
 * @param path Chemin du fichier image à charger.
 * @param image Matrice 2D pour stocker l'image chargée.
 * @param width Largeur de l'image chargée.
 * @param height Hauteur de l'image chargée.
 * @return true si l'image a été chargée avec succès, false sinon.
 */
bool loadImage(const std::string& path, std::vector<std::vector<unsigned char>>& image, int& width, int& height) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Erreur : Impossible de lire l'image PNG !" << std::endl;
        return false;
    }

    height = img.rows;
    width = img.cols;

    // Convertir l'image OpenCV en une matrice 2D
    image.resize(height, std::vector<unsigned char>(width));
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image[i][j] = img.at<uchar>(i, j);
        }
    }
    return true;
}

// Fonction pour sauvegarder une image dans un fichier
/**
 * @brief Sauvegarde une image au chemin spécifié.
 * 
 * @param path Chemin du fichier où l'image sera sauvegardée.
 * @param img Image à sauvegarder.
 * @return true si l'image a été sauvegardée avec succès, false sinon.
 */
bool saveImage(const std::string& path, const std::vector<std::vector<unsigned char>>& image, int width, int height) {
    cv::Mat img(height, width, CV_8UC1);

    // Convertir la matrice 2D en une image OpenCV
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            img.at<uchar>(i, j) = image[i][j];
        }
    }

    // Sauvegarder l'image
    if (!cv::imwrite(path, img)) {
        std::cerr << "Erreur : Impossible de sauvegarder l'image PNG !" << std::endl;
        return false;
    }
    return true;
}

// Fonction pour calculer l'histogramme d'une image en niveaux de gris
/**
 * @brief Calcule l'histogramme d'une image manuellement.
 * 
 * @param image Image en niveaux de gris pour laquelle calculer l'histogramme.
 * @return Vecteur contenant l'histogramme de l'image.
 */
std::vector<int> calculateHistogramManual(const cv::Mat& image) {
    // Vérifier que l'image est bien en niveaux de gris
    CV_Assert(image.type() == CV_8UC1);
    
    std::vector<int> histogram(256, 0);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            uchar pixel = image.at<uchar>(i, j);
            histogram[pixel]++;
        }
    }
    return histogram;
}


// Fonction pour normaliser un histogramme
/**
 * @brief Normalise un histogramme en divisant chaque valeur par le nombre total de pixels.
 * 
 * @param histogram Histogramme à normaliser.
 * @param totalPixels Nombre total de pixels dans l'image.
 * @return Vecteur contenant l'histogramme normalisé.
 */
std::vector<double> normalizeHistogram(const std::vector<int>& histogram, int totalPixels) {
    std::vector<double> normalizedHistogram(256, 0.0);

    for (int i = 0; i < 256; ++i) {
        normalizedHistogram[i] = static_cast<double>(histogram[i]) / totalPixels;
    }

    return normalizedHistogram;
}

// Fonction pour comparer deux histogrammes avec la corrélation
/**
 * @brief Compare deux histogrammes en utilisant le coefficient de corrélation.
 * 
 * @param hist1 Premier histogramme.
 * @param hist2 Deuxième histogramme.
 * @return Valeur de corrélation entre les deux histogrammes.
 */
double compareHistogramsManual(const std::vector<int>& hist1, const std::vector<int>& hist2) {
    double mean1 = 0.0, mean2 = 0.0, numerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;

    for (int i = 0; i < 256; ++i) {
        mean1 += hist1[i];
        mean2 += hist2[i];
    }

    mean1 /= 256;
    mean2 /= 256;

    // Calculer la corrélation
    for (int i = 0; i < 256; ++i) {
        numerator += (hist1[i] - mean1) * (hist2[i] - mean2);
        denominator1 += (hist1[i] - mean1) * (hist1[i] - mean1);
        denominator2 += (hist2[i] - mean2) * (hist2[i] - mean2);
    }

    double denominator = std::sqrt(denominator1 * denominator2);
    return (denominator == 0.0) ? 0.0 : numerator / denominator;
}



// Fonction pour appliquer une convolution sur une image
/**
 * @brief Applique une convolution sur une image en niveaux de gris.
 * 
 * @param inputImage Image d'entrée.
 * @param outputImage Image de sortie.
 * @param kernel Noyau de convolution.
 */
void applyConvolution(const std::vector<std::vector<unsigned char>>& inputImage,
                      std::vector<std::vector<unsigned char>>& outputImage,
                      const std::vector<std::vector<float>>& kernel) {
    int height = inputImage.size();
    int width = inputImage[0].size();
    int kernelSize = kernel.size();
    int radius = kernelSize / 2;

    // Initialiser l'image de sortie
    outputImage = inputImage;

    // Parcourir chaque pixel de l'image (en excluant les bords)
    for (int i = radius; i < height - radius; ++i) {
        for (int j = radius; j < width - radius; ++j) {
            float pixelValue = 0.0;

            // Appliquer le noyau
            for (int ki = -radius; ki <= radius; ++ki) {
                for (int kj = -radius; kj <= radius; ++kj) {
                    int neighborX = i + ki;
                    int neighborY = j + kj;
                    pixelValue += inputImage[neighborX][neighborY] * kernel[ki + radius][kj + radius];
                }
            }

            // Assigner la valeur convoluée
            outputImage[i][j] = std::clamp(static_cast<int>(pixelValue), 0, 255);
        }
    }
}

// showHistogram : Affiche un histogramme dans une fenêtre OpenCV
/**
 * @brief Affiche un histogramme dans une fenêtre OpenCV.
 * 
 * @param hist Histogramme à afficher.
 * @param windowName Nom de la fenêtre.
 */
void showHistogram(const std::vector<int>& hist, const std::string& windowName) {
    int histSize = 256;  // Taille de l'histogramme
    int hist_w = 512;    // Largeur de l'image d'affichage de l'histogramme
    int hist_h = 400;    // Hauteur de l'image d'affichage
    int bin_w = cvRound((double)hist_w / histSize);

    // Créer une image blanche pour l'affichage
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Trouver la valeur maximale de l'histogramme (pour la mise à l'échelle)
    int maxVal = 0;
    for (int i = 0; i < histSize; i++) {
        if (hist[i] > maxVal) {
            maxVal = hist[i];
        }
    }

    // Dessiner le graphe : une ligne verticale pour chaque niveau de gris
    for (int i = 0; i < histSize; i++) {
        int intensity = (int)( ( (double)hist[i] / maxVal ) * (hist_h - 20) ); // -20 pour un peu de marge en haut
        cv::line(histImage,
                 cv::Point(bin_w * i, hist_h),
                 cv::Point(bin_w * i, hist_h - intensity),
                 cv::Scalar(0, 0, 0), 1);
    }

    cv::imshow(windowName, histImage);
    cv::waitKey(0);
}

// saveHistogramImage : Sauvegarde un histogramme dans un fichier image
/**
 * @brief Sauvegarde un histogramme dans un fichier image.
 * 
 * @param hist Histogramme à sauvegarder.
 * @param windowName Nom de la fenêtre.
 * @param savePath Chemin du fichier image à sauvegarder.
 */
void saveHistogramImage(const std::vector<int>& hist, const std::string& windowName, const std::string& savePath) {
    int histSize = 256;  // Nombre de niveaux de gris
    int hist_w = 512;    // Largeur de l'image d'histogramme
    int hist_h = 400;    // Hauteur de l'image d'histogramme
    int bin_w = cvRound((double)hist_w / histSize);

    // Créer une image blanche
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Trouver la valeur maximale dans l'histogramme
    int maxVal = *std::max_element(hist.begin(), hist.end());

    // Dessiner les barres de l'histogramme
    for (int i = 0; i < histSize; i++) {
        int intensity = cvRound(((double)hist[i] / maxVal) * (hist_h - 20));
        cv::line(histImage,
                 cv::Point(bin_w * i, hist_h),
                 cv::Point(bin_w * i, hist_h - intensity),
                 cv::Scalar(0, 0, 0), 1);
    }

    // Ajouter le titre au-dessus de l'histogramme
    cv::putText(histImage, windowName, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);

    // Sauvegarder l'image
    cv::imwrite(savePath, histImage);
    std::cout << "Histogramme sauvegardé : " << savePath << std::endl;

    // Optionnel : Afficher l'histogramme à l'écran
    cv::imshow(windowName, histImage);
    cv::waitKey(0);
}

// showMultipleHistograms : Affiche plusieurs histogrammes dans une seule fenêtre OpenCV
/**
 * @brief Affiche plusieurs histogrammes dans une seule fenêtre OpenCV.
 * 
 * @param histData Vecteur contenant les histogrammes à afficher.
 * @param windowName Nom de la fenêtre.
 * @param savePath Chemin du fichier image à sauvegarder.
 */
void showMultipleHistograms(const std::vector<std::pair<std::vector<int>, std::string>>& histData, 
                            const std::string& windowName, 
                            const std::string& savePath) {
    int histSize = 256;
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Trouver la valeur maximale parmi tous les histogrammes
    int maxVal = 0;
    for (auto& hd : histData) {
        for (int i = 0; i < histSize; i++) {
            if (hd.first[i] > maxVal) {
                maxVal = hd.first[i];
            }
        }
    }

    // Palette de couleurs
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0,0,0),     // noir
        cv::Scalar(0,0,255),   // rouge
        cv::Scalar(0,255,0),   // vert
        cv::Scalar(255,0,0),   // bleu
        cv::Scalar(0,255,255), // cyan
        cv::Scalar(255,0,255), // magenta
        cv::Scalar(255,255,0)  // jaune
    };

    // Dessiner chaque histogramme
    for (size_t h = 0; h < histData.size(); h++) {
        cv::Scalar color = colors[h % colors.size()];
        const std::vector<int>& hist = histData[h].first;
        for (int i = 1; i < histSize; i++) {
            int y1 = hist_h - cvRound(((double)hist[i - 1] / maxVal) * (hist_h - 20));
            int y2 = hist_h - cvRound(((double)hist[i] / maxVal) * (hist_h - 20));
            cv::line(histImage, 
                     cv::Point(bin_w * (i - 1), y1), 
                     cv::Point(bin_w * i, y2), 
                     color, 2);
        }
    }

    // Ajouter la légende
    int legend_x = 20;
    int legend_y = 20;
    int legend_line_height = 20;
    for (size_t h = 0; h < histData.size(); h++) {
        cv::Scalar color = colors[h % colors.size()];
        cv::rectangle(histImage, cv::Rect(legend_x, legend_y + (int)h*legend_line_height, 10, 10), color, cv::FILLED);
        cv::putText(histImage, histData[h].second, 
                    cv::Point(legend_x + 20, legend_y + (int)h*legend_line_height + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1);
    }

    // Afficher et sauvegarder l'image
    cv::imshow(windowName, histImage);
    cv::imwrite(savePath, histImage); // Sauvegarder l'image
    cv::waitKey(0);
}

int main() {
    // Charger l'image PNG
    std::vector<std::vector<unsigned char>> image;
    int width, height;
    if (!loadImage("../donnee/Autre.png", image, width, height)) {
        return -1;
    }

    // Définir les noyaux
    // Noyau de moyenne
    std::vector<std::vector<float>> kernel_mean = {
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f},
        {1 / 9.0f, 1 / 9.0f, 1 / 9.0f}
    };

    // Noyau gaussien
    std::vector<std::vector<float>> kernel_gaussian = {
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
        {2 / 16.0f, 4 / 16.0f, 2 / 16.0f},
        {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}
    };

    // Noyau de Sobel (horizontal)
    std::vector<std::vector<float>> kernel_sobel_x = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // Appliquer les filtres manuels
    std::vector<std::vector<unsigned char>> result_mean, result_gaussian, result_sobel;
    applyConvolution(image, result_mean, kernel_mean);
    applyConvolution(image, result_gaussian, kernel_gaussian);
    applyConvolution(image, result_sobel, kernel_sobel_x);

    // Sauvegarder les résultats manuels
    saveImage("../resultat/result_mean_manual.png", result_mean, width, height);
    saveImage("../resultat/result_gaussian_manual.png", result_gaussian, width, height);
    saveImage("../resultat/result_sobel_manual.png", result_sobel, width, height);

    // Charger l'image directement avec OpenCV
    cv::Mat img = cv::imread("../donnee/Autre.png", cv::IMREAD_GRAYSCALE);

    // Appliquer les filtres OpenCV
    cv::Mat result_mean_opencv, result_gaussian_opencv, result_sobel_opencv;
    cv::blur(img, result_mean_opencv, cv::Size(3, 3));
    cv::GaussianBlur(img, result_gaussian_opencv, cv::Size(3, 3), 0);
    cv::Sobel(img, result_sobel_opencv, CV_8U, 1, 0, 3);

    // Sauvegarder les résultats OpenCV
    cv::imwrite("../resultat/result_mean_opencv.png", result_mean_opencv);
    cv::imwrite("../resultat/result_gaussian_opencv.png", result_gaussian_opencv);
    cv::imwrite("../resultat/result_sobel_opencv.png", result_sobel_opencv);

    // Charger les résultats manuels et OpenCV en tant que `cv::Mat`
    cv::Mat result_mean_manual = cv::imread("../resultat/result_mean_manual.png", cv::IMREAD_GRAYSCALE);
    cv::Mat result_mean_ocv = cv::imread("../resultat/result_mean_opencv.png", cv::IMREAD_GRAYSCALE);

    cv::Mat result_gaussian_manual = cv::imread("../resultat/result_gaussian_manual.png", cv::IMREAD_GRAYSCALE);
    cv::Mat result_gaussian_ocv = cv::imread("../resultat/result_gaussian_opencv.png", cv::IMREAD_GRAYSCALE);

    cv::Mat result_sobel_manual = cv::imread("../resultat/result_sobel_manual.png", cv::IMREAD_GRAYSCALE);
    cv::Mat result_sobel_ocv = cv::imread("../resultat/result_sobel_opencv.png", cv::IMREAD_GRAYSCALE);

    // Vérifier si toutes les images ont été chargées correctement
    if (result_mean_manual.empty() || result_mean_ocv.empty() ||
        result_gaussian_manual.empty() || result_gaussian_ocv.empty() ||
        result_sobel_manual.empty() || result_sobel_ocv.empty()) {
        std::cerr << "Erreur : Impossible de charger une ou plusieurs images pour la comparaison !" << std::endl;
        return -1;
    }

    // Calculer et comparer les histogrammes
    std::vector<int> hist_mean_manual = calculateHistogramManual(result_mean_manual);
    std::vector<int> hist_mean_opencv = calculateHistogramManual(result_mean_ocv);
    double similarity_mean = compareHistogramsManual(hist_mean_manual, hist_mean_opencv);

    // Sauvegarder les histogrammes individuellement
    saveHistogramImage(hist_mean_manual, "Mean Manual", "../resultat/hist_mean_manual.png");
    saveHistogramImage(hist_mean_opencv, "Mean OpenCV", "../resultat/hist_mean_opencv.png");

    std::vector<int> hist_gaussian_manual = calculateHistogramManual(result_gaussian_manual);
    std::vector<int> hist_gaussian_opencv = calculateHistogramManual(result_gaussian_ocv);
    double similarity_gaussian = compareHistogramsManual(hist_gaussian_manual, hist_gaussian_opencv);

    saveHistogramImage(hist_gaussian_manual, "Gaussian Manual", "../resultat/hist_gaussian_manual.png");
    saveHistogramImage(hist_gaussian_opencv, "Gaussian OpenCV", "../resultat/hist_gaussian_opencv.png");

    std::vector<int> hist_sobel_manual = calculateHistogramManual(result_sobel_manual);
    std::vector<int> hist_sobel_opencv = calculateHistogramManual(result_sobel_ocv);
    double similarity_sobel = compareHistogramsManual(hist_sobel_manual, hist_sobel_opencv);
    
    saveHistogramImage(hist_sobel_manual, "Sobel Manual", "../resultat/hist_sobel_manual.png");
    saveHistogramImage(hist_sobel_opencv, "Sobel OpenCV", "../resultat/hist_sobel_opencv.png");

    // Afficher les résultats de la comparaison
    std::cout << "Similarité des histogrammes (moyenne) : " << similarity_mean << std::endl;
    std::cout << "Similarité des histogrammes (gaussien) : " << similarity_gaussian << std::endl;
    std::cout << "Similarité des histogrammes (sobel) : " << similarity_sobel << std::endl;

    std::vector<std::pair<std::vector<int>, std::string>> allHistData = {
        {hist_mean_manual, "Mean Manual"},
        {hist_mean_opencv, "Mean OpenCV"},
        {hist_gaussian_manual, "Gaussian Manual"},
        {hist_gaussian_opencv, "Gaussian OpenCV"},
        {hist_sobel_manual, "Sobel Manual"},
        {hist_sobel_opencv, "Sobel OpenCV"}
    };

    // Afficher tous les histogrammes dans une seule fenêtre
    // JUSTE POUR VOIR ! rien de pertinent au TP
    // showMultipleHistograms(allHistData, "All Histograms Comparison", "../resultat/all_histograms.png");

    std::cout << "Filtrage terminé et résultats sauvegardés." << std::endl;
    return 0;
}
