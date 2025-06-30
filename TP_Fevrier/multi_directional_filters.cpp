#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>  // Pour std::abs

using namespace cv;
using namespace std;

/**
 * Fonction pour normaliser un masque en divisant par la somme des valeurs absolues.
 * @param kernel Masque non normalisé de type vector<vector<int>>
 * @return Masque normalisé de type vector<vector<float>>
 */
vector<vector<float>> normalizeKernel(const vector<vector<int>> &kernel) {
    float sumAbs = 0.0;
    // Calcul de la somme des valeurs absolues du masque
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sumAbs += std::abs(kernel[i][j]);
        }
    }
    // Éviter la division par zéro (bien que peu probable pour ces filtres)
    if (sumAbs == 0.0) {
        sumAbs = 1.0;
    }
    // Création et remplissage du masque normalisé
    vector<vector<float>> normalizedKernel(3, vector<float>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            normalizedKernel[i][j] = kernel[i][j] / sumAbs;
        }
    }
    return normalizedKernel;
}

/**
 * Fonction de convolution 2D avec un masque 3x3 de type float.
 * @param image Image en entrée (niveaux de gris)
 * @param kernel Masque normalisé de type vector<vector<float>>
 * @return Résultat de la convolution
 */
Mat applyConvolution(const Mat &image, const vector<vector<float>> &kernel) {
    int rows = image.rows;
    int cols = image.cols;
    //nouveau image dans meme size
    Mat result = Mat::zeros(rows, cols, CV_32F);

    // Parcours de l'image (ignorant les bords)
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            float sum = 0.0;

            // Application du masque 3x3
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sum += image.at<uchar>(i + ki, j + kj) * kernel[ki + 1][kj + 1];
                }
            }

            result.at<float>(i, j) = sum;
        }
    }

    return result;
}

/**
 * Application des filtres dans les directions multi-directionnelles.
 * @param image Image en entrée (niveaux de gris)
 */
void applyMultiDirectionalFilters(const Mat &image) {
    
    // ---------------- Filtres Prewitt (non normalisés) ----------------
    vector<vector<int>> prewittX_int = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    vector<vector<int>> prewittY_int = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    vector<vector<int>> prewitt45_int = {{0, 1, 1}, {-1, 0, 1}, {-1, -1, 0}};
    vector<vector<int>> prewitt135_int = {{1, 1, 0}, {1, 0, -1}, {0, -1, -1}};

    // ---------------- Normalisation des filtres Prewitt ----------------
    vector<vector<float>> prewittX = normalizeKernel(prewittX_int);
    vector<vector<float>> prewittY = normalizeKernel(prewittY_int);
    vector<vector<float>> prewitt45 = normalizeKernel(prewitt45_int);
    vector<vector<float>> prewitt135 = normalizeKernel(prewitt135_int);

    // ---------------- Filtres Sobel (non normalisés) ----------------
    vector<vector<int>> sobelX_int = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    vector<vector<int>> sobelY_int = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    vector<vector<int>> sobel45_int = {{0, 1, 2}, {-1, 0, 1}, {-2, -1, 0}};
    vector<vector<int>> sobel135_int = {{2, 1, 0}, {1, 0, -1}, {0, -1, -2}};

    // ---------------- Normalisation des filtres Sobel ----------------
    vector<vector<float>> sobelX = normalizeKernel(sobelX_int);
    vector<vector<float>> sobelY = normalizeKernel(sobelY_int);
    vector<vector<float>> sobel45 = normalizeKernel(sobel45_int);
    vector<vector<float>> sobel135 = normalizeKernel(sobel135_int);

    // ---------------- Filtres Kirsch (non normalisés) ----------------
    vector<vector<int>> kirschX_int = {{-3, -3, 5}, {-3, 0, 5}, {-3, -3, 5}};
    vector<vector<int>> kirschY_int = {{-3, -3, -3}, {5, 0, -3}, {5, 5, -3}};
    vector<vector<int>> kirsch45_int = {{-3, 5, 5}, {-3, 0, 5}, {-3, -3, -3}};
    vector<vector<int>> kirsch135_int = {{5, 5, -3}, {5, 0, -3}, {-3, -3, -3}};

    // ---------------- Normalisation des filtres Kirsch ----------------
    vector<vector<float>> kirschX = normalizeKernel(kirschX_int);
    vector<vector<float>> kirschY = normalizeKernel(kirschY_int);
    vector<vector<float>> kirsch45 = normalizeKernel(kirsch45_int);
    vector<vector<float>> kirsch135 = normalizeKernel(kirsch135_int);

    // ---------------- Application des filtres normalisés ----------------
    vector<Mat> prewittResults = {
        applyConvolution(image, prewittX), applyConvolution(image, prewittY),
        applyConvolution(image, prewitt45), applyConvolution(image, prewitt135)};

    vector<Mat> sobelResults = {
        applyConvolution(image, sobelX), applyConvolution(image, sobelY),
        applyConvolution(image, sobel45), applyConvolution(image, sobel135)};

    vector<Mat> kirschResults = {
        applyConvolution(image, kirschX), applyConvolution(image, kirschY),
        applyConvolution(image, kirsch45), applyConvolution(image, kirsch135)};

    // ---------------- Combinaison des résultats ----------------
    Mat prewittCombined = Mat::zeros(image.size(), CV_32F);
    Mat sobelCombined = Mat::zeros(image.size(), CV_32F);
    Mat kirschCombined = Mat::zeros(image.size(), CV_32F);

    // Somme des valeurs absolues pour chaque direction
    for (int i = 0; i < 4; i++) {
        prewittCombined += abs(prewittResults[i]);
        sobelCombined += abs(sobelResults[i]);
        kirschCombined += abs(kirschResults[i]);
    }

    // ---------------- Normalisation pour affichage ----------------
    Mat prewittNorm, sobelNorm, kirschNorm;
    normalize(prewittCombined, prewittNorm, 0, 255, NORM_MINMAX, CV_8U);
    normalize(sobelCombined, sobelNorm, 0, 255, NORM_MINMAX, CV_8U);
    normalize(kirschCombined, kirschNorm, 0, 255, NORM_MINMAX, CV_8U);

    // ---------------- Affichage des résultats ----------------
    imshow("Original", image);
    imshow("Prewitt Multi-Directionnel", prewittNorm);
    imshow("Sobel Multi-Directionnel", sobelNorm);
    imshow("Kirsch Multi-Directionnel", kirschNorm);

    // Combinaison colorée des résultats
    vector<Mat> channels(3);
    channels[0] = prewittNorm; // Bleu
    channels[1] = sobelNorm;   // Vert
    channels[2] = kirschNorm;  // Rouge
    Mat coloredEdges;
    merge(channels, coloredEdges);
    imshow("Contours Colorés", coloredEdges);

    // ---------------- Sauvegarde des résultats ----------------
    imwrite("./resultat/contour_colore.jpg", coloredEdges);
    imwrite("./resultat/prewitt.jpg", prewittNorm);
    imwrite("./resultat/sobel.jpg", sobelNorm);
    imwrite("./resultat/kirsch.jpg", kirschNorm);

    // ---------------- Calcul et affichage des histogrammes ----------------
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;

    Mat histPrewitt, histSobel, histKirsch;
    calcHist(&prewittNorm, 1, 0, Mat(), histPrewitt, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&sobelNorm, 1, 0, Mat(), histSobel, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&kirschNorm, 1, 0, Mat(), histKirsch, 1, &histSize, &histRange, uniform, accumulate);

    normalize(histPrewitt, histPrewitt, 0, 400, NORM_MINMAX);
    normalize(histSobel, histSobel, 0, 400, NORM_MINMAX);
    normalize(histKirsch, histKirsch, 0, 400, NORM_MINMAX);

    Mat histPrewittImg(400, 512, CV_8UC3, Scalar(255,255,255));
    Mat histSobelImg(400, 512, CV_8UC3, Scalar(255,255,255));
    Mat histKirschImg(400, 512, CV_8UC3, Scalar(255,255,255));

    for (int i = 1; i < histSize; i++) {
        line(histPrewittImg,
             Point((i-1)*2, 400 - cvRound(histPrewitt.at<float>(i-1))),
             Point(i*2, 400 - cvRound(histPrewitt.at<float>(i))),
             Scalar(0,0,0), 2);
        line(histSobelImg,
             Point((i-1)*2, 400 - cvRound(histSobel.at<float>(i-1))),
             Point(i*2, 400 - cvRound(histSobel.at<float>(i))),
             Scalar(0,0,0), 2);
        line(histKirschImg,
             Point((i-1)*2, 400 - cvRound(histKirsch.at<float>(i-1))),
             Point(i*2, 400 - cvRound(histKirsch.at<float>(i))),
             Scalar(0,0,0), 2);
    }

    imwrite("./resultat/hist_prewitt.jpg", histPrewittImg);
    imwrite("./resultat/hist_sobel.jpg", histSobelImg);
    imwrite("./resultat/hist_kirsch.jpg", histKirschImg);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " ./image/Cathedrale-Lyon.jpg" << endl;
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Erreur : Impossible de charger l'image !" << endl;
        return -1;
    }

    applyMultiDirectionalFilters(image);

    waitKey(0);
    return 0;
}