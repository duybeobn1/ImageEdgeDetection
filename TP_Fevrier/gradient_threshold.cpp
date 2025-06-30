#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

/**
 * Fonction de convolution 2D
 * @param image : Image source en niveaux de gris.
 * @param kernel : Noyau de convolution (filtre).
 * @return Matrice résultante après convolution.
 */
Mat convolution2D(const Mat& image, const vector<vector<int>>& kernel) {
    int kSize = kernel.size();
    int offset = kSize / 2;
    Mat result = Mat::zeros(image.size(), CV_32F);

    // Balayage de l'image en ignorant les bords pour éviter le dépassement du noyau
    for (int i = offset; i < image.rows - offset; i++) {
        for (int j = offset; j < image.cols - offset; j++) {
            float sum = 0.0;

            // Application du noyau de convolution
            for (int ki = 0; ki < kSize; ki++) {
                for (int kj = 0; kj < kSize; kj++) {
                    sum += image.at<uchar>(i + ki - offset, j + kj - offset) * kernel[ki][kj];
                }
            }

            result.at<float>(i, j) = sum;
        }
    }

    return result;
}

/**
 * Applique un seuillage global (seuil unique)
 * @param gradient : Image du gradient.
 * @param threshold_value : Valeur du seuil.
 */
Mat globalThresholding(const Mat& gradient, double threshold_value) {
    Mat thresholded;
    threshold(gradient, thresholded, threshold_value, 255, THRESH_BINARY);
    return thresholded;
}

/**
 * Seuillage par hystérésis
 * @param gradient : Image du gradient.
 * @param lowThresh : Seuil bas.
 * @param highThresh : Seuil haut.
 */
Mat hysteresisThresholding(const Mat& gradient, double lowThresh, double highThresh) {
    Mat edges = Mat::zeros(gradient.size(), CV_8U);

    for (int i = 1; i < gradient.rows - 1; i++) {
        for (int j = 1; j < gradient.cols - 1; j++) {
            float val = gradient.at<float>(i, j);
            if (val >= highThresh) {
                edges.at<uchar>(i, j) = 255;
            } else if (val >= lowThresh) {
                // Vérifie les pixels voisins pour une connexion avec des pixels forts
                bool connected = false;
                for (int x = -1; x <= 1; x++) {
                    for (int y = -1; y <= 1; y++) {
                        if (gradient.at<float>(i + x, j + y) >= highThresh) {
                            connected = true;
                        }
                    }
                }
                if (connected) edges.at<uchar>(i, j) = 255;
            }
        }
    }

    return edges;
}

/**
 * Génère une image colorée en fonction des directions de gradient.
 * @param Gx : Gradient en X.
 * @param Gy : Gradient en Y.
 * @param G45 : Gradient en diagonale 45°.
 * @param G135 : Gradient en diagonale 135°.
 */
Mat colorizeGradient(const Mat& Gx, const Mat& Gy, const Mat& G45, const Mat& G135) {
    vector<Mat> channels(3);

    // Normalisation des gradients pour la visualisation
    normalize(abs(Gx), channels[0], 0, 255, NORM_MINMAX);   // Rouge : Gradient X
    normalize(abs(Gy), channels[1], 0, 255, NORM_MINMAX);   // Vert : Gradient Y
    normalize(abs(G45 + G135), channels[2], 0, 255, NORM_MINMAX);  // Bleu : Diagonales

    Mat coloredGradient;
    merge(channels, coloredGradient);
    return coloredGradient;
}
/**
 * Enumération des directions principales des gradients.
 */
enum DirectionCategory { HORIZONTAL, VERTICAL, DIAGONAL_45, DIAGONAL_135 };

/**
 * Détermine la direction dominante d'un gradient donné.
 */
DirectionCategory getDirectionCategory(float gx, float gy, float g45, float g135) {
  float maxVal = max(abs(gx), max(abs(gy), max(abs(g45), abs(g135))));
  if (abs(gx) == maxVal) return HORIZONTAL;
  else if (abs(gy) == maxVal) return VERTICAL;
  else if (abs(g45) == maxVal) return DIAGONAL_45;
  else return DIAGONAL_135;
}

/**
 * Suppression des non-maxima pour affiner les contours.
 */
Mat nonMaximumSuppression(const Mat& gradientMagnitude, const Mat& Gx, const Mat& Gy, const Mat& G45, const Mat& G135) {
  Mat suppressed = gradientMagnitude.clone();
  int rows = gradientMagnitude.rows;
  int cols = gradientMagnitude.cols;
  for (int i = 1; i < rows - 1; i++) {
      for (int j = 1; j < cols - 1; j++) {
          float gx = Gx.at<float>(i, j);
          float gy = Gy.at<float>(i, j);
          float g45 = G45.at<float>(i, j);
          float g135 = G135.at<float>(i, j);
          DirectionCategory category = getDirectionCategory(gx, gy, g45, g135);
          float mag = gradientMagnitude.at<float>(i, j);
          switch (category) {
              case HORIZONTAL:
                  if (mag < gradientMagnitude.at<float>(i, j-1) || mag < gradientMagnitude.at<float>(i, j+1)) suppressed.at<float>(i, j) = 0;
                  break;
              case VERTICAL:
                  if (mag < gradientMagnitude.at<float>(i-1, j) || mag < gradientMagnitude.at<float>(i+1, j)) suppressed.at<float>(i, j) = 0;
                  break;
              case DIAGONAL_45:
                  if (mag < gradientMagnitude.at<float>(i-1, j+1) || mag < gradientMagnitude.at<float>(i+1, j-1)) suppressed.at<float>(i, j) = 0;
                  break;
              case DIAGONAL_135:
                  if (mag < gradientMagnitude.at<float>(i-1, j-1) || mag < gradientMagnitude.at<float>(i+1, j+1)) suppressed.at<float>(i, j) = 0;
                  break;
          }
      }
  }
  return suppressed;
}

/**
 * Détection des contours avec affichage et seuillage.
 */
 void detectAndDisplayContours(const Mat& image) {
    vector<vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    vector<vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    vector<vector<int>> kernel45 = {{0, 1, 2}, {-1, 0, 1}, {-2, -1, 0}};
    vector<vector<int>> kernel135 = {{2, 1, 0}, {1, 0, -1}, {0, -1, -2}};

    Mat Gx = convolution2D(image, sobelX);
    Mat Gy = convolution2D(image, sobelY);
    Mat G45 = convolution2D(image, kernel45);
    Mat G135 = convolution2D(image, kernel135);

    Mat gradientMagnitude = Mat::zeros(image.size(), CV_32F);
    magnitude(Gx, Gy, gradientMagnitude);
    gradientMagnitude += abs(G45) + abs(G135);

    Mat suppressed = nonMaximumSuppression(gradientMagnitude, Gx, Gy, G45, G135);

    Mat coloredGradient = colorizeGradient(Gx, Gy, G45, G135);
    imshow("Gradient Multidirectionnel Coloré", coloredGradient);

    Mat suppressedNormalized;
    normalize(suppressed, suppressedNormalized, 0, 255, NORM_MINMAX, CV_8U);
    imshow("Gradient après suppression", suppressedNormalized);

    double globalThresh = mean(suppressed)[0];
    Mat globalThresholded = globalThresholding(suppressed, globalThresh);
    imshow("Seuillage Global", globalThresholded);

    double highThresh = 1.5 * globalThresh;
    double lowThresh = 0.5 * globalThresh;
    Mat hysteresisThresholded = hysteresisThresholding(suppressed, lowThresh, highThresh);
    imshow("Seuillage par Hystérésis", hysteresisThresholded);

    imwrite("./resultat/gradient_colore.jpg", coloredGradient);
    imwrite("./resultat/suppressed_gradient.jpg", suppressedNormalized);
    imwrite("./resultat/seuillage_global.jpg", globalThresholded);
    imwrite("./resultat/seuillage_hysteresis.jpg", hysteresisThresholded);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " ./image/ExempleSimple.jpg" << endl;
        return -1;
    }

    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Erreur : Impossible de charger l'image !" << endl;
        return -1;
    }

    detectAndDisplayContours(image);

    waitKey(0);
    return 0;
}