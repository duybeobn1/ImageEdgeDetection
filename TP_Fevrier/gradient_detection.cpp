#include <cmath> // Pour std::abs, std::sqrt, std::atan2
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/**
 * Fonction pour normaliser un masque en divisant par la somme des valeurs
 * absolues.
 * @param kernel Masque non normalisé de type vector<vector<int>>
 * @return Masque normalisé de type vector<vector<float>>
 */
vector<vector<float>> normalizeKernel(const vector<vector<int>> &kernel) {
  float sumAbs = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sumAbs += std::abs(kernel[i][j]);
    }
  }
  if (sumAbs == 0.0) {
    sumAbs = 1.0;
  }
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
  Mat result = Mat::zeros(rows, cols, CV_32F);

  for (int i = 1; i < rows - 1; i++) {
    for (int j = 1; j < cols - 1; j++) {
      float sum = 0.0;
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
 * Fonction pour le cas bidirectionnel : calcule le module et la pente en
 * utilisant les filtres X et Y.
 * @param image Image en entrée (niveaux de gris)
 */
void applyBidirectionalFilters(const Mat &image) {
  // Filtres Prewitt non normalisés (directions X et Y)
  vector<vector<int>> prewittX_int = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
  vector<vector<int>> prewittY_int = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};

  // Normalisation des filtres
  vector<vector<float>> prewittX = normalizeKernel(prewittX_int);
  vector<vector<float>> prewittY = normalizeKernel(prewittY_int);

  // Application des convolutions pour X et Y
  Mat Gx = applyConvolution(image, prewittX);
  Mat Gy = applyConvolution(image, prewittY);

  // Calcul du module (magnitude) et de la pente (direction)
  Mat magnitude = Mat::zeros(image.size(), CV_32F);
  Mat direction = Mat::zeros(image.size(), CV_32F);
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float gx = Gx.at<float>(i, j);
      float gy = Gy.at<float>(i, j);
      // Module : sqrt(gx² + gy²)
      magnitude.at<float>(i, j) = std::sqrt(gx * gx + gy * gy);
      // Pente (direction en degrés) : atan2(gy, gx) * 180 / PI
      direction.at<float>(i, j) = std::atan2(gy, gx) * 180 / M_PI;
    }
  }

  // Normalisation pour affichage
  Mat magnitudeNorm;
  normalize(magnitude, magnitudeNorm, 0, 255, NORM_MINMAX, CV_8U);

  // Visualisation de la direction (optionnel)
  Mat directionVis = Mat::zeros(image.size(), CV_8UC3);
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float dir = direction.at<float>(i, j);
      // Normaliser dir (-180 à +180) vers 0-180 pour la teinte HSV
      float hue = (dir + 180.0) / 2.0; // Teinte entre 0 et 180
      directionVis.at<Vec3b>(i, j) = Vec3b(hue, 255, 255);
    }
  }
  cvtColor(directionVis, directionVis, COLOR_HSV2BGR);

  // Affichage et sauvegarde des résultats
  imshow("Magnitude (Module) Bidirectionnelle", magnitudeNorm);
  imshow("Direction (Pente) Bidirectionnelle", directionVis);
  imwrite("./resultat/magnitude_bidirectional.jpg", magnitudeNorm);
  imwrite("./resultat/direction_bidirectional.jpg", directionVis);
}

/**
 * Fonction mise à jour pour le cas multidirectionnel : calcule le module et la
 * pente en utilisant 4 directions.
 * @param image Image en entrée (niveaux de gris)
 */
void applyMultiDirectionalFilters(const Mat &image) {
  // ---------------- Filtres Prewitt (non normalisés) ----------------
  vector<vector<int>> prewittX_int = {
      {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}}; // 0° (X)
  vector<vector<int>> prewittY_int = {
      {-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}; // 90° (Y)
  vector<vector<int>> prewitt45_int = {
      {0, 1, 1}, {-1, 0, 1}, {-1, -1, 0}}; // 45°
  vector<vector<int>> prewitt135_int = {
      {1, 1, 0}, {1, 0, -1}, {0, -1, -1}}; // 135°

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

  // ---------------- Calcul du module et de la pente pour Prewitt
  // (multidirectionnel) ----------------
  Mat magnitudePrewitt = Mat::zeros(image.size(), CV_32F);
  Mat directionPrewitt = Mat::zeros(image.size(), CV_32F);

  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float maxVal = 0.0;
      int maxDir = -1;
      for (int d = 0; d < 4; d++) {
        float absVal = std::abs(prewittResults[d].at<float>(i, j));
        if (absVal > maxVal) {
          maxVal = absVal;
          maxDir = d;
        }
      }
      magnitudePrewitt.at<float>(i, j) = maxVal;
      // Assignation de la direction en degrés (basée sur la direction maximale)
      float angle;
      switch (maxDir) {
      case 0:
        angle = 0.0;
        break; // X (0°)
      case 1:
        angle = 90.0;
        break; // Y (90°)
      case 2:
        angle = 45.0;
        break; // 45°
      case 3:
        angle = 135.0;
        break; // 135°
      default:
        angle = 0.0;
      }
      directionPrewitt.at<float>(i, j) = angle;
    }
  }

  // Normalisation pour affichage
  Mat magnitudeNormPrewitt;
  normalize(magnitudePrewitt, magnitudeNormPrewitt, 0, 255, NORM_MINMAX, CV_8U);

  // Visualisation de la direction
  Mat directionVisPrewitt = Mat::zeros(image.size(), CV_8UC3);
  for (int i = 0; i < image.rows; i++) {
    for (int j = 0; j < image.cols; j++) {
      float dir = directionPrewitt.at<float>(i, j);
      // Normaliser dir (0 à 180) pour la teinte HSV
      float hue = dir; // Déjà entre 0 et 135°, mais on peut normaliser à 0-180
      directionVisPrewitt.at<Vec3b>(i, j) = Vec3b(hue, 255, 255);
    }
  }
  cvtColor(directionVisPrewitt, directionVisPrewitt, COLOR_HSV2BGR);

  // Affichage et sauvegarde pour Prewitt
  imshow("Magnitude (Module) Multidirectionnelle Prewitt",
         magnitudeNormPrewitt);
  imshow("Direction (Pente) Multidirectionnelle Prewitt", directionVisPrewitt);
  imwrite("./resultat/magnitude_multidirectional_prewitt.jpg",
          magnitudeNormPrewitt);
  imwrite("./resultat/direction_multidirectional_prewitt.jpg",
          directionVisPrewitt);

  // ---------------- Gestion des autres filtres (Sobel et Kirsch) pour le
  // module et la pente ---------------- (Même processus que pour Prewitt, peut
  // être ajouté si nécessaire)
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

  applyBidirectionalFilters(image);
  applyMultiDirectionalFilters(image);

  waitKey(0);
  return 0;
}