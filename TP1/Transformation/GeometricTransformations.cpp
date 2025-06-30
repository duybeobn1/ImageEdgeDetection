#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Fonction pour zoomer à partir d'un point précis
cv::Mat zoomImageAtPoint(const cv::Mat& inputImage, double scale, int centerX, int centerY) {
    // Calculer la nouvelle taille de l'image après zoom
    int newWidth = static_cast<int>(inputImage.cols / scale);
    int newHeight = static_cast<int>(inputImage.rows / scale);
    
    // S'assurer que les coordonnées du centre sont dans les limites de l'image
    centerX = std::clamp(centerX, newWidth / 2, inputImage.cols - newWidth / 2);
    centerY = std::clamp(centerY, newHeight / 2, inputImage.rows - newHeight / 2);

    // Définir la région d'intérêt (ROI) centrée sur le point spécifié
    int xStart = centerX - newWidth / 2;
    int yStart = centerY - newHeight / 2;

    // Créer une nouvelle image avec la même taille que l'image d'entrée
    cv::Mat zoomedImage(inputImage.size(), inputImage.type(), cv::Scalar(0));

    // Interpolation bilinéaire manuelle pour agrandir la région d'intérêt
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            double srcX = xStart + (x / scale);
            double srcY = yStart + (y / scale);

            // Calculer les voisins pour l'interpolation
            int x1 = static_cast<int>(std::floor(srcX));
            int y1 = static_cast<int>(std::floor(srcY));
            int x2 = std::min(x1 + 1, inputImage.cols - 1);
            int y2 = std::min(y1 + 1, inputImage.rows - 1);

            double a = srcX - x1;
            double b = srcY - y1;

            // Interpolation pour chaque canal (R, G, B)
            for (int c = 0; c < inputImage.channels(); ++c) {
                uchar p1 = inputImage.at<cv::Vec3b>(y1, x1)[c];
                uchar p2 = inputImage.at<cv::Vec3b>(y1, x2)[c];
                uchar p3 = inputImage.at<cv::Vec3b>(y2, x1)[c];
                uchar p4 = inputImage.at<cv::Vec3b>(y2, x2)[c];

                uchar interpolated = static_cast<uchar>(
                    (1 - a) * (1 - b) * p1 +
                    a * (1 - b) * p2 +
                    (1 - a) * b * p3 +
                    a * b * p4
                );

                zoomedImage.at<cv::Vec3b>(y, x)[c] = interpolated;
            }
        }
    }

    return zoomedImage;
}

// Fonction pour zoomer (agrandir l'image entière)
cv::Mat zoomImage(const cv::Mat& inputImage, double scale) {
    int newWidth = static_cast<int>(inputImage.cols * scale);
    int newHeight = static_cast<int>(inputImage.rows * scale);
    cv::Mat outputImage(newHeight, newWidth, inputImage.type());

    // Copier les pixels avec mise à l'échelle
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int srcX = static_cast<int>(x / scale);
            int srcY = static_cast<int>(y / scale);

            if (srcX < inputImage.cols && srcY < inputImage.rows) {
                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(srcY, srcX);
            }
        }
    }

    return outputImage;
}

// Fonction pour réduire l'image (diminuer sa taille)
cv::Mat reduceImage(const cv::Mat& inputImage, double scale) {
    return zoomImage(inputImage, scale); // Utilise zoomImage avec un facteur < 1
}

// Fonction pour faire pivoter l'image
cv::Mat rotateImage(const cv::Mat& inputImage, double angle) {
    int newWidth = inputImage.cols;
    int newHeight = inputImage.rows;
    cv::Mat outputImage(newHeight, newWidth, inputImage.type(), cv::Scalar(0, 0, 0));

    double radian = angle * M_PI / 180.0;
    double cosTheta = std::cos(radian);
    double sinTheta = std::sin(radian);
    int cx = inputImage.cols / 2;
    int cy = inputImage.rows / 2;

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            // Calculer les nouvelles coordonnées après rotation
            int srcX = static_cast<int>((x - cx) * cosTheta + (y - cy) * sinTheta + cx);
            int srcY = static_cast<int>((y - cy) * cosTheta - (x - cx) * sinTheta + cy);

            if (srcX >= 0 && srcX < inputImage.cols && srcY >= 0 && srcY < inputImage.rows) {
                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(srcY, srcX);
            }
        }
    }

    return outputImage;
}

// Fonction pour retourner l'image (flip)
cv::Mat flipImage(const cv::Mat& inputImage, int flipCode) {
    cv::Mat outputImage(inputImage.rows, inputImage.cols, inputImage.type());

    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            if (flipCode == 0) { // Flip vertical
                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(inputImage.rows - y - 1, x);
            } else if (flipCode == 1) { // Flip horizontal
                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(y, inputImage.cols - x - 1);
            } else if (flipCode == -1) { // Flip vertical et horizontal
                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(inputImage.rows - y - 1, inputImage.cols - x - 1);
            }
        }
    }

    return outputImage;
}

int main() {
    // Charger l'image d'entree
    cv::Mat inputImage = cv::imread("../donnee/Autre.png");
    if (inputImage.empty()) {
        std::cerr << "Error: Unable to load the image." << std::endl;
        return -1;
    }

    // Afficher l'image originale
    cv::imshow("Original Image", inputImage);
    cv::imwrite("../resultat/original_image.png", inputImage); // Sauvegarder l'image originale

    // Zoom a partir du centre
    cv::Mat zoomedAtPointImage = zoomImageAtPoint(inputImage, 2.0, inputImage.cols / 2, inputImage.rows / 2);
    cv::imshow("Zoomed Image (Center Point)", zoomedAtPointImage);
    cv::imwrite("../resultat/zoomed_at_center.png", zoomedAtPointImage); // Sauvegarder l'image zoomee au centre

    // Zoom complet (sur toute l'image)
    cv::Mat zoomedImage = zoomImage(inputImage, 2.0);
    cv::imshow("Zoomed Image", zoomedImage);
    cv::imwrite("../resultat/zoomed_image.png", zoomedImage); // Sauvegarder l'image zoomee

    // Reduction d'image
    cv::Mat reducedImage = reduceImage(inputImage, 0.5);
    cv::imshow("Reduced Image", reducedImage);
    cv::imwrite("../resultat/reduced_image.png", reducedImage); // Sauvegarder l'image reduite

    // Rotation de l'image
    cv::Mat rotatedImage = rotateImage(inputImage, 45);
    cv::imshow("Rotated Image (45 degrees)", rotatedImage);
    cv::imwrite("../resultat/rotated_image.png", rotatedImage); // Sauvegarder l'image tournee

    // Flip horizontal
    cv::Mat flippedImage = flipImage(inputImage, 1);
    cv::imshow("Flipped Image (Horizontal)", flippedImage);
    cv::imwrite("../resultat/flipped_image.png", flippedImage); // Sauvegarder l'image retournee horizontalement

    // Attente d'une touche pour fermer les fenetres
    cv::waitKey(0);
    return 0;
}
