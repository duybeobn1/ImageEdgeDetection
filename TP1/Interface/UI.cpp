#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Function prototypes
void displayMenu();
cv::Mat loadImage();
void applyMedianFilter(const cv::Mat &inputImage, cv::Mat &outputImage,
                       int kernelSize);
void applyMeanFilter(const cv::Mat &inputImage, cv::Mat &outputImage,
                     int kernelSize);
void applyGaussianBlur(const cv::Mat &inputImage, cv::Mat &outputImage,
                       int kernelSize, double sigma);
void applySobelFilter(const cv::Mat &inputImage, cv::Mat &outputImage);

cv::Mat zoomImage(const cv::Mat &inputImage, double scale);
cv::Mat reduceImage(const cv::Mat &inputImage, double scale);
cv::Mat rotateImage(const cv::Mat &inputImage, double angle);
cv::Mat flipImage(const cv::Mat &inputImage, int flipCode);
cv::Mat zoomImageAtPoint(const cv::Mat &inputImage, double scale, int centerX,
                         int centerY);

// Histogramme
void calculateHistogram(const cv::Mat &inputImage, std::vector<int> &histogram);
void equalizeHistogramManual(const cv::Mat &inputImage, cv::Mat &outputImage);
void stretchHistogram(const cv::Mat &inputImage, cv::Mat &outputImage);
void plotHistogram(const std::vector<int> &histogram,
                   const std::string &windowName);
void calculateCumulativeHistogram(const std::vector<int> &histogram,
                                  std::vector<int> &cumulativeHistogram);
int main() {
  cv::Mat inputImage, outputImage;
  bool running = true;

  while (running) {
    displayMenu();
    int choice;
    std::cin >> choice;

    switch (choice) {
    case 1: {
      inputImage = loadImage();
      if (inputImage.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
      } else {
        std::cout << "Image loaded successfully!" << std::endl;
      }
      break;
    }
    case 2: { // Apply Median Filter
      if (inputImage.empty()) {
        std::cerr << "Error: No image loaded. Please load an image first."
                  << std::endl;
        break;
      }

      int kernelSize;
      std::cout << "Enter kernel size (odd number, e.g., 3, 5, 7): ";
      std::cin >> kernelSize;

      applyMedianFilter(inputImage, outputImage, kernelSize);
      cv::imshow("Custom Median Filtered Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 3: { // Apply Mean Filter
      if (inputImage.empty()) {
        std::cerr << "Error: No image loaded. Please load an image first."
                  << std::endl;
        break;
      }

      int kernelSize;
      std::cout << "Enter kernel size (odd number, e.g., 3, 5, 7): ";
      std::cin >> kernelSize;

      applyMeanFilter(inputImage, outputImage, kernelSize);
      cv::imshow("Custom Mean Filtered Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 4: { // Apply Gaussian Blur
      if (inputImage.empty()) {
        std::cerr << "Error: No image loaded. Please load an image first."
                  << std::endl;
        break;
      }

      int kernelSize;
      double sigma;
      std::cout << "Enter kernel size (odd number, e.g., 3, 5, 7): ";
      std::cin >> kernelSize;
      std::cout << "Enter Gaussian sigma (e.g., 1.0): ";
      std::cin >> sigma;

      applyGaussianBlur(inputImage, outputImage, kernelSize, sigma);
      cv::imshow("Custom Gaussian Filtered Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 5: {
      if (inputImage.empty()) {
        std::cerr << "Error: No image loaded. Please load an image first."
                  << std::endl;
        break;
      }

      applySobelFilter(inputImage, outputImage);
      cv::imshow("Sobel Filtered Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 6: { // Zoom Image
      double scale;
      std::cout << "Enter zoom scale (e.g., 2.0 for 2x zoom): ";
      std::cin >> scale;

      outputImage = zoomImage(inputImage, scale);
      cv::imshow("Zoomed Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 7: { // Reduce Image
      double scale;
      std::cout << "Enter reduction scale (e.g., 0.5 for half-size): ";
      std::cin >> scale;

      outputImage = reduceImage(inputImage, scale);
      cv::imshow("Reduced Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 8: { // Rotate Image
      double angle;
      std::cout << "Enter rotation angle (degrees): ";
      std::cin >> angle;

      outputImage = rotateImage(inputImage, angle);
      cv::imshow("Rotated Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 9: { // Flip Image
      int flipCode;
      std::cout << "Enter flip code (0=Vertical, 1=Horizontal, -1=Both): ";
      std::cin >> flipCode;

      outputImage = flipImage(inputImage, flipCode);
      cv::imshow("Flipped Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 10: { // Histogram Equalization
      equalizeHistogramManual(inputImage, outputImage);
      cv::imshow("Equalized Histogram Image", outputImage);
      cv::waitKey(0);
      break;
    }
    case 11: { // Zoom Image at Specific Point
      if (inputImage.empty()) {
        std::cerr << "Error: No image loaded. Please load an image first."
                  << std::endl;
        break;
      }

      double scale;
      int centerX, centerY;
      std::cout << "Enter zoom scale (e.g., 2.0 for 2x zoom): ";
      std::cin >> scale;
      std::cout << "Enter center X coordinate: ";
      std::cin >> centerX;
      std::cout << "Enter center Y coordinate: ";
      std::cin >> centerY;

      outputImage = zoomImageAtPoint(inputImage, scale, centerX, centerY);
      cv::imshow("Zoomed Image at Point", outputImage);
      cv::waitKey(0);
      break;
    }
    case 0: {
      running = false;
      std::cout << "Exiting program..." << std::endl;
      break;
    }
    default:
      std::cerr << "Invalid choice. Please try again." << std::endl;
    }
  }

  return 0;
}

void displayMenu() {
  std::cout << "\nImage Processing Console Interface\n";
  std::cout << "1. Load Image\n";
  std::cout << "2. Apply Median Filter\n";
  std::cout << "3. Apply Mean Filter\n";
  std::cout << "4. Apply Gaussian Blur\n";
  std::cout << "5. Apply Sobel Filter (Custom)\n";
  std::cout << "6. Zoom Image\n";
  std::cout << "7. Reduce Image\n";
  std::cout << "8. Rotate Image\n";
  std::cout << "9. Flip Image\n";
  std::cout << "10. Equalize Histogram\n";
  std::cout << "11. Zoom Image at Specific Point\n";
  std::cout << "0. Exit\n";
  std::cout << "Enter your choice: ";
}

cv::Mat loadImage() {
  std::string imagePath;
  std::cout << "Enter the path to the image: ";
  std::cin >> imagePath;
  return cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
}

// Fonction de filtrage médian personnalisée
void applyMedianFilter(const cv::Mat &inputImage, cv::Mat &outputImage,
                       int kernelSize) {
  // Vérifier que le kernelSize est impair
  if (kernelSize % 2 == 0) {
    std::cerr << "Error: Kernel size must be an odd number." << std::endl;
    return;
  }

  // Initialiser l'image de sortie avec la même taille et le même type
  outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

  // Rayon du noyau (par exemple, kernelSize = 3 -> radius = 1)
  int radius = kernelSize / 2;

  // Parcourir chaque pixel de l'image
  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      // Collecter les pixels dans la fenêtre glissante
      std::vector<uchar> neighborhood;
      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          neighborhood.push_back(inputImage.at<uchar>(i + ki, j + kj));
        }
      }

      // Trier les valeurs pour trouver la médiane
      std::nth_element(neighborhood.begin(),
                       neighborhood.begin() + neighborhood.size() / 2,
                       neighborhood.end());
      uchar median = neighborhood[neighborhood.size() / 2];

      // Assigner la valeur médiane au pixel central
      outputImage.at<uchar>(i, j) = median;
    }
  }

  std::cout << "Custom median filter applied with kernel size " << kernelSize
            << "." << std::endl;
}

void applyMeanFilter(const cv::Mat &inputImage, cv::Mat &outputImage,
                     int kernelSize) {
  // Vérifier que le kernelSize est impair
  if (kernelSize % 2 == 0) {
    std::cerr << "Error: Kernel size must be an odd number." << std::endl;
    return;
  }

  // Initialiser l'image de sortie avec la même taille et le même type
  outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

  // Rayon du noyau (par exemple, kernelSize = 3 -> radius = 1)
  int radius = kernelSize / 2;

  // Parcourir chaque pixel de l'image
  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      int sum = 0;

      // Parcourir la fenêtre autour du pixel
      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          sum += inputImage.at<uchar>(i + ki, j + kj);
        }
      }

      // Calculer la moyenne et assigner la valeur au pixel central
      int mean = sum / (kernelSize * kernelSize);
      outputImage.at<uchar>(i, j) = static_cast<uchar>(mean);
    }
  }

  std::cout << "Custom mean filter applied with kernel size " << kernelSize
            << "." << std::endl;
}

// Fonction pour appliquer un flou gaussien personnalisé
void applyGaussianBlur(const cv::Mat &inputImage, cv::Mat &outputImage,
                       int kernelSize, double sigma) {
  // Vérifier que le kernelSize est impair
  if (kernelSize % 2 == 0) {
    std::cerr << "Error: Kernel size must be an odd number." << std::endl;
    return;
  }

  // Calculer le rayon du noyau
  int radius = kernelSize / 2;

  // Générer le noyau gaussien
  cv::Mat kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_64F);
  double sum = 0.0;

  for (int x = -radius; x <= radius; ++x) {
    for (int y = -radius; y <= radius; ++y) {
      double value = std::exp(-(x * x + y * y) / (2 * sigma * sigma)) /
                     (2 * M_PI * sigma * sigma);
      kernel.at<double>(x + radius, y + radius) = value;
      sum += value;
    }
  }

  // Normaliser le noyau pour que la somme des poids soit égale à 1
  kernel /= sum;

  // Initialiser l'image de sortie
  outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

  // Appliquer le noyau à chaque pixel
  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      double pixelValue = 0.0;

      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          int pixelX = i + ki;
          int pixelY = j + kj;

          double kernelValue = kernel.at<double>(ki + radius, kj + radius);
          pixelValue += inputImage.at<uchar>(pixelX, pixelY) * kernelValue;
        }
      }

      // Assigner la valeur calculée au pixel central
      outputImage.at<uchar>(i, j) = static_cast<uchar>(pixelValue);
    }
  }

  std::cout << "Custom Gaussian blur applied with kernel size " << kernelSize
            << " and sigma " << sigma << "." << std::endl;
}

// Fonction de filtrage Sobel personnalisé
void applySobelFilter(const cv::Mat &inputImage, cv::Mat &outputImage) {
  // Vérifier que l'image est valide
  if (inputImage.empty()) {
    std::cerr << "Erreur : Image vide dans applySobelFilter." << std::endl;
    return;
  }

  // Initialiser les noyaux Sobel (horizontal et vertical)
  std::vector<std::vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

  std::vector<std::vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  // Rayon du noyau
  int radius = 1;

  // Initialiser l'image de sortie
  outputImage = cv::Mat::zeros(inputImage.size(), CV_32F);

  // Appliquer les noyaux Sobel
  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      float gx = 0.0, gy = 0.0;

      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          int pixelX = i + ki;
          int pixelY = j + kj;

          gx += inputImage.at<uchar>(pixelX, pixelY) *
                sobelX[ki + radius][kj + radius];
          gy += inputImage.at<uchar>(pixelX, pixelY) *
                sobelY[ki + radius][kj + radius];
        }
      }

      float gradient = std::sqrt(gx * gx + gy * gy);
      outputImage.at<float>(i, j) = gradient;
    }
  }

  // Convertir en CV_8U pour l'affichage
  outputImage.convertTo(outputImage, CV_8U);

  std::cout << "Filtre Sobel appliqué avec succès." << std::endl;
}

cv::Mat zoomImage(const cv::Mat& inputImage, double scale) {
    // Validate scale factor
    if (scale <= 0.0) {
        std::cerr << "Error: Scale factor must be greater than 0!" << std::endl;
        return cv::Mat();
    }

    // Compute new dimensions
    int newWidth = static_cast<int>(inputImage.cols * scale);
    int newHeight = static_cast<int>(inputImage.rows * scale);

    // Create output image
    cv::Mat outputImage = cv::Mat::zeros(newHeight, newWidth, inputImage.type());

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int srcX = static_cast<int>(x / scale);
            int srcY = static_cast<int>(y / scale);

            // Ensure within bounds
            if (srcX >= inputImage.cols || srcY >= inputImage.rows)
                continue;

            if (inputImage.channels() == 3) { // Color image
                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(srcY, srcX);
            } else if (inputImage.channels() == 1) { // Grayscale image
                outputImage.at<uchar>(y, x) = inputImage.at<uchar>(srcY, srcX);
            }
        }
    }

    return outputImage;
}


cv::Mat reduceImage(const cv::Mat &inputImage, double scale) {
    // Validate the scale factor
    if (scale <= 0.0 || scale >= 1.0) {
        std::cerr << "Error: Scale factor for reduction must be > 0 and < 1.0!" << std::endl;
        return cv::Mat();
    }

    // Compute new dimensions
    int newWidth = static_cast<int>(inputImage.cols * scale);
    int newHeight = static_cast<int>(inputImage.rows * scale);

    // Create output image
    cv::Mat outputImage = cv::Mat::zeros(newHeight, newWidth, inputImage.type());

    // Scaling factors to map output pixels to input regions
    double scaleX = 1.0 / scale;
    double scaleY = 1.0 / scale;

    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int xStart = static_cast<int>(x * scaleX);
            int yStart = static_cast<int>(y * scaleY);
            int xEnd = std::min(static_cast<int>((x + 1) * scaleX), inputImage.cols);
            int yEnd = std::min(static_cast<int>((y + 1) * scaleY), inputImage.rows);

            int sumB = 0, sumG = 0, sumR = 0, count = 0;

            for (int row = yStart; row < yEnd; ++row) {
                for (int col = xStart; col < xEnd; ++col) {
                    if (inputImage.channels() == 3) { // Color image
                        cv::Vec3b pixel = inputImage.at<cv::Vec3b>(row, col);
                        sumB += pixel[0];
                        sumG += pixel[1];
                        sumR += pixel[2];
                        count++;
                    } else { // Grayscale image
                        sumB += inputImage.at<uchar>(row, col);
                        count++;
                    }
                }
            }

            if (count > 0) {
                if (inputImage.channels() == 3) { // Color image
                    outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        static_cast<uchar>(sumB / count),
                        static_cast<uchar>(sumG / count),
                        static_cast<uchar>(sumR / count));
                } else { // Grayscale image
                    outputImage.at<uchar>(y, x) = static_cast<uchar>(sumB / count);
                }
            }
        }
    }

    return outputImage;
}

cv::Mat rotateImage(const cv::Mat &inputImage, double angle) {
    // Convert angle to radians
    double radian = angle * CV_PI / 180.0;
    double cosTheta = std::abs(std::cos(radian));
    double sinTheta = std::abs(std::sin(radian));

    // Calculate new dimensions for the rotated image
    int newWidth = static_cast<int>(inputImage.rows * sinTheta + inputImage.cols * cosTheta);
    int newHeight = static_cast<int>(inputImage.rows * cosTheta + inputImage.cols * sinTheta);

    // Create an empty output image
    cv::Mat outputImage(newHeight, newWidth, inputImage.type(), cv::Scalar(0, 0, 0));

    // Find center of the input and output images
    cv::Point2f centerInput(inputImage.cols / 2.0, inputImage.rows / 2.0);
    cv::Point2f centerOutput(newWidth / 2.0, newHeight / 2.0);

    // Iterate over each pixel in the output image
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            // Map output coordinates (x, y) back to input coordinates
            double newX = (x - centerOutput.x) * std::cos(-radian) - 
                          (y - centerOutput.y) * std::sin(-radian) + centerInput.x;
            double newY = (x - centerOutput.x) * std::sin(-radian) + 
                          (y - centerOutput.y) * std::cos(-radian) + centerInput.y;

            // Check if the mapped coordinates are within bounds
            if (newX >= 0 && newX < inputImage.cols && newY >= 0 && newY < inputImage.rows) {
                if (inputImage.channels() == 3) { // Color image
                    outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(static_cast<int>(newY), static_cast<int>(newX));
                } else { // Grayscale image
                    outputImage.at<uchar>(y, x) = inputImage.at<uchar>(static_cast<int>(newY), static_cast<int>(newX));
                }
            }
        }
    }

    return outputImage;
}

cv::Mat flipImage(const cv::Mat &inputImage, int flipCode) {
    // Create an output image of the same type and size
    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

    int rows = inputImage.rows;
    int cols = inputImage.cols;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (flipCode == 0) { // Vertical flip
                if (inputImage.channels() == 3) { // Color image
                    outputImage.at<cv::Vec3b>(rows - y - 1, x) = inputImage.at<cv::Vec3b>(y, x);
                } else { // Grayscale image
                    outputImage.at<uchar>(rows - y - 1, x) = inputImage.at<uchar>(y, x);
                }
            } else if (flipCode == 1) { // Horizontal flip
                if (inputImage.channels() == 3) { // Color image
                    outputImage.at<cv::Vec3b>(y, cols - x - 1) = inputImage.at<cv::Vec3b>(y, x);
                } else { // Grayscale image
                    outputImage.at<uchar>(y, cols - x - 1) = inputImage.at<uchar>(y, x);
                }
            } else if (flipCode == -1) { // Both horizontal and vertical flip
                if (inputImage.channels() == 3) { // Color image
                    outputImage.at<cv::Vec3b>(rows - y - 1, cols - x - 1) = inputImage.at<cv::Vec3b>(y, x);
                } else { // Grayscale image
                    outputImage.at<uchar>(rows - y - 1, cols - x - 1) = inputImage.at<uchar>(y, x);
                }
            }
        }
    }

    return outputImage;
}


void calculateHistogram(const cv::Mat &inputImage,
                        std::vector<int> &histogram) {
  histogram.assign(256, 0);
  for (int i = 0; i < inputImage.rows; ++i) {
    for (int j = 0; j < inputImage.cols; ++j) {
      histogram[inputImage.at<uchar>(i, j)]++;
    }
  }
}

void calculateCumulativeHistogram(const std::vector<int> &histogram,
                                  std::vector<int> &cumulativeHistogram) {
  cumulativeHistogram.resize(histogram.size(), 0);
  cumulativeHistogram[0] = histogram[0];
  for (size_t i = 1; i < histogram.size(); ++i) {
    cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
  }
}

void equalizeHistogramManual(const cv::Mat &inputImage, cv::Mat &outputImage) {
  // Vérification : l'image doit être en niveaux de gris
  if (inputImage.channels() != 1) {
    std::cerr << "Erreur : L'image doit être en niveaux de gris !" << std::endl;
    return;
  }

  // Calculer l'histogramme et l'histogramme cumulé
  std::vector<int> histogram, cumulativeHistogram;
  calculateHistogram(inputImage, histogram);
  calculateCumulativeHistogram(histogram, cumulativeHistogram);

  // Nombre total de pixels dans l'image
  int totalPixels = inputImage.rows * inputImage.cols;

  // Calculer la nouvelle valeur de chaque niveau de gris
  std::vector<uchar> equalizedLUT(256, 0);
  int cmin =
      *std::find_if(cumulativeHistogram.begin(), cumulativeHistogram.end(),
                    [](int value) { return value > 0; });

  for (int i = 0; i < 256; ++i) {
    equalizedLUT[i] = static_cast<uchar>((cumulativeHistogram[i] - cmin) *
                                         255.0 / (totalPixels - cmin));
  }

  // Appliquer la transformation LUT (Look-Up Table)
  outputImage = cv::Mat(inputImage.size(), inputImage.type());
  for (int y = 0; y < inputImage.rows; ++y) {
    for (int x = 0; x < inputImage.cols; ++x) {
      int pixelValue = inputImage.at<uchar>(y, x);
      outputImage.at<uchar>(y, x) = equalizedLUT[pixelValue];
    }
  }
}

void findMinMaxGray(const cv::Mat &inputImage, double &minGray,
                    double &maxGray) {
  minGray = 255; // Initialiser au maximum possible pour les niveaux de gris
  maxGray = 0;   // Initialiser au minimum possible pour les niveaux de gris

  for (int y = 0; y < inputImage.rows; ++y) {
    for (int x = 0; x < inputImage.cols; ++x) {
      uchar pixelValue = inputImage.at<uchar>(y, x);
      if (pixelValue < minGray)
        minGray = pixelValue; // Trouver le minimum
      if (pixelValue > maxGray)
        maxGray = pixelValue; // Trouver le maximum
    }
  }
}

void stretchHistogram(const cv::Mat &inputImage, cv::Mat &outputImage) {
  // Vérification : l'image doit être en niveaux de gris
  if (inputImage.channels() != 1) {
    std::cerr << "Erreur : L'image doit être en niveaux de gris !" << std::endl;
    return;
  }

  double minGray, maxGray;
  findMinMaxGray(inputImage, minGray, maxGray);

  // Créer l'image de sortie avec le même format
  outputImage = cv::Mat(inputImage.size(), inputImage.type());

  // Appliquer la transformation linéaire pour étirer l'histogramme
  for (int y = 0; y < inputImage.rows; ++y) {
    for (int x = 0; x < inputImage.cols; ++x) {
      uchar pixelValue = inputImage.at<uchar>(y, x);
      uchar newPixelValue = static_cast<uchar>((pixelValue - minGray) *
                                               (255.0 / (maxGray - minGray)));
      outputImage.at<uchar>(y, x) = newPixelValue;
    }
  }

  std::cout << "Histogramme étiré : minGray = " << minGray
            << ", maxGray = " << maxGray << std::endl;
}

void plotHistogram(const std::vector<int> &histogram,
                   const std::string &windowName) {
  int histSize = histogram.size();
  int histHeight = 400;
  cv::Mat histImage(histHeight, histSize, CV_8UC1, cv::Scalar(255));

  int maxValue = *std::max_element(histogram.begin(), histogram.end());
  for (int i = 0; i < histSize; ++i) {
    int barHeight =
        cv::saturate_cast<int>((double)histogram[i] / maxValue * histHeight);
    cv::line(histImage, cv::Point(i, histHeight),
             cv::Point(i, histHeight - barHeight), cv::Scalar(0));
  }

  cv::imshow(windowName, histImage);
}

cv::Mat zoomImageAtPoint(const cv::Mat &inputImage, double scale, int centerX,
                         int centerY) {
  int newWidth = static_cast<int>(inputImage.cols / scale);
  int newHeight = static_cast<int>(inputImage.rows / scale);

  // Ensure center coordinates are within bounds
  centerX = std::clamp(centerX, newWidth / 2, inputImage.cols - newWidth / 2);
  centerY = std::clamp(centerY, newHeight / 2, inputImage.rows - newHeight / 2);

  // Define the region of interest (ROI) around the center point
  int xStart = centerX - newWidth / 2;
  int yStart = centerY - newHeight / 2;

  cv::Mat zoomedImage(inputImage.size(), inputImage.type(), cv::Scalar(0));

  // Manual interpolation to scale up the ROI
  for (int y = 0; y < inputImage.rows; ++y) {
    for (int x = 0; x < inputImage.cols; ++x) {
      // Map destination coordinates to source coordinates in the ROI
      double srcX = xStart + (x / scale);
      double srcY = yStart + (y / scale);

      // Bilinear interpolation
      int x1 = static_cast<int>(std::floor(srcX));
      int y1 = static_cast<int>(std::floor(srcY));
      int x2 = std::min(x1 + 1, inputImage.cols - 1);
      int y2 = std::min(y1 + 1, inputImage.rows - 1);

      double a = srcX - x1;
      double b = srcY - y1;

      for (int c = 0; c < inputImage.channels(); ++c) {
        uchar p1 = inputImage.at<cv::Vec3b>(y1, x1)[c];
        uchar p2 = inputImage.at<cv::Vec3b>(y1, x2)[c];
        uchar p3 = inputImage.at<cv::Vec3b>(y2, x1)[c];
        uchar p4 = inputImage.at<cv::Vec3b>(y2, x2)[c];

        uchar interpolated =
            static_cast<uchar>((1 - a) * (1 - b) * p1 + a * (1 - b) * p2 +
                               (1 - a) * b * p3 + a * b * p4);

        zoomedImage.at<cv::Vec3b>(y, x)[c] = interpolated;
      }
    }
  }

  return zoomedImage;
}