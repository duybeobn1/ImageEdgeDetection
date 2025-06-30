#include <QApplication>
#include <QComboBox>
#include <QFileDialog>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPixmap>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

// Variables globales pour stocker l'image
cv::Mat loadedImage, processedImage;
cv::Mat zoomImage(const cv::Mat &inputImage, double scale);
cv::Mat reduceImage(const cv::Mat &inputImage, double scale);
cv::Mat rotateImage(const cv::Mat &inputImage, double angle);
cv::Mat flipImage(const cv::Mat &inputImage, int flipCode);
void calculateHistogram(const cv::Mat &inputImage, std::vector<int> &histogram);
void equalizeHistogram(const cv::Mat &inputImage, cv::Mat &outputImage);
void stretchHistogram(const cv::Mat &inputImage, cv::Mat &outputImage);
cv::Mat zoomImageAtPoint(const cv::Mat &inputImage, double scale, int centerX,
                         int centerY);

// Convertir cv::Mat en QImage
QImage cvMatToQImage(const cv::Mat &mat) {
  if (mat.channels() == 3) {
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step,
                  QImage::Format_RGB888);
  } else if (mat.channels() == 1) {
    return QImage(mat.data, mat.cols, mat.rows, mat.step,
                  QImage::Format_Grayscale8);
  } else {
    return QImage();
  }
}

// Fonction de filtrage médian personnalisé
void applyMedianFilterCustom(const cv::Mat &inputImage, cv::Mat &outputImage,
                             int kernelSize) {
  if (kernelSize % 2 == 0) {
    std::cerr << "Error: Kernel size must be an odd number." << std::endl;
    return;
  }

  outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
  int radius = kernelSize / 2;

  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      std::vector<uchar> neighborhood;
      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          neighborhood.push_back(inputImage.at<uchar>(i + ki, j + kj));
        }
      }

      std::nth_element(neighborhood.begin(),
                       neighborhood.begin() + neighborhood.size() / 2,
                       neighborhood.end());
      uchar median = neighborhood[neighborhood.size() / 2];
      outputImage.at<uchar>(i, j) = median;
    }
  }
}

// Filtrage moyen personnalisé
void applyMeanFilterCustom(const cv::Mat &inputImage, cv::Mat &outputImage,
                           int kernelSize) {
  if (kernelSize % 2 == 0) {
    std::cerr << "Error: Kernel size must be an odd number." << std::endl;
    return;
  }

  outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
  int radius = kernelSize / 2;

  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      int sum = 0;
      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          sum += inputImage.at<uchar>(i + ki, j + kj);
        }
      }
      int meanVal = sum / (kernelSize * kernelSize);
      outputImage.at<uchar>(i, j) = static_cast<uchar>(meanVal);
    }
  }
}

// Flou gaussien personnalisé
void applyGaussianBlurCustom(const cv::Mat &inputImage, cv::Mat &outputImage,
                             int kernelSize, double sigma) {
  if (kernelSize % 2 == 0) {
    std::cerr << "Error: Kernel size must be an odd number." << std::endl;
    return;
  }

  int radius = kernelSize / 2;
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

  kernel /= sum;
  outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      double pixelValue = 0.0;
      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          double kernelVal = kernel.at<double>(ki + radius, kj + radius);
          pixelValue += inputImage.at<uchar>(i + ki, j + kj) * kernelVal;
        }
      }
      outputImage.at<uchar>(i, j) = static_cast<uchar>(pixelValue);
    }
  }
}

// Filtrage Sobel personnalisé (3x3)
void applySobelFilterCustom(const cv::Mat &inputImage, cv::Mat &outputImage) {
  if (inputImage.empty())
    return;

  std::vector<std::vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

  std::vector<std::vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int radius = 1;
  cv::Mat floatImage = cv::Mat::zeros(inputImage.size(), CV_32F);

  for (int i = radius; i < inputImage.rows - radius; ++i) {
    for (int j = radius; j < inputImage.cols - radius; ++j) {
      float gx = 0.0f;
      float gy = 0.0f;

      for (int ki = -radius; ki <= radius; ++ki) {
        for (int kj = -radius; kj <= radius; ++kj) {
          uchar pixelVal = inputImage.at<uchar>(i + ki, j + kj);
          gx += pixelVal * sobelX[ki + radius][kj + radius];
          gy += pixelVal * sobelY[ki + radius][kj + radius];
        }
      }

      float gradient = std::sqrt(gx * gx + gy * gy);
      floatImage.at<float>(i, j) = gradient;
    }
  }

  floatImage.convertTo(outputImage, CV_8U);
}

// Fonction pour appliquer un filtre en fonction du type choisi
void applyFilter(const QString &filterType, int kernelSize,
                 QLabel *imageLabel) {
  if (loadedImage.empty()) {
    QMessageBox::warning(nullptr, "Warning", "Please load an image first!");
    return;
  }

  // Selon le type de filtre, on utilise les fonctions personnalisées
  if (filterType == "Median Filter") {
    applyMedianFilterCustom(loadedImage, processedImage, kernelSize);
  } else if (filterType == "Mean Filter") {
    applyMeanFilterCustom(loadedImage, processedImage, kernelSize);
  } else if (filterType == "Gaussian Blur") {
    // On fixe sigma à 1.0 par exemple, ou demander à l'utilisateur via une
    // autre interface
    double sigma = 1.0;
    applyGaussianBlurCustom(loadedImage, processedImage, kernelSize, sigma);
  } else if (filterType == "Sobel Filter") {
    // Le Sobel est codé pour un noyau 3x3. Si kernelSize != 3, on peut avertir.
    if (kernelSize != 3) {
      QMessageBox::information(
          nullptr, "Info", "Sobel filter is fixed to a 3x3 kernel. Using 3.");
    }
    applySobelFilterCustom(loadedImage, processedImage);
  } else {
    QMessageBox::warning(nullptr, "Warning", "Unknown filter type!");
    return;
  }

  QImage qImage = cvMatToQImage(processedImage);
  imageLabel->setPixmap(QPixmap::fromImage(qImage).scaled(imageLabel->size(),
                                                          Qt::KeepAspectRatio));
}

// Fonction pour charger une image
void loadImage(QLabel *imageLabel) {
  QString filePath = QFileDialog::getOpenFileName(
      nullptr, "Open Image File", "", "Images (*.png *.jpg *.bmp)");
  if (!filePath.isEmpty()) {
    loadedImage = cv::imread(filePath.toStdString(), cv::IMREAD_GRAYSCALE);
    if (!loadedImage.empty()) {
      QImage qImage = cvMatToQImage(loadedImage);
      imageLabel->setPixmap(QPixmap::fromImage(qImage).scaled(
          imageLabel->size(), Qt::KeepAspectRatio));
    } else {
      QMessageBox::critical(nullptr, "Error", "Failed to load the image.");
    }
  }
}

// Fonction pour appliquer les transformations géométriques
void applyTransformation(const QString &transformType, QLabel *imageLabel) {
  if (loadedImage.empty()) {
    QMessageBox::warning(nullptr, "Warning", "Please load an image first!");
    return;
  }

  if (transformType == "Zoom") {
    processedImage = zoomImage(loadedImage, 2.0);
  } else if (transformType == "Reduce") {
    processedImage = reduceImage(loadedImage, 0.5);
  } else if (transformType == "Rotate") {
    processedImage = rotateImage(loadedImage, 45.0);
  } else if (transformType == "Flip") {
    processedImage = flipImage(loadedImage, 1);
  } else {
    QMessageBox::warning(nullptr, "Warning", "Unknown transformation!");
    return;
  }

  if (!processedImage.empty()) {
    QImage qImage = cvMatToQImage(processedImage);
    if (qImage.isNull()) {
      QMessageBox::critical(
          nullptr, "Error",
          "Failed to convert processed image to display format!");
      return;
    }
    imageLabel->setPixmap(QPixmap::fromImage(qImage).scaled(
        imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
  } else {
    QMessageBox::warning(
        nullptr, "Error",
        "Processing failed! Check the input scale or transformation.");
  }
}

cv::Mat zoomImage(const cv::Mat &inputImage, double scale) {
  if (scale <= 0.0) {
    std::cerr << "Error: Scale factor must be greater than 0!" << std::endl;
    return cv::Mat();
  }

  int newWidth = static_cast<int>(inputImage.cols * scale);
  int newHeight = static_cast<int>(inputImage.rows * scale);
  cv::Mat outputImage(newHeight, newWidth, inputImage.type());

  for (int y = 0; y < newHeight; ++y) {
    for (int x = 0; x < newWidth; ++x) {
      double srcX = x / scale;
      double srcY = y / scale;

      int x1 = static_cast<int>(srcX);
      int y1 = static_cast<int>(srcY);
      int x2 = std::min(x1 + 1, inputImage.cols - 1);
      int y2 = std::min(y1 + 1, inputImage.rows - 1);

      double dx = srcX - x1;
      double dy = srcY - y1;

      if (inputImage.channels() == 3) { // Color Image
        cv::Vec3b q11 = inputImage.at<cv::Vec3b>(y1, x1);
        cv::Vec3b q12 = inputImage.at<cv::Vec3b>(y2, x1);
        cv::Vec3b q21 = inputImage.at<cv::Vec3b>(y1, x2);
        cv::Vec3b q22 = inputImage.at<cv::Vec3b>(y2, x2);

        for (int c = 0; c < 3; ++c) {
          double val = q11[c] * (1 - dx) * (1 - dy) + q21[c] * dx * (1 - dy) +
                       q12[c] * (1 - dx) * dy + q22[c] * dx * dy;
          outputImage.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(val);
        }
      } else { // Grayscale Image
        uchar q11 = inputImage.at<uchar>(y1, x1);
        uchar q12 = inputImage.at<uchar>(y2, x1);
        uchar q21 = inputImage.at<uchar>(y1, x2);
        uchar q22 = inputImage.at<uchar>(y2, x2);

        double val = q11 * (1 - dx) * (1 - dy) + q21 * dx * (1 - dy) +
                     q12 * (1 - dx) * dy + q22 * dx * dy;
        outputImage.at<uchar>(y, x) = static_cast<uchar>(val);
      }
    }
  }
  return outputImage;
}

cv::Mat reduceImage(const cv::Mat &inputImage, double scale) {
  if (scale <= 0.0 || scale >= 1.0) {
    std::cerr << "Error: Scale factor for reduction must be > 0 and < 1.0!"
              << std::endl;
    return cv::Mat();
  }

  // Calculate new dimensions
  int newWidth = static_cast<int>(inputImage.cols * scale);
  int newHeight = static_cast<int>(inputImage.rows * scale);

  // Initialize output image
  cv::Mat outputImage = cv::Mat::zeros(newHeight, newWidth, inputImage.type());

  // Perform bilinear interpolation for downscaling
  for (int y = 0; y < newHeight; ++y) {
    for (int x = 0; x < newWidth; ++x) {
      // Map output pixel to source coordinates
      double srcX = x / scale;
      double srcY = y / scale;

      // Calculate the four neighboring pixels
      int x1 = static_cast<int>(srcX);
      int y1 = static_cast<int>(srcY);
      int x2 = std::min(x1 + 1, inputImage.cols - 1);
      int y2 = std::min(y1 + 1, inputImage.rows - 1);

      double dx = srcX - x1;
      double dy = srcY - y1;

      // Bilinear interpolation
      if (inputImage.channels() == 3) { // For color images
        for (int c = 0; c < 3; ++c) {
          double value =
              (1 - dx) * (1 - dy) * inputImage.at<cv::Vec3b>(y1, x1)[c] +
              dx * (1 - dy) * inputImage.at<cv::Vec3b>(y1, x2)[c] +
              (1 - dx) * dy * inputImage.at<cv::Vec3b>(y2, x1)[c] +
              dx * dy * inputImage.at<cv::Vec3b>(y2, x2)[c];

          outputImage.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(value);
        }
      } else if (inputImage.channels() == 1) { // For grayscale images
        double value = (1 - dx) * (1 - dy) * inputImage.at<uchar>(y1, x1) +
                       dx * (1 - dy) * inputImage.at<uchar>(y1, x2) +
                       (1 - dx) * dy * inputImage.at<uchar>(y2, x1) +
                       dx * dy * inputImage.at<uchar>(y2, x2);

        outputImage.at<uchar>(y, x) = static_cast<uchar>(value);
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
  int newWidth =
      static_cast<int>(inputImage.cols * cosTheta + inputImage.rows * sinTheta);
  int newHeight =
      static_cast<int>(inputImage.cols * sinTheta + inputImage.rows * cosTheta);

  // Create a new canvas for the rotated image
  cv::Mat outputImage(newHeight, newWidth, inputImage.type(),
                      cv::Scalar(0, 0, 0));

  // Calculate the centers
  int centerX = inputImage.cols / 2;
  int centerY = inputImage.rows / 2;
  int newCenterX = newWidth / 2;
  int newCenterY = newHeight / 2;

  // Perform rotation manually
  for (int y = 0; y < newHeight; ++y) {
    for (int x = 0; x < newWidth; ++x) {
      // Map the coordinates back to the source image
      int srcX =
          static_cast<int>((x - newCenterX) * std::cos(-radian) -
                           (y - newCenterY) * std::sin(-radian) + centerX);
      int srcY =
          static_cast<int>((x - newCenterX) * std::sin(-radian) +
                           (y - newCenterY) * std::cos(-radian) + centerY);

      // Check bounds and copy pixel
      if (srcX >= 0 && srcX < inputImage.cols && srcY >= 0 &&
          srcY < inputImage.rows) {
        outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(srcY, srcX);
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
      if (flipCode == 0) {                // Vertical flip
        if (inputImage.channels() == 3) { // Color image
          outputImage.at<cv::Vec3b>(rows - y - 1, x) =
              inputImage.at<cv::Vec3b>(y, x);
        } else { // Grayscale image
          outputImage.at<uchar>(rows - y - 1, x) = inputImage.at<uchar>(y, x);
        }
      } else if (flipCode == 1) {         // Horizontal flip
        if (inputImage.channels() == 3) { // Color image
          outputImage.at<cv::Vec3b>(y, cols - x - 1) =
              inputImage.at<cv::Vec3b>(y, x);
        } else { // Grayscale image
          outputImage.at<uchar>(y, cols - x - 1) = inputImage.at<uchar>(y, x);
        }
      } else if (flipCode == -1) {        // Both horizontal and vertical flip
        if (inputImage.channels() == 3) { // Color image
          outputImage.at<cv::Vec3b>(rows - y - 1, cols - x - 1) =
              inputImage.at<cv::Vec3b>(y, x);
        } else { // Grayscale image
          outputImage.at<uchar>(rows - y - 1, cols - x - 1) =
              inputImage.at<uchar>(y, x);
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

void equalizeHistogram(const cv::Mat &inputImage, cv::Mat &outputImage) {
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

cv::Mat zoomImageAtPoint(const cv::Mat &inputImage, double scale,
                         int centerX = -1, int centerY = -1) {
  // Automatically set center to the middle of the image if no center is
  // provided
  if (centerX == -1)
    centerX = inputImage.cols / 2;
  if (centerY == -1)
    centerY = inputImage.rows / 2;

  // Calculate the dimensions of the ROI
  int newWidth = static_cast<int>(inputImage.cols / scale);
  int newHeight = static_cast<int>(inputImage.rows / scale);

  // Ensure center coordinates are within bounds
  centerX = std::clamp(centerX, newWidth / 2, inputImage.cols - newWidth / 2);
  centerY = std::clamp(centerY, newHeight / 2, inputImage.rows - newHeight / 2);

  // Define the region of interest (ROI) around the center point
  int xStart = centerX - newWidth / 2;
  int yStart = centerY - newHeight / 2;

  // Create an output image with scaled dimensions
  cv::Mat zoomedImage(newHeight * scale, newWidth * scale, inputImage.type());

  // Perform bilinear interpolation to scale up the ROI
  for (int y = 0; y < zoomedImage.rows; ++y) {
    for (int x = 0; x < zoomedImage.cols; ++x) {
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

      if (inputImage.channels() == 3) { // Color image
        for (int c = 0; c < 3; ++c) {
          uchar p1 = inputImage.at<cv::Vec3b>(y1, x1)[c];
          uchar p2 = inputImage.at<cv::Vec3b>(y1, x2)[c];
          uchar p3 = inputImage.at<cv::Vec3b>(y2, x1)[c];
          uchar p4 = inputImage.at<cv::Vec3b>(y2, x2)[c];

          uchar interpolated =
              static_cast<uchar>((1 - a) * (1 - b) * p1 + a * (1 - b) * p2 +
                                 (1 - a) * b * p3 + a * b * p4);

          zoomedImage.at<cv::Vec3b>(y, x)[c] = interpolated;
        }
      } else { // Grayscale image
        uchar p1 = inputImage.at<uchar>(y1, x1);
        uchar p2 = inputImage.at<uchar>(y1, x2);
        uchar p3 = inputImage.at<uchar>(y2, x1);
        uchar p4 = inputImage.at<uchar>(y2, x2);

        uchar interpolated =
            static_cast<uchar>((1 - a) * (1 - b) * p1 + a * (1 - b) * p2 +
                               (1 - a) * b * p3 + a * b * p4);

        zoomedImage.at<uchar>(y, x) = interpolated;
      }
    }
  }

  return zoomedImage;
}

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);

  QWidget window;
  window.setWindowTitle("Image Processing");

  QVBoxLayout *layout = new QVBoxLayout();

  QLabel *imageLabel = new QLabel();
  imageLabel->setFixedSize(500, 400);
  imageLabel->setStyleSheet("border: 1px solid black;");
  layout->addWidget(imageLabel);

  QPushButton *loadButton = new QPushButton("Load Image");
  layout->addWidget(loadButton);

  QComboBox *filterComboBox = new QComboBox();
  filterComboBox->addItem("Median Filter");
  filterComboBox->addItem("Mean Filter");
  filterComboBox->addItem("Gaussian Blur");
  filterComboBox->addItem("Sobel Filter");
  filterComboBox->addItem("Zoom");
  filterComboBox->addItem("Reduce");
  filterComboBox->addItem("Rotate");
  filterComboBox->addItem("Flip");
  filterComboBox->addItem("Equalize Histogram");
  filterComboBox->addItem("Zoom at Point");
  layout->addWidget(filterComboBox);

  QSpinBox *kernelSizeSpinBox = new QSpinBox();
  kernelSizeSpinBox->setRange(3, 15);
  kernelSizeSpinBox->setSingleStep(2);
  kernelSizeSpinBox->setValue(3);
  layout->addWidget(kernelSizeSpinBox);

  QLabel *additionalLabel = new QLabel("Extra Parameter:");
  QLineEdit *extraParameter = new QLineEdit();
  layout->addWidget(additionalLabel);
  layout->addWidget(extraParameter);

  QPushButton *applyButton = new QPushButton("Apply Filter");
  layout->addWidget(applyButton);

  QObject::connect(loadButton, &QPushButton::clicked,
                   [&]() { loadImage(imageLabel); });

  QObject::connect(applyButton, &QPushButton::clicked, [&]() {
    QString filterType = filterComboBox->currentText();
    int kernelSize = kernelSizeSpinBox->value();
    QString param = extraParameter->text();

    if (filterType == "Zoom" || filterType == "Reduce") {
      double scale = param.toDouble();
      if (scale <= 0.0) {
        QMessageBox::warning(nullptr, "Warning", "Invalid scale value!");
        return;
      }
      processedImage = (filterType == "Zoom") ? zoomImage(loadedImage, scale)
                                              : reduceImage(loadedImage, scale);
    } else if (filterType == "Zoom at Point") {
      double scale = param.toDouble();
      if (scale <= 0.0) {
        QMessageBox::warning(nullptr, "Warning", "Invalid scale value!");
        return;
      }
      processedImage = zoomImageAtPoint(loadedImage, scale);
    } else if (filterType == "Rotate") {
      double angle = param.toDouble();
      processedImage = rotateImage(loadedImage, angle);
    } else if (filterType == "Flip") {
      int flipCode = param.toInt();
      processedImage = flipImage(loadedImage, flipCode);
    } else if (filterType == "Equalize Histogram") {
      equalizeHistogram(loadedImage, processedImage);
    } else {
      applyFilter(filterType, kernelSize, imageLabel);
      return;
    }

    if (!processedImage.empty()) {
      QImage qImage = cvMatToQImage(processedImage);
      imageLabel->setPixmap(QPixmap::fromImage(qImage).scaled(
          imageLabel->size(), Qt::KeepAspectRatio));
    } else {
      QMessageBox::warning(nullptr, "Error", "Processing failed!");
    }
  });

  window.setLayout(layout);
  window.show();

  return app.exec();
}
