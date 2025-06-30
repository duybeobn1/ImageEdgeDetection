// #include "mywindow.h"
// #include <QDebug>
// #include <QImage>
// #include <QPixmap>
// #include <QTabWidget>
// #include <cmath>

// // ------------------
// // Fonctions OpenCV utilitaires (similaires aux précédentes)
// // ------------------

// // Normalisation d’un noyau 3x3 (int) en float
// static std::vector<std::vector<float>> normalizeKernel(const std::vector<std::vector<int>>& kernel)
// {
//     float sumAbs = 0.f;
//     for(int i = 0; i < 3; i++){
//         for(int j = 0; j < 3; j++){
//             sumAbs += std::abs(kernel[i][j]);
//         }
//     }
//     if(sumAbs == 0) sumAbs = 1.f;
//     std::vector<std::vector<float>> normK(3, std::vector<float>(3, 0.f));
//     for(int i = 0; i < 3; i++){
//         for(int j = 0; j < 3; j++){
//             normK[i][j] = kernel[i][j] / sumAbs;
//         }
//     }
//     return normK;
// }

// // Convolution 3x3
// static cv::Mat applyConvolution(const cv::Mat& gray, const std::vector<std::vector<float>>& kernel)
// {
//     cv::Mat result = cv::Mat::zeros(gray.size(), CV_32F);
//     for(int y = 1; y < gray.rows - 1; y++){
//         for(int x = 1; x < gray.cols - 1; x++){
//             float sum = 0.f;
//             for(int j = -1; j <= 1; j++){
//                 for(int i = -1; i <= 1; i++){
//                     float val = static_cast<float>(gray.at<uchar>(y + j, x + i));
//                     sum += val * kernel[j + 1][i + 1];
//                 }
//             }
//             result.at<float>(y, x) = sum;
//         }
//     }
//     return result;
// }

// // Combine plusieurs directions (somme des valeurs absolues)
// static cv::Mat combineDirections(const std::vector<cv::Mat>& dirs)
// {
//     cv::Mat combined = cv::Mat::zeros(dirs[0].size(), CV_32F);
//     for(const auto &m : dirs){
//         combined += cv::abs(m);
//     }
//     return combined;
// }

// // ------------------
// // Constructeur
// // ------------------
// MyWindow::MyWindow(QWidget *parent)
//     : QMainWindow(parent)
//     , m_inputImageLabel(new QLabel(this))
//     , m_outputImageLabel(new QLabel(this))
//     , m_filterCombo(new QComboBox(this))
//     , m_thresholdSlider(new QSlider(Qt::Horizontal, this))
//     , m_loadButton(new QPushButton("Charger Image", this))
//     , m_applyBidirectionnelleButton(new QPushButton("Appliquer Détection Bidirectionnelle", this))
//     , m_biMagLabel(new QLabel(this))
//     , m_biDirLabel(new QLabel(this))
//     , m_applyContoursButton(new QPushButton("Appliquer Contours par Seuillage", this))
//     , m_colorGradLabel(new QLabel(this))
//     , m_globalThreshLabel(new QLabel(this))
//     , m_hystThreshLabel(new QLabel(this))
//     , m_hystLowSlider(new QSlider(Qt::Horizontal, this))
//     , m_hystHighSlider(new QSlider(Qt::Horizontal, this))
//     , m_tabWidget(new QTabWidget(this))
//     , m_tabFiltrage(new QWidget(this))
//     , m_tabBidirectionnel(new QWidget(this))
//     , m_tabContours(new QWidget(this))
//     , m_currentFilter(0)
//     , m_thresholdValue(50)
// {
//     // --- Onglet Filtrage (mode existant) ---
//     m_thresholdSlider->setRange(0, 255);
//     m_thresholdSlider->setValue(m_thresholdValue);
//     m_filterCombo->addItem("Prewitt");
//     m_filterCombo->addItem("Sobel");
//     m_filterCombo->addItem("Kirsch");

//     QHBoxLayout* controlLayout = new QHBoxLayout;
//     controlLayout->addWidget(m_loadButton);
//     controlLayout->addWidget(m_filterCombo);
//     controlLayout->addWidget(m_thresholdSlider);

//     QHBoxLayout* imagesLayout = new QHBoxLayout;
//     imagesLayout->addWidget(m_inputImageLabel);
//     imagesLayout->addWidget(m_outputImageLabel);

//     QVBoxLayout* filtrageLayout = new QVBoxLayout;
//     filtrageLayout->addLayout(controlLayout);
//     filtrageLayout->addLayout(imagesLayout);
//     m_tabFiltrage->setLayout(filtrageLayout);

//     // --- Onglet Détection Bidirectionnelle ---
//     QVBoxLayout* biLayout = new QVBoxLayout;
//     biLayout->addWidget(m_applyBidirectionnelleButton);
//     QHBoxLayout* biImagesLayout = new QHBoxLayout;
//     biImagesLayout->addWidget(m_biMagLabel);
//     biImagesLayout->addWidget(m_biDirLabel);
//     biLayout->addLayout(biImagesLayout);
//     m_tabBidirectionnel->setLayout(biLayout);

//     // --- Onglet Contours par Seuillage ---
//     QVBoxLayout* contoursLayout = new QVBoxLayout;
//     contoursLayout->addWidget(m_applyContoursButton);
    
//     // Ajout des sliders pour le seuillage par hystérésis
//     m_hystLowSlider->setRange(0, 255);
//     m_hystHighSlider->setRange(0, 255);
//     // Valeurs par défaut (vous pouvez les ajuster)
//     m_hystLowSlider->setValue(50);
//     m_hystHighSlider->setValue(200);
//     QHBoxLayout* hystSliderLayout = new QHBoxLayout;
//     QLabel* lowLabel = new QLabel("Hyst Low", this);
//     QLabel* highLabel = new QLabel("Hyst High", this);
//     hystSliderLayout->addWidget(lowLabel);
//     hystSliderLayout->addWidget(m_hystLowSlider);
//     hystSliderLayout->addWidget(highLabel);
//     hystSliderLayout->addWidget(m_hystHighSlider);
//     contoursLayout->addLayout(hystSliderLayout);

//     QHBoxLayout* contoursImagesLayout = new QHBoxLayout;
//     contoursImagesLayout->addWidget(m_colorGradLabel);
//     contoursImagesLayout->addWidget(m_globalThreshLabel);
//     contoursImagesLayout->addWidget(m_hystThreshLabel);
//     contoursLayout->addLayout(contoursImagesLayout);
//     m_tabContours->setLayout(contoursLayout);

//     // --- Ajouter les onglets au QTabWidget ---
//     m_tabWidget->addTab(m_tabFiltrage, "Filtrage");
//     m_tabWidget->addTab(m_tabBidirectionnel, "Détection Bidirectionnelle");
//     m_tabWidget->addTab(m_tabContours, "Contours par Seuillage");

//     setCentralWidget(m_tabWidget);

//     // Connecter les signaux/slots
//     connect(m_loadButton, &QPushButton::clicked, this, &MyWindow::onLoadImage);
//     connect(m_filterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
//             this, &MyWindow::onFilterTypeChanged);
//     connect(m_thresholdSlider, &QSlider::valueChanged,
//             this, &MyWindow::onThresholdChanged);
//     connect(m_applyBidirectionnelleButton, &QPushButton::clicked,
//             this, &MyWindow::onApplyBidirectionnelle);
//     connect(m_applyContoursButton, &QPushButton::clicked,
//             this, &MyWindow::onApplyContours);
//     // Connexion des sliders d'hystérésis
//     connect(m_hystLowSlider, &QSlider::valueChanged, this, &MyWindow::onHystThresholdChanged);
//     connect(m_hystHighSlider, &QSlider::valueChanged, this, &MyWindow::onHystThresholdChanged);
// }

// // Destructeur
// MyWindow::~MyWindow()
// {
// }

// // ------------------
// // Slots
// // ------------------

// void MyWindow::onLoadImage()
// {
//     QString fileName = QFileDialog::getOpenFileName(this, "Ouvrir une image", "",
//                                                     "Images (*.png *.jpg *.jpeg *.bmp)");
//     if(fileName.isEmpty()){
//         return;
//     }
//     m_originalMat = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
//     if(m_originalMat.empty()){
//         qWarning() << "Impossible de charger l'image :" << fileName;
//         return;
//     }
//     QImage qimgIn((const uchar*)m_originalMat.data, m_originalMat.cols, m_originalMat.rows,
//                   m_originalMat.step, QImage::Format_Grayscale8);
//     m_inputImageLabel->setPixmap(QPixmap::fromImage(qimgIn).scaled(m_inputImageLabel->size(),
//                                                                    Qt::KeepAspectRatio,
//                                                                    Qt::SmoothTransformation));
//     // Appliquer le traitement Filtrage (onglet 1)
//     applyFilterAndThreshold();
// }

// void MyWindow::onFilterTypeChanged(int index)
// {
//     m_currentFilter = index;
//     applyFilterAndThreshold();
// }

// void MyWindow::onThresholdChanged(int value)
// {
//     m_thresholdValue = value;
//     applyFilterAndThreshold();
// }

// void MyWindow::onApplyBidirectionnelle()
// {
//     if(m_originalMat.empty()) return;
//     applyBidirectionalDetection();
// }

// void MyWindow::onApplyContours()
// {
//     if(m_originalMat.empty()) return;
//     applyContourDetection();
// }

// void MyWindow::onHystThresholdChanged(int /*value*/)
// {
//     if(m_originalMat.empty()) return;
//     // Recalcule le seuillage par hystérésis avec les nouvelles valeurs de slider
//     applyContourDetection();
// }

// // ------------------
// // Traitements
// // ------------------

// // Mode Filtrage (basé sur l'interface initiale)
// void MyWindow::applyFilterAndThreshold()
// {
//     if(m_originalMat.empty()) return;
//     // Définition des noyaux selon le filtre choisi
//     static std::vector<std::vector<int>> prewittX_int  = {{-1,0,1},{-1,0,1},{-1,0,1}};
//     static std::vector<std::vector<int>> prewittY_int  = {{-1,-1,-1},{0,0,0},{1,1,1}};
//     static std::vector<std::vector<int>> prewitt45_int = {{0,1,1},{-1,0,1},{-1,-1,0}};
//     static std::vector<std::vector<int>> prewitt135_int= {{1,1,0},{1,0,-1},{0,-1,-1}};
   
//     static std::vector<std::vector<int>> sobelX_int    = {{-1,0,1},{-2,0,2},{-1,0,1}};
//     static std::vector<std::vector<int>> sobelY_int    = {{-1,-2,-1},{0,0,0},{1,2,1}};
//     static std::vector<std::vector<int>> sobel45_int   = {{0,1,2},{-1,0,1},{-2,-1,0}};
//     static std::vector<std::vector<int>> sobel135_int  = {{2,1,0},{1,0,-1},{0,-1,-2}};

//     static std::vector<std::vector<int>> kirschX_int   = {{-3,-3,5},{-3,0,5},{-3,-3,5}};
//     static std::vector<std::vector<int>> kirschY_int   = {{-3,-3,-3},{5,0,-3},{5,5,-3}};
//     static std::vector<std::vector<int>> kirsch45_int  = {{-3,5,5},{-3,0,5},{-3,-3,-3}};
//     static std::vector<std::vector<int>> kirsch135_int = {{5,5,-3},{5,0,-3},{-3,-3,-3}};

//     // Sélection des noyaux
//     std::vector<std::vector<int>> kx, ky, k45, k135;
//     if(m_currentFilter == 0){ // Prewitt
//         kx   = prewittX_int;  ky   = prewittY_int;
//         k45  = prewitt45_int; k135 = prewitt135_int;
//     }
//     else if(m_currentFilter == 1){ // Sobel
//         kx   = sobelX_int;  ky   = sobelY_int;
//         k45  = sobel45_int; k135 = sobel135_int;
//     }
//     else { // Kirsch
//         kx   = kirschX_int;   ky   = kirschY_int;
//         k45  = kirsch45_int;  k135 = kirsch135_int;
//     }
//     // Normaliser et appliquer les convolutions
//     auto nkx   = normalizeKernel(kx);
//     auto nky   = normalizeKernel(ky);
//     auto nk45  = normalizeKernel(k45);
//     auto nk135 = normalizeKernel(k135);
//     cv::Mat gx   = applyConvolution(m_originalMat, nkx);
//     cv::Mat gy   = applyConvolution(m_originalMat, nky);
//     cv::Mat g45  = applyConvolution(m_originalMat, nk45);
//     cv::Mat g135 = applyConvolution(m_originalMat, nk135);
//     cv::Mat combined = combineDirections({gx, gy, g45, g135});
//     // Seuillage binaire
//     cv::Mat thresholded = cv::Mat::zeros(combined.size(), CV_8U);
//     for(int y = 0; y < combined.rows; y++){
//         for(int x = 0; x < combined.cols; x++){
//             float val = combined.at<float>(y, x);
//             thresholded.at<uchar>(y, x) = (val >= m_thresholdValue) ? 255 : 0;
//         }
//     }
//     m_resultMat = thresholded;
//     QImage qimgOut((const uchar*)m_resultMat.data, m_resultMat.cols,
//                    m_resultMat.rows, m_resultMat.step,
//                    QImage::Format_Grayscale8);
//     m_outputImageLabel->setPixmap(QPixmap::fromImage(qimgOut).scaled(m_outputImageLabel->size(),
//                                                                      Qt::KeepAspectRatio,
//                                                                      Qt::SmoothTransformation));
// }

// // Mode Détection Bidirectionnelle
// void MyWindow::applyBidirectionalDetection()
// {
//     // Utiliser les filtres Prewitt pour X et Y
//     static std::vector<std::vector<int>> prewittX_int = {{-1,0,1},{-1,0,1},{-1,0,1}};
//     static std::vector<std::vector<int>> prewittY_int = {{-1,-1,-1},{0,0,0},{1,1,1}};
//     auto prewittX = normalizeKernel(prewittX_int);
//     auto prewittY = normalizeKernel(prewittY_int);
//     cv::Mat Gx = applyConvolution(m_originalMat, prewittX);
//     cv::Mat Gy = applyConvolution(m_originalMat, prewittY);
//     // Calcul de la magnitude et de la direction
//     cv::Mat magnitude(m_originalMat.size(), CV_32F, cv::Scalar(0));
//     cv::Mat direction(m_originalMat.size(), CV_32F, cv::Scalar(0));
//     for (int i = 0; i < m_originalMat.rows; i++){
//         for (int j = 0; j < m_originalMat.cols; j++){
//             float gx = Gx.at<float>(i, j);
//             float gy = Gy.at<float>(i, j);
//             magnitude.at<float>(i, j) = std::sqrt(gx * gx + gy * gy);
//             direction.at<float>(i, j) = std::atan2(gy, gx) * 180.0f / CV_PI;
//         }
//     }
//     // Normaliser la magnitude pour affichage
//     cv::Mat magnitudeNorm;
//     cv::normalize(magnitude, magnitudeNorm, 0, 255, cv::NORM_MINMAX, CV_8U);

//     // Création d'une image pour visualiser la direction (conversion en HSV puis BGR)
//     cv::Mat directionVis(m_originalMat.size(), CV_8UC3, cv::Scalar(0,0,0));
//     for (int i = 0; i < m_originalMat.rows; i++){
//         for (int j = 0; j < m_originalMat.cols; j++){
//             float dir = direction.at<float>(i, j);
//             // Normaliser de -180..+180 vers 0..180
//             uchar hue = static_cast<uchar>((dir + 180.f) / 2.f);
//             directionVis.at<cv::Vec3b>(i, j) = cv::Vec3b(hue, 255, 255);
//         }
//     }
//     cv::cvtColor(directionVis, directionVis, cv::COLOR_HSV2BGR);

//     // Convertir pour affichage dans Qt
//     QImage qMag((const uchar*)magnitudeNorm.data, magnitudeNorm.cols, magnitudeNorm.rows,
//                 magnitudeNorm.step, QImage::Format_Grayscale8);
//     m_biMagLabel->setPixmap(QPixmap::fromImage(qMag).scaled(m_biMagLabel->size(),
//                                                              Qt::KeepAspectRatio,
//                                                              Qt::SmoothTransformation));
//     QImage qDir((const uchar*)directionVis.data, directionVis.cols, directionVis.rows,
//                 directionVis.step, QImage::Format_BGR888);
//     m_biDirLabel->setPixmap(QPixmap::fromImage(qDir).scaled(m_biDirLabel->size(),
//                                                              Qt::KeepAspectRatio,
//                                                              Qt::SmoothTransformation));
// }

// // Mode Contours par Seuillage
// void MyWindow::applyContourDetection()
// {
//     // Correction des filtres Sobel
//     static std::vector<std::vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
//     static std::vector<std::vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
//     static std::vector<std::vector<int>> kernel45 = {{0, 1, 2}, {-1, 0, 1}, {-2, -1, 0}};
//     static std::vector<std::vector<int>> kernel135 = {{2, 1, 0}, {1, 0, -1}, {0, -1, -2}};
    
//     // Convolution pour chaque direction
//     cv::Mat Gx = applyConvolution(m_originalMat, normalizeKernel(sobelX));
//     cv::Mat Gy = applyConvolution(m_originalMat, normalizeKernel(sobelY));
//     cv::Mat G45 = applyConvolution(m_originalMat, normalizeKernel(kernel45));
//     cv::Mat G135 = applyConvolution(m_originalMat, normalizeKernel(kernel135));
    
//     // Calcul du gradient combiné
//     cv::Mat gradientMagnitude;
//     cv::magnitude(Gx, Gy, gradientMagnitude);
//     gradientMagnitude = gradientMagnitude + cv::abs(G45) + cv::abs(G135);
    
//     // Création d'une image colorée (canaux : X -> rouge, Y -> vert, diagonales -> bleu)
//     std::vector<cv::Mat> channels(3);
//     cv::normalize(cv::abs(Gx), channels[0], 0, 255, cv::NORM_MINMAX, CV_8U);
//     cv::normalize(cv::abs(Gy), channels[1], 0, 255, cv::NORM_MINMAX, CV_8U);
//     cv::normalize(cv::abs(G45 + G135), channels[2], 0, 255, cv::NORM_MINMAX, CV_8U);
//     cv::Mat coloredGradient;
//     cv::merge(channels, coloredGradient);
    
//     // Seuillage global
//     double globalThresh = cv::mean(gradientMagnitude)[0];
//     cv::Mat globalThresholded;
//     cv::threshold(gradientMagnitude, globalThresholded, globalThresh, 255, cv::THRESH_BINARY);
//     // Conversion en CV_8U pour un affichage correct
//     globalThresholded.convertTo(globalThresholded, CV_8U);
    
//     // Seuillage par hystérésis avec valeurs issues des sliders
//     double lowThresh = m_hystLowSlider->value();
//     double highThresh = m_hystHighSlider->value();
//     cv::Mat hysteresisThresholded = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);
//     for (int i = 1; i < gradientMagnitude.rows - 1; i++){
//         for (int j = 1; j < gradientMagnitude.cols - 1; j++){
//             float val = gradientMagnitude.at<float>(i, j);
//             if (val >= highThresh){
//                 hysteresisThresholded.at<uchar>(i, j) = 255;
//             } else if (val >= lowThresh){
//                 bool connected = false;
//                 for (int x = -1; x <= 1; x++){
//                     for (int y = -1; y <= 1; y++){
//                         if (gradientMagnitude.at<float>(i + x, j + y) >= highThresh)
//                             connected = true;
//                     }
//                 }
//                 if (connected)
//                     hysteresisThresholded.at<uchar>(i, j) = 255;
//             }
//         }
//     }
    
//     // Conversion et affichage via QImage
//     QImage qColor((const uchar*)coloredGradient.data, coloredGradient.cols, coloredGradient.rows,
//                   coloredGradient.step, QImage::Format_BGR888);
//     m_colorGradLabel->setPixmap(QPixmap::fromImage(qColor).scaled(m_colorGradLabel->size(),
//                                                                    Qt::KeepAspectRatio,
//                                                                    Qt::SmoothTransformation));
    
//     QImage qGlobal((const uchar*)globalThresholded.data, globalThresholded.cols, globalThresholded.rows,
//                    globalThresholded.step, QImage::Format_Grayscale8);
//     m_globalThreshLabel->setPixmap(QPixmap::fromImage(qGlobal).scaled(m_globalThreshLabel->size(),
//                                                                        Qt::KeepAspectRatio,
//                                                                        Qt::SmoothTransformation));
    
//     QImage qHyst((const uchar*)hysteresisThresholded.data, hysteresisThresholded.cols, hysteresisThresholded.rows,
//                  hysteresisThresholded.step, QImage::Format_Grayscale8);
//     m_hystThreshLabel->setPixmap(QPixmap::fromImage(qHyst).scaled(m_hystThreshLabel->size(),
//                                                                    Qt::KeepAspectRatio,
//                                                                    Qt::SmoothTransformation));
// }

#include "mywindow.h"
#include <QDebug>
#include <QImage>
#include <QPixmap>
#include <QTabWidget>
#include <cmath>

// ------------------
// Fonctions OpenCV utilitaires (similaires aux précédentes)
// ------------------

// Normalisation d’un noyau 3x3 (int) en float
static std::vector<std::vector<float>> normalizeKernel(const std::vector<std::vector<int>>& kernel)
{
    float sumAbs = 0.f;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            sumAbs += std::abs(kernel[i][j]);
        }
    }
    if(sumAbs == 0) sumAbs = 1.f;
    std::vector<std::vector<float>> normK(3, std::vector<float>(3, 0.f));
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            normK[i][j] = kernel[i][j] / sumAbs;
        }
    }
    return normK;
}

// Convolution 3x3
static cv::Mat applyConvolution(const cv::Mat& gray, const std::vector<std::vector<float>>& kernel)
{
    cv::Mat result = cv::Mat::zeros(gray.size(), CV_32F);
    for(int y = 1; y < gray.rows - 1; y++){
        for(int x = 1; x < gray.cols - 1; x++){
            float sum = 0.f;
            for(int j = -1; j <= 1; j++){
                for(int i = -1; i <= 1; i++){
                    float val = static_cast<float>(gray.at<uchar>(y + j, x + i));
                    sum += val * kernel[j + 1][i + 1];
                }
            }
            result.at<float>(y, x) = sum;
        }
    }
    return result;
}

// Combine plusieurs directions (somme des valeurs absolues)
static cv::Mat combineDirections(const std::vector<cv::Mat>& dirs)
{
    cv::Mat combined = cv::Mat::zeros(dirs[0].size(), CV_32F);
    for(const auto &m : dirs){
        combined += cv::abs(m);
    }
    return combined;
}

// Fonction de suppression des non-maxima pour l'affinage de contour
static cv::Mat nonMaximumSuppression(const cv::Mat& gradientMagnitude, const cv::Mat& Gx,
                                       const cv::Mat& Gy, const cv::Mat& G45, const cv::Mat& G135) {
    cv::Mat suppressed = gradientMagnitude.clone();
    int rows = gradientMagnitude.rows;
    int cols = gradientMagnitude.cols;
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            float gx = Gx.at<float>(i, j);
            float gy = Gy.at<float>(i, j);
            float g45 = G45.at<float>(i, j);
            float g135 = G135.at<float>(i, j);
            float maxVal = std::max({std::abs(gx), std::abs(gy), std::abs(g45), std::abs(g135)});
            if (std::abs(gx) == maxVal) {
                if (gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i, j - 1) ||
                    gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i, j + 1))
                    suppressed.at<float>(i, j) = 0;
            } else if (std::abs(gy) == maxVal) {
                if (gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i - 1, j) ||
                    gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i + 1, j))
                    suppressed.at<float>(i, j) = 0;
            } else if (std::abs(g45) == maxVal) {
                if (gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i - 1, j + 1) ||
                    gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i + 1, j - 1))
                    suppressed.at<float>(i, j) = 0;
            } else { // g135 dominant
                if (gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i - 1, j - 1) ||
                    gradientMagnitude.at<float>(i, j) < gradientMagnitude.at<float>(i + 1, j + 1))
                    suppressed.at<float>(i, j) = 0;
            }
        }
    }
    return suppressed;
}

// ------------------
// Constructeur
// ------------------
MyWindow::MyWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_inputImageLabel(new QLabel(this))
    , m_outputImageLabel(new QLabel(this))
    , m_filterCombo(new QComboBox(this))
    , m_thresholdSlider(new QSlider(Qt::Horizontal, this))
    , m_loadButton(new QPushButton("Charger Image", this))
    , m_applyBidirectionnelleButton(new QPushButton("Appliquer Détection Bidirectionnelle", this))
    , m_biMagLabel(new QLabel(this))
    , m_biDirLabel(new QLabel(this))
    , m_applyContoursButton(new QPushButton("Appliquer Contours par Seuillage", this))
    , m_colorGradLabel(new QLabel(this))
    , m_globalThreshLabel(new QLabel(this))
    , m_hystThreshLabel(new QLabel(this))
    , m_hystLowSlider(new QSlider(Qt::Horizontal, this))
    , m_hystHighSlider(new QSlider(Qt::Horizontal, this))
    , m_tabWidget(new QTabWidget(this))
    , m_tabFiltrage(new QWidget(this))
    , m_tabBidirectionnel(new QWidget(this))
    , m_tabContours(new QWidget(this))
    , m_currentFilter(0)
    , m_thresholdValue(50)
{
    // --- Onglet Filtrage (mode existant) ---
    m_thresholdSlider->setRange(0, 255);
    m_thresholdSlider->setValue(m_thresholdValue);
    m_filterCombo->addItem("Prewitt");
    m_filterCombo->addItem("Sobel");
    m_filterCombo->addItem("Kirsch");

    QHBoxLayout* controlLayout = new QHBoxLayout;
    controlLayout->addWidget(m_loadButton);
    controlLayout->addWidget(m_filterCombo);
    controlLayout->addWidget(m_thresholdSlider);

    QHBoxLayout* imagesLayout = new QHBoxLayout;
    imagesLayout->addWidget(m_inputImageLabel);
    imagesLayout->addWidget(m_outputImageLabel);

    QVBoxLayout* filtrageLayout = new QVBoxLayout;
    filtrageLayout->addLayout(controlLayout);
    filtrageLayout->addLayout(imagesLayout);
    m_tabFiltrage->setLayout(filtrageLayout);

    // --- Onglet Détection Bidirectionnelle ---
    QVBoxLayout* biLayout = new QVBoxLayout;
    biLayout->addWidget(m_applyBidirectionnelleButton);
    QHBoxLayout* biImagesLayout = new QHBoxLayout;
    biImagesLayout->addWidget(m_biMagLabel);
    biImagesLayout->addWidget(m_biDirLabel);
    biLayout->addLayout(biImagesLayout);
    m_tabBidirectionnel->setLayout(biLayout);

    // --- Onglet Contours par Seuillage ---
    QVBoxLayout* contoursLayout = new QVBoxLayout;
    contoursLayout->addWidget(m_applyContoursButton);
    
    // Ajout des sliders pour le seuillage par hystérésis
    m_hystLowSlider->setRange(0, 255);
    m_hystHighSlider->setRange(0, 255);
    m_hystLowSlider->setValue(50);
    m_hystHighSlider->setValue(200);
    QHBoxLayout* hystSliderLayout = new QHBoxLayout;
    QLabel* lowLabel = new QLabel("Hyst Low", this);
    QLabel* highLabel = new QLabel("Hyst High", this);
    hystSliderLayout->addWidget(lowLabel);
    hystSliderLayout->addWidget(m_hystLowSlider);
    hystSliderLayout->addWidget(highLabel);
    hystSliderLayout->addWidget(m_hystHighSlider);
    contoursLayout->addLayout(hystSliderLayout);

    QHBoxLayout* contoursImagesLayout = new QHBoxLayout;
    contoursImagesLayout->addWidget(m_colorGradLabel);
    contoursImagesLayout->addWidget(m_globalThreshLabel);
    contoursImagesLayout->addWidget(m_hystThreshLabel);
    contoursLayout->addLayout(contoursImagesLayout);

    // Nouveaux widgets pour l'affinage de contour (option additionnelle)
    m_applyContourRefinementButton = new QPushButton("Appliquer Affinage de Contour", this);
    m_refinedContourLabel = new QLabel(this);
    m_refinedContourLabel->setMinimumSize(200, 200); // Ajustez la taille si besoin
    contoursLayout->addWidget(m_applyContourRefinementButton);
    contoursLayout->addWidget(m_refinedContourLabel);

    m_tabContours->setLayout(contoursLayout);

    // --- Ajouter les onglets au QTabWidget ---
    m_tabWidget->addTab(m_tabFiltrage, "Filtrage");
    m_tabWidget->addTab(m_tabBidirectionnel, "Détection Bidirectionnelle");
    m_tabWidget->addTab(m_tabContours, "Contours par Seuillage");

    setCentralWidget(m_tabWidget);

    // Connecter les signaux/slots
    connect(m_loadButton, &QPushButton::clicked, this, &MyWindow::onLoadImage);
    connect(m_filterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MyWindow::onFilterTypeChanged);
    connect(m_thresholdSlider, &QSlider::valueChanged,
            this, &MyWindow::onThresholdChanged);
    connect(m_applyBidirectionnelleButton, &QPushButton::clicked,
            this, &MyWindow::onApplyBidirectionnelle);
    connect(m_applyContoursButton, &QPushButton::clicked,
            this, &MyWindow::onApplyContours);
    connect(m_hystLowSlider, &QSlider::valueChanged, this, &MyWindow::onHystThresholdChanged);
    connect(m_hystHighSlider, &QSlider::valueChanged, this, &MyWindow::onHystThresholdChanged);
    // Connexion du bouton d'affinage de contour
    connect(m_applyContourRefinementButton, &QPushButton::clicked,
            this, &MyWindow::onApplyContourRefinement);
}

// Destructeur
MyWindow::~MyWindow()
{
}

// ------------------
// Slots
// ------------------

void MyWindow::onLoadImage()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Ouvrir une image", "",
                                                    "Images (*.png *.jpg *.jpeg *.bmp)");
    if(fileName.isEmpty()){
        return;
    }
    m_originalMat = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
    if(m_originalMat.empty()){
        qWarning() << "Impossible de charger l'image :" << fileName;
        return;
    }
    QImage qimgIn((const uchar*)m_originalMat.data, m_originalMat.cols, m_originalMat.rows,
                  m_originalMat.step, QImage::Format_Grayscale8);
    m_inputImageLabel->setPixmap(QPixmap::fromImage(qimgIn).scaled(m_inputImageLabel->size(),
                                                                   Qt::KeepAspectRatio,
                                                                   Qt::SmoothTransformation));
    // Appliquer le traitement Filtrage (onglet 1)
    applyFilterAndThreshold();
}

void MyWindow::onFilterTypeChanged(int index)
{
    m_currentFilter = index;
    applyFilterAndThreshold();
}

void MyWindow::onThresholdChanged(int value)
{
    m_thresholdValue = value;
    applyFilterAndThreshold();
}

void MyWindow::onApplyBidirectionnelle()
{
    if(m_originalMat.empty()) return;
    applyBidirectionalDetection();
}

void MyWindow::onApplyContours()
{
    if(m_originalMat.empty()) return;
    applyContourDetection();
}

void MyWindow::onHystThresholdChanged(int /*value*/)
{
    if(m_originalMat.empty()) return;
    // Recalcule le seuillage par hystérésis avec les nouvelles valeurs de slider
    applyContourDetection();
}

// Nouveau slot pour l'affinage de contour (suppression des non-maxima)
void MyWindow::onApplyContourRefinement()
{
    if(m_originalMat.empty()) return;
    
    // Utilisation des mêmes filtres que pour le seuillage
    static std::vector<std::vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static std::vector<std::vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    static std::vector<std::vector<int>> kernel45 = {{0, 1, 2}, {-1, 0, 1}, {-2, -1, 0}};
    static std::vector<std::vector<int>> kernel135 = {{2, 1, 0}, {1, 0, -1}, {0, -1, -2}};
    
    cv::Mat Gx = applyConvolution(m_originalMat, normalizeKernel(sobelX));
    cv::Mat Gy = applyConvolution(m_originalMat, normalizeKernel(sobelY));
    cv::Mat G45 = applyConvolution(m_originalMat, normalizeKernel(kernel45));
    cv::Mat G135 = applyConvolution(m_originalMat, normalizeKernel(kernel135));
    
    cv::Mat gradientMagnitude;
    cv::magnitude(Gx, Gy, gradientMagnitude);
    gradientMagnitude = gradientMagnitude + cv::abs(G45) + cv::abs(G135);
    
    // Application de la suppression des non-maxima pour affiner les contours
    cv::Mat refined = nonMaximumSuppression(gradientMagnitude, Gx, Gy, G45, G135);
    
    // Normalisation pour affichage (conversion en 8 bits)
    cv::Mat refinedNormalized;
    cv::normalize(refined, refinedNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    QImage qRefined((const uchar*)refinedNormalized.data, refinedNormalized.cols, refinedNormalized.rows,
                    refinedNormalized.step, QImage::Format_Grayscale8);
    m_refinedContourLabel->setPixmap(QPixmap::fromImage(qRefined).scaled(m_refinedContourLabel->size(),
                                                                         Qt::KeepAspectRatio,
                                                                         Qt::SmoothTransformation));
}

// ------------------
// Traitements
// ------------------

// Mode Filtrage (basé sur l'interface initiale)
void MyWindow::applyFilterAndThreshold()
{
    if(m_originalMat.empty()) return;
    // Définition des noyaux selon le filtre choisi
    static std::vector<std::vector<int>> prewittX_int  = {{-1,0,1},{-1,0,1},{-1,0,1}};
    static std::vector<std::vector<int>> prewittY_int  = {{-1,-1,-1},{0,0,0},{1,1,1}};
    static std::vector<std::vector<int>> prewitt45_int = {{0,1,1},{-1,0,1},{-1,-1,0}};
    static std::vector<std::vector<int>> prewitt135_int= {{1,1,0},{1,0,-1},{0,-1,-1}};
   
    static std::vector<std::vector<int>> sobelX_int    = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static std::vector<std::vector<int>> sobelY_int    = {{-1,-2,-1},{0,0,0},{1,2,1}};
    static std::vector<std::vector<int>> sobel45_int   = {{0,1,2},{-1,0,1},{-2,-1,0}};
    static std::vector<std::vector<int>> sobel135_int  = {{2,1,0},{1,0,-1},{0,-1,-2}};

    static std::vector<std::vector<int>> kirschX_int   = {{-3,-3,5},{-3,0,5},{-3,-3,5}};
    static std::vector<std::vector<int>> kirschY_int   = {{-3,-3,-3},{5,0,-3},{5,5,-3}};
    static std::vector<std::vector<int>> kirsch45_int  = {{-3,5,5},{-3,0,5},{-3,-3,-3}};
    static std::vector<std::vector<int>> kirsch135_int = {{5,5,-3},{5,0,-3},{-3,-3,-3}};

    // Sélection des noyaux
    std::vector<std::vector<int>> kx, ky, k45, k135;
    if(m_currentFilter == 0){ // Prewitt
        kx   = prewittX_int;  ky   = prewittY_int;
        k45  = prewitt45_int; k135 = prewitt135_int;
    }
    else if(m_currentFilter == 1){ // Sobel
        kx   = sobelX_int;  ky   = sobelY_int;
        k45  = sobel45_int; k135 = sobel135_int;
    }
    else { // Kirsch
        kx   = kirschX_int;   ky   = kirschY_int;
        k45  = kirsch45_int;  k135 = kirsch135_int;
    }
    // Normaliser et appliquer les convolutions
    auto nkx   = normalizeKernel(kx);
    auto nky   = normalizeKernel(ky);
    auto nk45  = normalizeKernel(k45);
    auto nk135 = normalizeKernel(k135);
    cv::Mat gx   = applyConvolution(m_originalMat, nkx);
    cv::Mat gy   = applyConvolution(m_originalMat, nky);
    cv::Mat g45  = applyConvolution(m_originalMat, nk45);
    cv::Mat g135 = applyConvolution(m_originalMat, nk135);
    cv::Mat combined = combineDirections({gx, gy, g45, g135});
    // Seuillage binaire
    cv::Mat thresholded = cv::Mat::zeros(combined.size(), CV_8U);
    for(int y = 0; y < combined.rows; y++){
        for(int x = 0; x < combined.cols; x++){
            float val = combined.at<float>(y, x);
            thresholded.at<uchar>(y, x) = (val >= m_thresholdValue) ? 255 : 0;
        }
    }
    m_resultMat = thresholded;
    QImage qimgOut((const uchar*)m_resultMat.data, m_resultMat.cols,
                   m_resultMat.rows, m_resultMat.step,
                   QImage::Format_Grayscale8);
    m_outputImageLabel->setPixmap(QPixmap::fromImage(qimgOut).scaled(m_outputImageLabel->size(),
                                                                     Qt::KeepAspectRatio,
                                                                     Qt::SmoothTransformation));
}

// Mode Détection Bidirectionnelle
void MyWindow::applyBidirectionalDetection()
{
    // Utiliser les filtres Prewitt pour X et Y
    static std::vector<std::vector<int>> prewittX_int = {{-1,0,1},{-1,0,1},{-1,0,1}};
    static std::vector<std::vector<int>> prewittY_int = {{-1,-1,-1},{0,0,0},{1,1,1}};
    auto prewittX = normalizeKernel(prewittX_int);
    auto prewittY = normalizeKernel(prewittY_int);
    cv::Mat Gx = applyConvolution(m_originalMat, prewittX);
    cv::Mat Gy = applyConvolution(m_originalMat, prewittY);
    // Calcul de la magnitude et de la direction
    cv::Mat magnitude(m_originalMat.size(), CV_32F, cv::Scalar(0));
    cv::Mat direction(m_originalMat.size(), CV_32F, cv::Scalar(0));
    for (int i = 0; i < m_originalMat.rows; i++){
        for (int j = 0; j < m_originalMat.cols; j++){
            float gx = Gx.at<float>(i, j);
            float gy = Gy.at<float>(i, j);
            magnitude.at<float>(i, j) = std::sqrt(gx * gx + gy * gy);
            direction.at<float>(i, j) = std::atan2(gy, gx) * 180.0f / CV_PI;
        }
    }
    // Normaliser la magnitude pour affichage
    cv::Mat magnitudeNorm;
    cv::normalize(magnitude, magnitudeNorm, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Création d'une image pour visualiser la direction (conversion en HSV puis BGR)
    cv::Mat directionVis(m_originalMat.size(), CV_8UC3, cv::Scalar(0,0,0));
    for (int i = 0; i < m_originalMat.rows; i++){
        for (int j = 0; j < m_originalMat.cols; j++){
            float dir = direction.at<float>(i, j);
            // Normaliser de -180..+180 vers 0..180
            uchar hue = static_cast<uchar>((dir + 180.f) / 2.f);
            directionVis.at<cv::Vec3b>(i, j) = cv::Vec3b(hue, 255, 255);
        }
    }
    cv::cvtColor(directionVis, directionVis, cv::COLOR_HSV2BGR);

    // Convertir pour affichage dans Qt
    QImage qMag((const uchar*)magnitudeNorm.data, magnitudeNorm.cols, magnitudeNorm.rows,
                magnitudeNorm.step, QImage::Format_Grayscale8);
    m_biMagLabel->setPixmap(QPixmap::fromImage(qMag).scaled(m_biMagLabel->size(),
                                                             Qt::KeepAspectRatio,
                                                             Qt::SmoothTransformation));
    QImage qDir((const uchar*)directionVis.data, directionVis.cols, directionVis.rows,
                directionVis.step, QImage::Format_BGR888);
    m_biDirLabel->setPixmap(QPixmap::fromImage(qDir).scaled(m_biDirLabel->size(),
                                                             Qt::KeepAspectRatio,
                                                             Qt::SmoothTransformation));
}

// Mode Contours par Seuillage
void MyWindow::applyContourDetection()
{
    // Correction des filtres Sobel
    static std::vector<std::vector<int>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static std::vector<std::vector<int>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    static std::vector<std::vector<int>> kernel45 = {{0, 1, 2}, {-1, 0, 1}, {-2, -1, 0}};
    static std::vector<std::vector<int>> kernel135 = {{2, 1, 0}, {1, 0, -1}, {0, -1, -2}};
    
    // Convolution pour chaque direction
    cv::Mat Gx = applyConvolution(m_originalMat, normalizeKernel(sobelX));
    cv::Mat Gy = applyConvolution(m_originalMat, normalizeKernel(sobelY));
    cv::Mat G45 = applyConvolution(m_originalMat, normalizeKernel(kernel45));
    cv::Mat G135 = applyConvolution(m_originalMat, normalizeKernel(kernel135));
    
    // Calcul du gradient combiné
    cv::Mat gradientMagnitude;
    cv::magnitude(Gx, Gy, gradientMagnitude);
    gradientMagnitude = gradientMagnitude + cv::abs(G45) + cv::abs(G135);
    
    // Création d'une image colorée (canaux : X -> rouge, Y -> vert, diagonales -> bleu)
    std::vector<cv::Mat> channels(3);
    cv::normalize(cv::abs(Gx), channels[0], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(cv::abs(Gy), channels[1], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(cv::abs(G45 + G135), channels[2], 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat coloredGradient;
    cv::merge(channels, coloredGradient);
    
    // Seuillage global
    double globalThresh = cv::mean(gradientMagnitude)[0];
    cv::Mat globalThresholded;
    cv::threshold(gradientMagnitude, globalThresholded, globalThresh, 255, cv::THRESH_BINARY);
    // Conversion en CV_8U pour un affichage correct
    globalThresholded.convertTo(globalThresholded, CV_8U);
    
    // Seuillage par hystérésis avec valeurs issues des sliders
    double lowThresh = m_hystLowSlider->value();
    double highThresh = m_hystHighSlider->value();
    cv::Mat hysteresisThresholded = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);
    for (int i = 1; i < gradientMagnitude.rows - 1; i++){
        for (int j = 1; j < gradientMagnitude.cols - 1; j++){
            float val = gradientMagnitude.at<float>(i, j);
            if (val >= highThresh){
                hysteresisThresholded.at<uchar>(i, j) = 255;
            } else if (val >= lowThresh){
                bool connected = false;
                for (int x = -1; x <= 1; x++){
                    for (int y = -1; y <= 1; y++){
                        if (gradientMagnitude.at<float>(i + x, j + y) >= highThresh)
                            connected = true;
                    }
                }
                if (connected)
                    hysteresisThresholded.at<uchar>(i, j) = 255;
            }
        }
    }
    
    // Conversion et affichage via QImage
    QImage qColor((const uchar*)coloredGradient.data, coloredGradient.cols, coloredGradient.rows,
                  coloredGradient.step, QImage::Format_BGR888);
    m_colorGradLabel->setPixmap(QPixmap::fromImage(qColor).scaled(m_colorGradLabel->size(),
                                                                   Qt::KeepAspectRatio,
                                                                   Qt::SmoothTransformation));
    
    QImage qGlobal((const uchar*)globalThresholded.data, globalThresholded.cols, globalThresholded.rows,
                   globalThresholded.step, QImage::Format_Grayscale8);
    m_globalThreshLabel->setPixmap(QPixmap::fromImage(qGlobal).scaled(m_globalThreshLabel->size(),
                                                                       Qt::KeepAspectRatio,
                                                                       Qt::SmoothTransformation));
    
    QImage qHyst((const uchar*)hysteresisThresholded.data, hysteresisThresholded.cols, hysteresisThresholded.rows,
                 hysteresisThresholded.step, QImage::Format_Grayscale8);
    m_hystThreshLabel->setPixmap(QPixmap::fromImage(qHyst).scaled(m_hystThreshLabel->size(),
                                                                   Qt::KeepAspectRatio,
                                                                   Qt::SmoothTransformation));
}
