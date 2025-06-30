#ifndef MYWINDOW_H
#define MYWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QSlider>
#include <QComboBox>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <QFileDialog>
#include <QImage>
#include <QPixmap>
#include <opencv2/opencv.hpp>

// Notre classe principale
class MyWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MyWindow(QWidget* parent = nullptr);
    ~MyWindow();
    

private slots:
    void onLoadImage();             // Charger l’image
    void onFilterTypeChanged(int);  // Changement du type de filtre (Filtrage onglet 1)
    void onThresholdChanged(int);   // Changement du seuil (Filtrage onglet 1)
    void onApplyBidirectionnelle(); // Appliquer le mode détection bidirectionnelle (onglet 2)
    void onApplyContours();         // Appliquer le mode contours par seuillage (onglet 3)
    void onHystThresholdChanged(int); // Mise à jour des seuils d'hystérésis
    void onApplyContourRefinement();


private:
    // Méthodes internes de traitement
    void applyFilterAndThreshold(); // Mode Filtrage (onglet 1)
    void applyBidirectionalDetection();
    void applyContourDetection();
    

    // Widgets pour onglet Filtrage (mode existant)
    QLabel*     m_inputImageLabel;   // Affiche l’image originale
    QLabel*     m_outputImageLabel;  // Affiche le résultat filtré+seuillé
    QComboBox*  m_filterCombo;
    QSlider*    m_thresholdSlider;
    QPushButton* m_loadButton;

    // Widgets pour onglet Détection bidirectionnelle
    QPushButton* m_applyBidirectionnelleButton;
    QLabel*      m_biMagLabel;  // Affiche la magnitude
    QLabel*      m_biDirLabel;  // Affiche la direction (colorisée)

    // Widgets pour onglet Contours par seuillage
    QPushButton* m_applyContoursButton;
    QLabel*      m_colorGradLabel;   // Gradient coloré
    QLabel*      m_globalThreshLabel; // Seuillage global
    QLabel*      m_hystThreshLabel;   // Seuillage par hystérésis
    QSlider*     m_hystLowSlider;     // Slider pour le seuil bas d'hystérésis
    QSlider*     m_hystHighSlider;    // Slider pour le seuil haut d'hystérésis

    // Onglets
    QTabWidget* m_tabWidget;
    QWidget*    m_tabFiltrage;
    QWidget*    m_tabBidirectionnel;
    QWidget*    m_tabContours;

    QPushButton* m_applyContourRefinementButton;
    QLabel*      m_refinedContourLabel;

    // Données
    cv::Mat     m_originalMat;    // Image d’entrée (OpenCV)
    cv::Mat     m_resultMat;      // Image de sortie (pour mode Filtrage)
    int         m_currentFilter;  // 0=Prewitt, 1=Sobel, 2=Kirsch
    int         m_thresholdValue; // Valeur de seuil
};

#endif // MYWINDOW_H
