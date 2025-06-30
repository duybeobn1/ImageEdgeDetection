#include <QApplication>
#include "mywindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MyWindow window;
    window.resize(1200, 600); // Taille approximative
    window.show();

    return app.exec();
}
