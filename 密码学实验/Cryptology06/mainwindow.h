#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    QLineEdit *lineEditAlpha;
    QLineEdit *lineEditP;
    QLineEdit *lineEditX;
    QLineEdit *lineEditY;
    QPushButton *calculateButton;
    QLabel *resultLabel;

    long long modPow(long long base, long long exp, long long modulus);
    void onCalculateClicked();
    bool isPrime(long long n);
    bool isPrimitiveRoot(long long alpha, long long p);
};
#endif // MAINWINDOW_H


