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

private slots:
    void calculate();

private:
    Ui::MainWindow *ui;
    QLineEdit *num1Input; // 输入框：第一个数
    QLineEdit *num2Input; // 输入框：第二个数
    QPushButton *calcButton; // 计算按钮
    QLineEdit *gcdResult; // 显示GCD结果
    QLineEdit *inverseResult; // 显示模乘逆元
    QLineEdit *sResult; // 显示贝祖等式系数s
    QLineEdit *tResult; // 显示贝祖等式系数t

    struct GCDResult
    {
        int64_t gcd;
        int64_t x;
        int64_t y;
    };
    GCDResult extendedGCD(int64_t a, int64_t b);
    int64_t modInverse(int64_t a,int64_t m);
    int64_t gcd(int64_t a, int64_t b);
};
#endif // MAINWINDOW_H
