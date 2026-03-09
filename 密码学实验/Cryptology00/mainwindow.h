#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTableWidget>
#include <QTextEdit>
#include <QString>
#include <map>
#include <vector>
#include <tuple>
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onDecryptButtonClicked(); // 解密按钮槽函数

private:
    Ui::MainWindow *ui;
    void getFrequency(const QString &ciphertext, vector<tuple<char, int, double>> &result);
    QString strDecrypt(const QString &ciphertext, const map<char, char> &mapping);
    void updateTable(const vector<tuple<char, int, double>> &freq);
    void updateText(const QString &plaintext);
};

#endif // MAINWINDOW_H
