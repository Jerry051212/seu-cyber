#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QTextEdit>
#include <QPushButton>
#include <vector>
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui
{
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
    void on_encryptButton_clicked(); // 加密按钮槽函数
    void  on_decryptButton_clicked(); //解密按钮槽函数

private:
    Ui::MainWindow *ui;


    // DES算法函数
    void generateSubKeys(const vector<bool> &key,vector<vector<bool>> &subKeys);
    vector<bool> desEncryptBlock(const vector<bool> &plainBlock, const vector<vector<bool>> &subKeys);
    vector<bool> desDecryptBlock(const vector<bool> &cipherBlock, const vector<vector<bool>> &subKeys);
    vector<bool> feistel(const vector<bool> &right, const vector<bool> &subKey);

    // CBC模式函数
    QString desEncryptCBC(const QString &plainText); //DES加密
    QString desDecryptCBC(const QString &cipherText); // DES解密
    vector<vector<bool>> padData(const vector<bool> &data);

    // 其他
    vector<bool> stringToBits(const QString &str);
    QString bitsToString(const vector<bool> &bits);
    vector<bool> hexToBits(const QString &hex);
    QString bitsToHex(const vector<bool> &bits);
    vector<bool> xorBits(const vector<bool> &a, const vector<bool> &b);
    vector<bool> getUserKey();
    vector<bool> iv = vector<bool>(64,0);
};
#endif // MAINWINDOW_H
