#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QLineEdit>
#include <QTextEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <vector>
#include <string>
using namespace std;

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
    void onEncryptClicked();
    void onDecryptClicked();

private:
    Ui::MainWindow *ui;
    // UI 控件
    QLineEdit *pInput;      // 大素数p
    QLineEdit *aInput;      // 私钥a
    QLineEdit *plainInput;  // 明文或密文
    QTextEdit *output;      // 输出结果
    QPushButton *encryptBtn;
    QPushButton *decryptBtn;

    // ElGamal算法函数
    long long modPow(long long base, long long exp, long long mod); // 模幂运算
    long long modInverse(long long a, long long m); // 求模逆元
    bool isPrime(long long n); // 判断是否为素数
    long long findGenerator(long long p); // 寻找生成元
    long long randomK(long long p); // 生成随机k

    // 字符串与字节转化
    vector<unsigned char> stringToBytes(const QString &str);
    QString bytesToString(const vector<unsigned char> &bytes);

    // 块编码与解码
    vector<long long> encodeBlock(const vector<unsigned char> &bytes, size_t blockSize, long long p, size_t &paddingLen);
    vector<unsigned char> decodeBlock(const vector<long long> &blocks, size_t blockSize, size_t originalSize, size_t paddingLen);

    // CBC加解密
    vector<pair<long long, long long>> encryptCBC(const vector<long long> &blocks, long long ivBlock, long long p, long long alpha, long long beta);
    vector<long long> decryptCBC(const vector<pair<long long, long long>> &cipher, long long ivBlock, long long p, long long a);
};

#endif // MAINWINDOW_H
