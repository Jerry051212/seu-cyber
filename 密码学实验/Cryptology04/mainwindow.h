#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QVector>

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
    void generateKeys();      // 生成公钥和私钥
    void encryptMessage();    // 加密消息
    void decryptMessage();    // 解密消息

private:
    Ui::MainWindow *ui;
    // UI 控件
    QLineEdit *pInput;        // 输入 p 的文本框
    QLineEdit *qInput;        // 输入 q 的文本框
    QLineEdit *eInput;        // 输入 e 的文本框
    QLineEdit *messageInput;  // 输入消息的文本框
    QLineEdit *resultOutput;  // 显示结果的文本框
    QLabel *publicKeyLabel;   // 显示公钥的标签
    QLabel *privateKeyLabel;  // 显示私钥的标签
    QPushButton *generateButton;  // 生成密钥按钮
    QPushButton *encryptButton;   // 加密按钮
    QPushButton *decryptButton;   // 解密按钮

    // RSA 相关变量
    int64_t p, q, n, e, d, phi;   // RSA 参数：p, q, n, e, d 和欧拉函数值 phi
    QByteArray iv = QByteArray::fromHex("1234567890ABCDEF");  // 初始化向量 IV

    // RSA 算法函数
    int64_t gcd(int64_t a, int64_t b);         // 计算最大公约数
    int64_t modPow(int64_t base, int64_t exp, int64_t modulus);  // 模幂运算
    int64_t modInverse(int64_t e, int64_t phi);  // 计算模逆
    bool isPrime(int64_t n);                    // 判断是否为素数

    // 分块和 CBC 相关函数
    QVector<int64_t> stringToBlocks(const QString &text);  // 将字符串转为数字块
    QString blocksToString(const QVector<int64_t> &blocks); // 将数字块转为字符串
    QVector<int64_t> encryptCBC(const QVector<int64_t> &plainBlocks); // CBC 模式加密
    QVector<int64_t> decryptCBC(const QVector<int64_t> &cipherBlocks); // CBC 模式解密
    int64_t blockSize() const;  // 计算块大小
};

#endif // MAINWINDOW_H
