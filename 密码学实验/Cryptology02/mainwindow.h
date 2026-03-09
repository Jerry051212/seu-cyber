#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>


QT_BEGIN_NAMESPACE
namespace Ui{
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
    void encryptText();
    void decryptText();

private:
    Ui::MainWindow *ui;

    QLineEdit *plainTextEdit;
    QLineEdit *cipherTextEdit;
    QLineEdit *keyEdit;
    QTextEdit *resultEdit;
    QPushButton *encryptButton;
    QPushButton *decryptButton;

    // 加密辅助函数
    QByteArray padTo128bit(const QByteArray& input);
    QByteArray encryptCBC(const QByteArray& plaintext, const QByteArray& key, const QByteArray& iv);
    QByteArray decryptCBC(const QByteArray& ciphertext, const QByteArray& key, const QByteArray& iv);
    QByteArray aesEncrypt(const QByteArray& input, const QByteArray& key);
    QByteArray aesDecrypt(const QByteArray& input, const QByteArray& key);
    QByteArray xorBytes(const QByteArray& a, const QByteArray& b);
    void subBytes(QByteArray& state, bool inverse = false); // 字节代换
    void shiftRows(QByteArray& state, bool inverse = false); // 行移位
    void mixColumns(QByteArray& state, bool inverse = false); // 列混合
    void addRoundKey(QByteArray& state, const QByteArray& roundKey); // 密钥加
    QVector<QByteArray> keyExpansion(const QByteArray& key); // 密钥扩展
};

#endif // MAINWINDOW_H
