 #include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QLabel>
#include <QGridLayout>
#include <QMessageBox>
#include <random>
#include <QRegularExpression>
#include <cmath>
#include <iomanip>
#include <sstream>
using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QGridLayout *layout = new QGridLayout(centralWidget);

    // 添加标签与输入框
    layout->addWidget(new QLabel("Prime p:"), 0, 0);
    pInput = new QLineEdit();
    layout->addWidget(pInput, 0, 1);

    layout->addWidget(new QLabel("Private key a:"), 1, 0);
    aInput = new QLineEdit();
    layout->addWidget(aInput, 1, 1);

    layout->addWidget(new QLabel("Plaintext/Ciphertext:"), 2, 0);
    plainInput = new QLineEdit();
    layout->addWidget(plainInput, 2, 1);

    encryptBtn = new QPushButton("Encrypt");
    decryptBtn = new QPushButton("Decrypt");
    layout->addWidget(encryptBtn, 3, 0);
    layout->addWidget(decryptBtn, 3, 1);

    output = new QTextEdit();
    output->setReadOnly(true);
    layout->addWidget(new QLabel("Result:"), 4, 0);
    layout->addWidget(output, 4, 1, 2, 1);

    connect(encryptBtn, &QPushButton::clicked, this, &MainWindow::onEncryptClicked);
    connect(decryptBtn, &QPushButton::clicked, this, &MainWindow::onDecryptClicked);

    setWindowTitle("ElGamal Encryption/Decryption");
    resize(500, 400);
}

MainWindow::~MainWindow()
{
    delete ui;
}

long long MainWindow::modPow(long long base, long long exp, long long mod)
{ // 模幂运算
    long long result = 1;
    base = (base % mod + mod) % mod; // 确保base小于mod
    while (exp > 0)
    {
        if (exp & 1) // 指数最后一位为1
            result = (result * base) % mod; // 乘base再取模
        base = (base * base) % mod; // base平方取模
        exp >>= 1; // 指数除以2
    }
    return result;
}

long long MainWindow::modInverse(long long a, long long m)
{ // 求模逆元
    long long m0 = m, t, q;
    long long x0 = 0, x1 = 1;
    if (m == 1)
        return 0;
    while (a > 1)
    {
        q = a / m;
        t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < 0)
        x1 += m0;
    return x1;
}

bool MainWindow::isPrime(long long n)
{ // 判断是否为素数
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    for (long long i = 5; i * i <= n; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

long long MainWindow::findGenerator(long long p)
{ // 寻找生成元
    for (long long g = 2; g < p; g++)
    {
        bool isGen = true;
        for (long long i = 1; i < p - 1; i++)
        {
            if (modPow(g, i, p) == 1)
            { // 若g^i = 1，则不是生成元
                isGen = false;
                break;
            }
        }
        if (isGen)
            return g;
    }
    return -1;
}

long long MainWindow::randomK(long long p)
{ // 生成随机数，1 <=k<= p-2
    // 随机数生成器random_device 和 mt19937
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<long long> dis(1, p - 2);
    return dis(gen);
}

vector<unsigned char> MainWindow::stringToBytes(const QString &str)
{
    QByteArray ba = str.toUtf8(); // 使用toUtf8将 QString 转为字节数组
    return vector<unsigned char>(ba.begin(), ba.end()); // 将ba字节数组转为char向量返回
}

QString MainWindow::bytesToString(const vector<unsigned char> &bytes)
{
    QByteArray ba(reinterpret_cast<const char*>(bytes.data()), bytes.size()); // 将char向量转为字节数组
    return QString::fromUtf8(ba); // 转回QString
}

vector<long long> MainWindow::encodeBlock(const vector<unsigned char> &bytes, size_t blockSize, long long p, size_t &paddingLen)
 { // 块编码函数
    vector<unsigned char> paddedBytes = bytes; // 复制字节向量
    // PKCS#7 填充，确保数据长度为blockSize整数倍
    paddingLen = blockSize - (bytes.size() % blockSize); // 计算需要填充的字节数
    if (paddingLen == 0)
        paddingLen = blockSize; // 若刚好整除，填充一个完整块
    for (size_t i = 0; i < paddingLen; i++)
    {
        paddedBytes.push_back(static_cast<unsigned char>(paddingLen)); // 填充值为填充长度
    }

    vector<long long> blocks; // 存储编码后的块
    if (blockSize == 1)
    {// 若块大小为1，逐字节处理
        for (size_t i = 0; i < paddedBytes.size(); i++)
        {
            long long byte = paddedBytes[i]; // 取单个字节
            if (byte >= p)
            { // 确保字节值小于p
                QMessageBox::warning(this, "Error", "Byte value too large for p.");
                return {};
            }
            blocks.push_back(byte); // 将字节作为块加入
        }
    }

    else
    {
        for (size_t i = 0; i < paddedBytes.size(); i += blockSize)
        { // 块大小大于1，多个字节合并为一个块
            long long block = 0;
            for (size_t j = 0; j < blockSize && i + j < paddedBytes.size(); j++)
            { // 每次处理blockSize个字节
                block = (block << 8) + paddedBytes[i + j]; // 将多个字节合并为一个块
            }

            if (block >= p)
            {// 回退到逐字节处理
                blocks.clear();
                for (size_t j = 0; j < paddedBytes.size(); j++)
                {
                    long long byte = paddedBytes[j];
                    if (byte >= p)
                    {
                        QMessageBox::warning(this, "Error", "Byte value too large for p.");
                        return {};
                    }
                    blocks.push_back(byte);
                }
                break;
            }
            blocks.push_back(block);
        }
    }
    return blocks;
}

vector<unsigned char> MainWindow::decodeBlock(const vector<long long> &blocks, size_t blockSize, size_t originalSize, size_t paddingLen)
{ // 将解密后的数字快转回字节向量
    vector<unsigned char> bytes;
    if (blockSize == 1)
    {
        for (long long block : blocks)
        {
            bytes.push_back(static_cast<unsigned char>(block & 0xFF)); // 直接取最低字节
        }
    }

    else
    {
        for (long long block : blocks)
        {
            vector<unsigned char> blockBytes;
            for (size_t i = 0; i < blockSize; i++)
            {
                blockBytes.push_back((block >> (8 * (blockSize - 1 - i))) & 0xFF); // 拆分块为字节
            }
            bytes.insert(bytes.end(), blockBytes.begin(), blockBytes.end());
        }
    }

    // 移除 PKCS#7 填充
    if (paddingLen > 0 && bytes.size() >= paddingLen)
    {
        bool validPadding = true;
        for (size_t i = 1; i <= paddingLen; i++)
        {
            if (bytes[bytes.size() - i] != paddingLen)
            {
                validPadding = false;
                break;
            }
        }

        if (validPadding)
        {
            bytes.resize(bytes.size() - paddingLen); // 移除填充结果
        }
    }
    // 截断到原始长度
    if (bytes.size() > originalSize)
    {
        bytes.resize(originalSize);
    }
    return bytes;
}

vector<pair<long long, long long>> MainWindow::encryptCBC(const vector<long long> &blocks, long long ivBlock, long long p, long long alpha, long long beta)
{ // 输入明文块blocks，初始向量ivblock，素数p，生成元alpha，公钥beta
    vector<pair<long long, long long>> cipher;
    long long prev = ivBlock; // 初始化为前一块
    for (long long m : blocks)
    {
        long long k = randomK(p); // 随机生成k
        long long gamma = modPow(alpha, k, p); // gamma = alpha^k mod p
        long long delta = ((m ^ prev) * modPow(beta, k, p)) % p; // delta = (m XOR prev) * beta^k mod p
        cipher.emplace_back(gamma, delta);
        prev = delta; // 更新前一块为当前delta
    }
    return cipher; // 返回加密后的（gamma,delta）对
}

vector<long long> MainWindow::decryptCBC(const vector<pair<long long, long long>> &cipher, long long ivBlock, long long p, long long a)
{
    vector<long long> blocks;
    long long prev = ivBlock; //初始化prev为IV
    for (const auto &pair : cipher)
    {
        long long gamma = pair.first;
        long long delta = pair.second;
        long long gamma_a = modPow(gamma, a, p); // gamma_a = gamma^a mod p
        long long inv_gamma_a = modInverse(gamma_a, p); // inv_gamma_a = (gamma^a)^(-1) mod p
        long long m = (delta * inv_gamma_a) % p; // m = delta * (gamma^a)^(-1) mod p
        m = m ^ prev;
        blocks.push_back(m);
        prev = delta; // 更新prev
    }
    return blocks;
}

void MainWindow::onEncryptClicked()
{
    bool ok;
    long long p = pInput->text().toLongLong(&ok);
    if (!ok || !isPrime(p))
    {
        QMessageBox::warning(this, "Error", "Please enter a valid prime number for p.");
        return;
    }

    long long a = aInput->text().toLongLong(&ok);
    if (!ok || a <= 0 || a >= p)
    {
        QMessageBox::warning(this, "Error", "Please enter a valid private key a (0 < a < p).");
        return;
    }

    QString plainText = plainInput->text();
    if (plainText.isEmpty())
    {
        QMessageBox::warning(this, "Error", "Please enter plaintext.");
        return;
    }

    long long alpha = findGenerator(p);
    if (alpha == -1)
    {
        QMessageBox::warning(this, "Error", "Cannot find a generator for p.");
        return;
    }

    long long beta = modPow(alpha, a, p); // 公钥beta

    vector<unsigned char> bytes = stringToBytes(plainText);
    size_t originalSize = bytes.size(); // 记录原始字节数
    size_t blockSize = max<size_t>(1, static_cast<size_t>(floor(log2(p) / 8))); // 确定blockSize,作为参数传给encodeBlock
    size_t paddingLen = 0;

    vector<long long> blocks = encodeBlock(bytes, blockSize, p, paddingLen);
    if (blocks.empty())
    {
        QMessageBox::warning(this, "Error", "Failed to encode plaintext.");
        return;
    }

    // 生成随机iv
    vector<unsigned char> iv(blockSize);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<unsigned char> dis(0, 255);
    for (size_t i = 0; i < blockSize; i++)
    {
        iv[i] = dis(gen);
    }
    long long ivBlock = 0;

    for (size_t i = 0; i < blockSize; i++)
    {
        ivBlock = (ivBlock << 8) + iv[i];
    }

    if (ivBlock >= p)
    {
        QMessageBox::warning(this, "Error", "IV too large for p.");
        return;
    }

    // CBC加密生成密文
    vector<pair<long long, long long>> cipher = encryptCBC(blocks, ivBlock, p, alpha, beta);

    // 格式化输出
    stringstream ss;
    ss << hex << setfill('0') << setw(16) << ivBlock;
    ss << ";" << hex << setfill('0') << setw(8) << originalSize;
    ss << ";" << hex << setfill('0') << setw(8) << paddingLen;
    for (const auto &pair : cipher)
    {
        ss << ";" << hex << setfill('0') << setw(16) << pair.first
           << "," << hex << setfill('0') << setw(16) << pair.second;
    }
    QString cipherText = QString::fromStdString(ss.str());

    output->setText(QString("Public key (p, alpha, beta): (%1, %2, %3)\nCiphertext: %4")
                        .arg(p).arg(alpha).arg(beta).arg(cipherText));
}

void MainWindow::onDecryptClicked()
{
    bool ok;
    long long p = pInput->text().toLongLong(&ok);
    if (!ok || !isPrime(p))
    {
        QMessageBox::warning(this, "Error", "Please enter a valid prime number for p.");
        return;
    }

    long long a = aInput->text().toLongLong(&ok);
    if (!ok || a <= 0 || a >= p)
    {
        QMessageBox::warning(this, "Error", "Please enter a valid private key a (0 < a < p).");
        return;
    }

    // 解析密文字符串
    QString cipherText = plainInput->text();
    QStringList parts = cipherText.split(";", Qt::SkipEmptyParts); // 按分号分割密文
    if (parts.size() < 3)
    { // 每一部分至少需要IV，originalSize，paddingLen三个部分
        QMessageBox::warning(this, "Error", "Please enter valid ciphertext (format: IV;originalSize;paddingLen;gamma,delta;...).");
        return;
    }

    // 从密文中提取ivBlock，originalSize，填充长度paddingLen和公钥(gamma,delta)
    bool ivOk, origOk, padOk;
    long long ivBlock = parts[0].toLongLong(&ivOk, 16); // 十六进制解析IV
    size_t originalSize = parts[1].toULongLong(&origOk, 16); // 解析originalSize
    size_t paddingLen = parts[2].toULongLong(&padOk, 16); // 解析paddingLen
    if (!ivOk || !origOk || !padOk || ivBlock >= p)
    { // 验证解析结果和IV是否正确
        QMessageBox::warning(this, "Error", "Invalid IV, originalSize, or paddingLen format.");
        return;
    }


    vector<pair<long long, long long>> cipher; // 用于存储(gamma,delta)对
    for (int i = 3; i < parts.size(); i++)
    { // 第四个部分开始为gamma,delta
        QStringList pair = parts[i].split(",");
        if (pair.size() != 2)
        {
            QMessageBox::warning(this, "Error", QString("Invalid ciphertext pair at position %1.").arg(i - 2));
            return;
        }

        bool gammaOk, deltaOk;
        long long gamma = pair[0].toLongLong(&gammaOk, 16); // 解析gamma
        long long delta = pair[1].toLongLong(&deltaOk, 16); // 解析delta
        if (!gammaOk || !deltaOk || gamma <= 0 || gamma >= p || delta <= 0 || delta >= p)
        {
            QMessageBox::warning(this, "Error", QString("Invalid ciphertext pair at position %1.").arg(i - 2));
            return;
        }
        cipher.emplace_back(gamma, delta); // 加入密文列表
    }

    size_t blockSize = max<size_t>(1, static_cast<size_t>(floor(log2(p) / 8))); // 计算块大小

    vector<long long> blocks = decryptCBC(cipher, ivBlock, p, a); // CBC解密
    vector<unsigned char> bytes = decodeBlock(blocks, blockSize, originalSize, paddingLen); // 将块解码为字节
    QString plainText = bytesToString(bytes);

    output->setText(QString("Decrypted plaintext: %1").arg(plainText)); // 显示解密结果
}
