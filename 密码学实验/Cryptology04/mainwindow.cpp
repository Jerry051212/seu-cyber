#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>
#include <QGridLayout>
#include <QMessageBox>
#include <QRandomGenerator>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    QGridLayout *gridLayout = new QGridLayout();

    pInput = new QLineEdit(this);
    qInput = new QLineEdit(this);
    eInput = new QLineEdit(this);
    messageInput = new QLineEdit(this);
    resultOutput = new QLineEdit(this);
    resultOutput->setReadOnly(true);

    publicKeyLabel = new QLabel("公钥: 未生成", this);
    privateKeyLabel = new QLabel("私钥: 未生成", this);

    generateButton = new QPushButton("生成密钥", this);
    encryptButton = new QPushButton("加密", this);
    decryptButton = new QPushButton("解密", this);

    gridLayout->addWidget(new QLabel("p:"), 0, 0);
    gridLayout->addWidget(pInput, 0, 1);
    gridLayout->addWidget(new QLabel("q:"), 1, 0);
    gridLayout->addWidget(qInput, 1, 1);
    gridLayout->addWidget(new QLabel("e:"), 2, 0);
    gridLayout->addWidget(eInput, 2, 1);
    gridLayout->addWidget(new QLabel("输入:"), 3, 0);
    gridLayout->addWidget(messageInput, 3, 1);
    gridLayout->addWidget(new QLabel("结果:"), 4, 0);
    gridLayout->addWidget(resultOutput, 4, 1);

    mainLayout->addLayout(gridLayout);
    mainLayout->addWidget(generateButton);
    mainLayout->addWidget(encryptButton);
    mainLayout->addWidget(decryptButton);
    mainLayout->addWidget(publicKeyLabel);
    mainLayout->addWidget(privateKeyLabel);

    connect(generateButton, &QPushButton::clicked, this, &MainWindow::generateKeys);
    connect(encryptButton, &QPushButton::clicked, this, &MainWindow::encryptMessage);
    connect(decryptButton, &QPushButton::clicked, this, &MainWindow::decryptMessage);

    setWindowTitle("RSA加密演示");
    resize(400, 300);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::generateKeys()
{
    p = pInput->text().toLongLong();
    q = qInput->text().toLongLong();
    e = eInput->text().toLongLong();

    if (p == q)
    {
        QMessageBox::warning(this, "错误", "p和q必须是不同的素数！");
        return;
    }

    if (!isPrime(p) || !isPrime(q))
    {
        QMessageBox::warning(this, "错误", "p和q必须是素数！");
        return;
    }

    n = p * q;
    phi = (p - 1) * (q - 1);

    if (e <= 1 || e >= phi || gcd(e, phi) != 1)
    {
        QMessageBox::warning(this, "错误", "e必须满足 1 < e < φ(n) 且与φ(n)互质！");
        return;
    }

    d = modInverse(e, phi);

    publicKeyLabel->setText(QString("公钥: (e=%1, n=%2)").arg(e).arg(n));
    privateKeyLabel->setText(QString("私钥: (d=%1)").arg(d));
}

void MainWindow::encryptMessage()
{
    QString plainText = messageInput->text();
    QVector<int64_t> plainBlocks = stringToBlocks(plainText);
    QVector<int64_t> cipherBlocks = encryptCBC(plainBlocks);

    QString result;
    for (int64_t block : cipherBlocks)
    {
        result += QString::number(block) + " ";
    }
    resultOutput->setText(result.trimmed());
}

void MainWindow::decryptMessage()
{
    QString cipherText = messageInput->text();
    QStringList cipherList = cipherText.split(" ", Qt::SkipEmptyParts);
    QVector<int64_t> cipherBlocks;
    for (const QString &block : cipherList)
    {
        cipherBlocks.append(block.toLongLong());
    }

    QVector<int64_t> plainBlocks = decryptCBC(cipherBlocks);
    QString result = blocksToString(plainBlocks);
    resultOutput->setText(result);
}

int64_t MainWindow::gcd(int64_t a, int64_t b)
{
    while (b != 0)
    {
        int64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int64_t MainWindow::modPow(int64_t base, int64_t exp, int64_t modulus)
{
    int64_t result = 1;
    base = base % modulus;
    while (exp > 0)
    {
        if (exp & 1)
            result = (result * base) % modulus;
        base = (base * base) % modulus;
        exp >>= 1;
    }
    return result;
}

int64_t MainWindow::modInverse(int64_t a, int64_t m)
{
    int64_t m0 = m, t, q;
    int64_t x0 = 0, x1 = 1;
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

bool MainWindow::isPrime(int64_t n)
{
    if (n <= 1)
        return false;
    if (n <= 3)
        return true;
    if (n % 2 == 0 || n % 3 == 0)
        return false;
    for (int64_t i = 5; i * i <= n; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

int64_t MainWindow::blockSize() const
{
    // 计算适合 n 的最大字节数
    int64_t maxBlock = n - 1;
    int bytes = 0;
    while (maxBlock > 0)
    {
        maxBlock >>= 8;
        bytes++;
    }
    return bytes - 1; // 留出余量
}

// 将字符串转换为数字块，使用小端序
QVector<int64_t> MainWindow::stringToBlocks(const QString &text)
{
    QByteArray bytes = text.toUtf8();  // 将字符串转为 UTF-8 字节数组
    int bSize = blockSize();  // 获取块大小
    QVector<int64_t> blocks;  // 存储生成的块

    for (int i = 0; i < bytes.size(); i += bSize)
    {
        int64_t block = 0;
        // 使用小端序打包字节
        for (int j = 0; j < bSize && (i + j) < bytes.size(); ++j)
        {
            block |= ((unsigned char)bytes[i + j]) << (j * 8); // 将字节放入块中，低位到高位
        }

        // 处理最后一个块的填充
        if (i + bSize > bytes.size())
        {
            int remaining = bytes.size() % bSize; // 计算剩余字节数
            if (remaining == 0)
                remaining = bSize;
            int padding = bSize - remaining; // 需要填充的字节数
            block |= padding << (remaining * 8); // 在高位添加 PKCS5/PKCS7 填充
        }
        blocks.append(block);  // 添加块到结果
    }
    return blocks;
}

// 将数字块转换回字符串，使用小端序
QString MainWindow::blocksToString(const QVector<int64_t> &blocks)
{
    QByteArray bytes; // 存储重组的字节
    int bSize = blockSize(); // 获取块大小

    for (int i = 0; i < blocks.size(); ++i)
    {
        int64_t block = blocks[i];
        // 按小端序提取字节（低字节在前）
        for (int j = 0; j < bSize; ++j)
        {
            bytes.append((char)(block >> (j * 8) & 0xFF));  // 从低位到高位提取字节
        }
    }

    // 移除 PKCS5/PKCS7 填充
    int padding = bytes[bytes.size() - 1];  // 最后一个字节表示填充长度
    if (padding <= bSize)
    {
        bytes.chop(padding);  // 删除填充字节
    }

    return QString::fromUtf8(bytes);  // 将字节数组转为字符串
}

// 使用 CBC 模式加密块
QVector<int64_t> MainWindow::encryptCBC(const QVector<int64_t> &plainBlocks)
{
    QVector<int64_t> cipherBlocks;  // 存储加密后的块
    int64_t previous = 0;   // 前一个密文块，初始为 0
    int bSize = blockSize();  // 获取块大小

    for (int i = 0; i < plainBlocks.size(); ++i)
    {
        int64_t block = plainBlocks[i];
        // 对第一个块使用 IV，其他块使用前一个密文进行 XOR
        if (i == 0)
        {
            for (int j = 0; j < bSize && j < iv.size(); ++j)
            {
                block ^= ((unsigned char)iv[j]) << (j * 8);  // 按小端序异或 IV
            }
        }

        else
        {
            block ^= previous;  // 与前一个密文块 XOR
        }
        int64_t cipher = modPow(block, e, n);  // RSA 加密
        cipherBlocks.append(cipher); // 添加密文块
        previous = cipher;  // 更新前一个密文块
    }
    return cipherBlocks;
}

// 使用 CBC 模式解密块
QVector<int64_t> MainWindow::decryptCBC(const QVector<int64_t> &cipherBlocks)
{
    QVector<int64_t> plainBlocks;  // 存储解密后的块
    int bSize = blockSize(); // 获取块大小

    for (int i = 0; i < cipherBlocks.size(); ++i)
    {
        int64_t block = modPow(cipherBlocks[i], d, n);  // RSA 解密
        // 对第一个块使用 IV，其他块使用前一个密文进行 XOR
        if (i == 0)
        {
            for (int j = 0; j < bSize && j < iv.size(); ++j)
            {
                block ^= ((unsigned char)iv[j]) << (j * 8);  //  小端序异或 IV
            }
        }

        else
        {
            block ^= cipherBlocks[i - 1];  // 与前一个密文块 XOR
        }
        plainBlocks.append(block);  // 添加到明文块
    }
    return plainBlocks;
}
