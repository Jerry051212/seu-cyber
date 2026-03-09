#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <cmath>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 创建中心部件和主布局
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // 创建输入框和标签
    lineEditAlpha = new QLineEdit(this);
    lineEditP = new QLineEdit(this);
    lineEditX = new QLineEdit(this);
    lineEditY = new QLineEdit(this);

    // 创建计算按钮
    calculateButton = new QPushButton("Calculate", this);

    // 创建结果显示标签
    resultLabel = new QLabel("Result will be shown here", this);
    resultLabel->setWordWrap(true);

    // 创建输入布局
    QHBoxLayout *alphaLayout = new QHBoxLayout();
    alphaLayout->addWidget(new QLabel("Public Alpha:"));
    alphaLayout->addWidget(lineEditAlpha);

    QHBoxLayout *pLayout = new QHBoxLayout();
    pLayout->addWidget(new QLabel("Prime p:"));
    pLayout->addWidget(lineEditP);

    QHBoxLayout *xLayout = new QHBoxLayout();
    xLayout->addWidget(new QLabel("A's Private Key (x):"));
    xLayout->addWidget(lineEditX);

    QHBoxLayout *yLayout = new QHBoxLayout();
    yLayout->addWidget(new QLabel("B's Private Key (y):"));
    yLayout->addWidget(lineEditY);

    // 将所有部件添加到主布局
    mainLayout->addLayout(alphaLayout);
    mainLayout->addLayout(pLayout);
    mainLayout->addLayout(xLayout);
    mainLayout->addLayout(yLayout);
    mainLayout->addWidget(calculateButton);
    mainLayout->addWidget(resultLabel);
    mainLayout->addStretch();

    // 连接按钮信号
    connect(calculateButton, &QPushButton::clicked, this, &MainWindow::onCalculateClicked);

    // 设置窗口标题和大小
    setWindowTitle("Diffie-Hellman Key Exchange");
    resize(400, 300);
}

MainWindow::~MainWindow()
{
    delete ui;
}


// 快速模幂算法
long long MainWindow::modPow(long long base, long long exp, long long modulus)
{
    long long result = 1;
    base = base % modulus;
    while (exp > 0)
    {
        if (exp & 1)
            result = (result * base) % modulus;
        exp = exp >> 1;
        base = (base * base) % modulus;
    }
    return result;
}

// 检查是否为素数
bool MainWindow::isPrime(long long n)
{
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

// 检查 alpha 是否为模 p 的生成元
bool MainWindow::isPrimitiveRoot(long long alpha, long long p)
{
    if (alpha <= 1 || alpha >= p)
        return false;

    // 阶必须是 p-1（Z_p^* 的阶）
    long long phi = p - 1;
    QSet<long long> seen;

    // 计算 alpha^k mod p，检查是否生成所有 1 到 p-1
    for (long long k = 1; k <= phi; ++k)
    {
        long long result = modPow(alpha, k, p);
        seen.insert(result);
    }

    // 如果生成的元素个数等于 p-1，则 alpha 是生成元
    return seen.size() == static_cast<size_t>(phi);
}

void MainWindow::onCalculateClicked()
{
    bool ok1, ok2, ok3, ok4;
    long long alpha = lineEditAlpha->text().toLongLong(&ok1);
    long long p = lineEditP->text().toLongLong(&ok2);
    long long x = lineEditX->text().toLongLong(&ok3);
    long long y = lineEditY->text().toLongLong(&ok4);

    // 检查输入是否为有效正整数
    if (!ok1 || !ok2 || !ok3 || !ok4 || alpha <= 0 || p <= 0 || x <= 0 || y <= 0)
    {
        QMessageBox::warning(this, "Input Error", "Please enter valid positive integers.");
        return;
    }

    // 检查 p 是否为素数
    if (!isPrime(p))
    {
        QMessageBox::warning(this, "Input Error", "p must be a prime number.");
        return;
    }

    // 检查 alpha 是否为模 p 的生成元
    if (!isPrimitiveRoot(alpha, p))
    {
        QMessageBox::warning(this, "Input Error", "alpha must be a primitive root modulo p.");
        return;
    }

    // 计算A的公钥: A = alpha^x mod p
    long long A = modPow(alpha, x, p);
    // 计算B的公钥: B = alpha^y mod p
    long long B = modPow(alpha, y, p);
    // 计算共享密钥: K_A = B^x mod p, K_B = A^y mod p
    long long K_A = modPow(B, x, p);
    long long K_B = modPow(A, y, p);

    // 验证共享密钥是否一致
    if (K_A == K_B)
    {
        resultLabel->setText(
            QString("A's Public Key: %1\nB's Public Key: %2\nShared Secret Key: %3")
                .arg(A).arg(B).arg(K_A));
    }
    else
    {
        resultLabel->setText("Error: Shared keys do not match!");
    }
}
