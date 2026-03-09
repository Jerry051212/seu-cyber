#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>
#include <QWidget>
#include <QMessageBox>

// 构造函数：设置UI界面
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // 创建主窗口部件
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // 创建布局
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    // 输入框1
    num1Input = new QLineEdit(this);
    num1Input->setPlaceholderText("Enter first number");
    layout->addWidget(num1Input);

    // 输入框2
    num2Input = new QLineEdit(this);
    num2Input->setPlaceholderText("Enter second number");
    layout->addWidget(num2Input);

    // 计算按钮
    calcButton = new QPushButton("Calculate", this);
    layout->addWidget(calcButton);

    // 显示结果
    gcdResult = new QLineEdit(this);
    gcdResult->setReadOnly(true);
    layout->addWidget(gcdResult);

    inverseResult = new QLineEdit(this);
    inverseResult->setReadOnly(true);
    layout->addWidget(inverseResult);

    sResult = new QLineEdit(this);
    sResult->setReadOnly(true);
    layout->addWidget(new QLabel("s (贝祖系数1):", this));  // 添加标签说明
    layout->addWidget(sResult);

    tResult = new QLineEdit(this);
    tResult->setReadOnly(true);
    layout->addWidget(new QLabel("t (贝祖系数2):", this));  // 添加标签说明
    layout->addWidget(tResult);

    // 连接按钮信号到计算槽函数
    connect(calcButton, &QPushButton::clicked, this, &MainWindow::calculate);

    // 设置窗口属性
    setWindowTitle("GCD and Modular Inverse Calculator");
    resize(300, 200);
}

MainWindow::~MainWindow()
{
    delete ui;
}

// 迭代实现的扩展欧几里得算法
MainWindow::GCDResult MainWindow::extendedGCD(int64_t a, int64_t b)
{
    int64_t x = 1, y = 0;
    int64_t x_last = 0, y_last = 1;

    while (b != 0)
    {
        int64_t quotient = a / b;
        int64_t remainder = a % b;

        int64_t temp_x = x;
        x = x_last;
        x_last = temp_x - quotient * x_last;

        int64_t temp_y = y;
        y = y_last;
        y_last = temp_y - quotient * y_last;

        a = b;
        b = remainder;
    }

    return {a, x, y};
}

// 计算模乘逆元
int64_t MainWindow::modInverse(int64_t a, int64_t m)
{
    GCDResult result = extendedGCD(a, m);
    if (result.gcd != 1)
    {
        return -1;
    }
    return (result.x % m + m) % m;
}

// 计算最大公因数
int64_t MainWindow::gcd(int64_t a, int64_t b)
{
    if (b != 0)
    {
        GCDResult result = extendedGCD(a,b);
        return result.gcd;
    }
}

// 计算并显示结果
void MainWindow::calculate()
{
    bool ok1, ok2;
    int64_t num1 = num1Input->text().toLongLong(&ok1);
    int64_t num2 = num2Input->text().toLongLong(&ok2);

    if (!ok1 || !ok2 || num2 == 0)
    {
        QMessageBox::warning(this, "Error", "Please enter valid integers (second number cannot be 0)");
        return;
    }

    // 计算GCD
    int64_t resultGCD = gcd(num1, num2);
    GCDResult result = extendedGCD(num1,num2);
    gcdResult->setText(QString("%1").arg(resultGCD));

    // 计算模乘逆元
    int64_t inv = modInverse(num1, num2);
    if (inv == -1)
    {
        inverseResult->setText("Does not exist");
    }
    else
    {
        inverseResult->setText(QString("%1").arg(inv));
    }

    sResult->setText(QString("%1").arg(result.x));
    tResult->setText(QString("%1").arg(result.y));
}
