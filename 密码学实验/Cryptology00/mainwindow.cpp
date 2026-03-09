#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <QString>
#include <vector>
#include <algorithm>
using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 解密按钮连接槽函数
    connect(ui->decryptButton, &QPushButton::clicked, this, &MainWindow::onDecryptButtonClicked);

    // 初始化 QTableWidget
    ui->tableWidget->setColumnCount(3);
    ui->tableWidget->setHorizontalHeaderLabels({"Letter","Count", "Frequency (%)"});
    ui->tableWidget->horizontalHeader()->setStretchLastSection(true);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::getFrequency(const QString &ciphertext, vector<tuple<char, int, double>> &result)
{
    map<char, int> letterFreq;
    int totalLetters = 0;

    for (const QChar &c : ciphertext)
    {
        if (c.isLetter())
        {
            char ch = c.toLatin1(); // 将QChar转化为char
            letterFreq[ch]++;
            totalLetters++;
        }
    }

    for (const auto &pair : letterFreq)
    {
        double freq = (double)pair.second / totalLetters * 100;
        result.push_back(make_tuple(pair.first, pair.second, freq));
    }

    sort(result.begin(), result.end(), [](const tuple<char, int, double> &a, const tuple<char, int, double> &b)
    {
        return get<2>(a) > get<2>(b); // 按频率降序排序
    });
}

QString MainWindow::strDecrypt(const QString &ciphertext, const map<char, char> &mapping)
{ // 根据映射进行解密，返回明文
    QString plaintext = "";
    for (const QChar &c : ciphertext)
    {
        char ch = c.toLatin1(); // 将QChar转化为char
        if(mapping.find(ch) != mapping.end())
        {
            plaintext += mapping.at(ch); // 获取对应的明文字母并添加到plaintext
        }

        else
        {
            plaintext += ch;
        }
    }
    return plaintext;
}

void MainWindow::updateTable(const vector<tuple<char, int, double>> &freq)
{ // 将字母频率统计结果显示到QTableWidget
    ui->tableWidget->setRowCount(freq.size());
    for (int i = 0; i < freq.size(); ++i)
    {
        ui->tableWidget->setItem(i, 0, new QTableWidgetItem(QString(get<0>(freq[i]))));
        ui->tableWidget->setItem(i,1,new QTableWidgetItem(QString::number(get<1>(freq[i]))));
        ui->tableWidget->setItem(i, 2, new QTableWidgetItem(QString::number(get<2>(freq[i]),'f',2) + "%"));
    }
}

void MainWindow::updateText(const QString &plaintext)
{
    ui->textEdit->setText(plaintext);
}

void MainWindow::onDecryptButtonClicked()
{

    QString ciphertext = "UZQSOVUOHXMOPVGPOZPEVSGZWSZOPFPESXUDBMETSXAIZVUEPHZHMDZSHZOWSFPAPPDTSVPQUZWYMXUZUH2SXEPYEPOPDZSZUFPOMBZWPFUPZHMDJUDTMOHMQ";


    map<char, char> mapping =
    {
        {'A', 'B'}, {'B', 'F'}, {'C', 'C'}, {'D', 'N'}, {'E', 'R'},
        {'F', 'V'}, {'G', 'Y'}, {'H', 'C'}, {'I', 'U'}, {'J', 'G'},
        {'K', 'K'}, {'L', 'L'}, {'M', 'O'}, {'N', 'N'}, {'O', 'S'},
        {'P', 'E'}, {'Q', 'W'}, {'R', 'R'}, {'S', 'A'}, {'T', 'M'},
        {'U', 'I'}, {'V', 'D'}, {'W', 'H'}, {'X', 'L'}, {'Y', 'P'},
        {'Z', 'T'}
    };


    vector<tuple<char,int,double>> freq;
    getFrequency(ciphertext, freq);
    updateTable(freq);


    QString plaintext = strDecrypt(ciphertext, mapping);
    updateText(plaintext);


    for (const auto &f : freq)
    {
        cout << get<0>(f) << ": " << get<1>(f) << " times, " << get<2>(f) << "%" << endl;
    }
}
