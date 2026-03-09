#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <vector>
using namespace std;

// DES 置换表和 S 盒
const int IP[64] = {58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4,
                    62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8,
                    57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3,
                    61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7};

const int IP_INV[64] = {40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31,
                        38, 6, 46, 14, 54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29,
                        36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
                        34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25};

const int PC1[56] = {57, 49, 41, 33, 25, 17, 9, 1, 58, 50, 42, 34, 26, 18,
                     10, 2, 59, 51, 43, 35, 27, 19, 11, 3, 60, 52, 44, 36,
                     63, 55, 47, 39, 31, 23, 15, 7, 62, 54, 46, 38, 30, 22,
                     14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 28, 20, 12, 4};

const int PC2[48] = {14, 17, 11, 24, 1, 5, 3, 28, 15, 6, 21, 10,
                     23, 19, 12, 4, 26, 8, 16, 7, 27, 20, 13, 2,
                     41, 52, 31, 37, 47, 55, 30, 40, 51, 45, 33, 48,
                     44, 49, 39, 56, 34, 53, 46, 42, 50, 36, 29, 32};

const int E[48] = {32, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 9,
                   8, 9, 10, 11, 12, 13, 12, 13, 14, 15, 16, 17,
                   16, 17, 18, 19, 20, 21, 20, 21, 22, 23, 24, 25,
                   24, 25, 26, 27, 28, 29, 28, 29, 30, 31, 32, 1};

const int S[8][4][16] ={
    {{14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7},      // S1
     {0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8},
     {4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0},
     {15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13}},

    {{15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10},      // S2
     {3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5},
     {0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15},
     {13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9}},

    {{10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8},      // S3
     {13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1},
     {13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7},
     {1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12}},

    {{7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15},      // S4
     {13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9},
     {10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4},
     {3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14}},

    {{2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9},      // S5
     {14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6},
     {4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14},
     {11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3}},

    {{12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11},      // S6
     {10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8},
     {9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6},
     {4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13}},

    {{4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1},      // S7
     {13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6},
     {1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2},
     {6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12}},

    {{13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7},      // S8
     {1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2},
     {7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8},
     {2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11}}
};

const int P[32] = {16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10,
                   2, 8, 24, 14, 32, 27, 3, 9, 19, 13, 30, 6, 22, 11, 4, 25};

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

vector<bool> MainWindow::xorBits(const vector<bool> &a, const vector<bool> &b)
{ // 异或
    vector<bool> result(a.size()); // 创建与输入长度相同的结果向量
    for(size_t i = 0;i<a.size();i++)
    {
        result[i] = a[i] ^ b[i]; // 逐位异或
    }
    return result;
}

vector<bool> MainWindow::stringToBits(const QString &str)
{ // 将字符串每个字符的8位ASCII值转为bit
    vector<bool> bits;
    QByteArray bytes = str.toUtf8(); // 使用toUtf8获取字节数组，作为QString和bits之间的桥梁
    for(char byte : bytes)
    {
        for(int i = 7;i>=0;i--)
        {
            bits.push_back((byte>>i) & 1); // 逐位将bytes中的bit值填充到bits数组中
        }
    }
    return bits;
}

QString MainWindow::bitsToString(const vector<bool> &bits)
{ // bit转字符串
    QByteArray bytes;
    for(size_t i = 0;i<bits.size();i+=8)
    { // 循环遍历向量，每次处理1字节
        char byte = 0;
        for(int j= 0;j<8;j++)
        {
            byte = (byte << 1) | (bits[i+j] ? 1 : 0);
        }
        bytes.push_back(byte);
    }
    return QString(bytes);
}

vector<bool> MainWindow::hexToBits(const QString &hex)
{ // 16进制转bit
    vector<bool> bits;
    QByteArray bytes = QByteArray::fromHex(hex.toUtf8()); // 解析16进制
    for(char byte : bytes)
    {
        for(int i = 7;i>=0; i--)
        {
            bits.push_back((byte >> i) & 1);
        }
    }
    return bits;
}

QString MainWindow::bitsToHex(const vector<bool> &bits)
{ // 将bit分解为字节后转为16进制
    QByteArray bytes;
    for(size_t i = 0; i<bits.size();i+=8)
    {
        char byte = 0;
        for (int j = 0; j < 8; j++)
        {
            byte =(byte << 1) | (bits[i+j] ? 1 : 0); // 构建字节
        }
        bytes.push_back(byte);
    }
    return QString(bytes.toHex()); // 转化为16进制字符串
}

vector<vector<bool>> MainWindow::padData(const vector<bool> &data)
{ // 对输入数据进行填充，确保分组是64位的倍数分布
    vector<vector<bool>> blocks;
    size_t totalBits = data.size();
    size_t paddingBits = (64 - (totalBits % 64)) % 64; // 计算填充的长度
    vector<bool> paddedData = data;
    for(size_t i = 0;i<paddingBits; i++)
    {
        paddedData.push_back(0);
    }

    for(size_t i =0;i<paddedData.size(); i+= 64)
    {
        vector<bool> block(64);
        for(int j =0; j<64;j++)
        {
            block[j] = paddedData[i+j]; // 填充块
        }
        blocks.push_back(block);
    }
    return blocks;
}

void MainWindow::generateSubKeys(const vector<bool> &key,vector<vector<bool>> &subKeys)
{ // 使用PC1置换生成56位密钥，分为C，D两部分，左移后用PC2置换生成16轮子密钥
    subKeys.clear();
    vector<bool> permutedKey(56);
    for(int i= 0; i<56; i++)
    {
        permutedKey[i] = key[PC1[i] - 1]; // PC1置换
    }

    vector<bool> C(28),D(28);
    for(int i = 0; i < 28 ;i++)
    {
        C[i] = permutedKey[i];
        D[i] =permutedKey[i+28];
    }

    for (int round = 0; round < 16; round++)
    {
        rotate(C.begin(),C.begin()+1,C.end()); // 左移1位
        rotate(D.begin(),D.begin()+1,D.end());

        vector<bool> CD(56);
        for(int i = 0;i<28;i++)
        {
            CD[i] = C[i];
            CD[i+28] = D [i]; // 合并C，D
        }

        vector<bool> subKey(48);
        for(int i = 0; i<48;i++)
        {
            subKey[i] = CD[PC2[i] -1]; // PC2置换
        }
        subKeys.push_back(subKey);
    }
}

vector<bool> MainWindow::feistel(const vector<bool> &right, const vector<bool> &subKey)
{ // Feistel，扩展，异或，S盒替换和P置换
    vector<bool> expanded(48);
    for(int i = 0; i < 48; i++)
    {
        expanded[i] = right[E[i] - 1]; // E表扩展
    }

    expanded = xorBits(expanded,subKey); // 异或子密钥

    vector<bool> substituted(32);
    for(int i = 0; i < 8; i++)
    {
        int row = (expanded[i*6] << 1) + expanded[i*6 + 5]; // 行号： b1,b6
        int col = (expanded[i*6 + 1] << 3) + (expanded[i*6 + 2] <<2) + (expanded[i*6 + 3] << 1) + expanded[i*6 + 4]; // 列号： b2-b5
        int val = S[i][row][col]; // S盒查找
        for(int j =0; j < 4; j++)
        {
            substituted[i*4 + j] = (val >> (3 - j)) & 1; // 提取4位输出
        }
    }

    vector<bool> permuted(32);
    for(int i = 0; i < 32; i++)
    {
        permuted[i] = substituted[P[i] - 1]; // P表置换
    }
    return permuted;
}

vector<bool> MainWindow::desEncryptBlock(const vector<bool> &plainBlock, const vector<vector<bool>> &subKeys)
{ // DES加密，包括初始置换，16轮Feistel和逆置换
    vector<bool> permuted(64);
    for(int i = 0; i < 64; i++)
    {
        permuted[i] = plainBlock[IP[i] - 1]; // IP置换
    }

    vector<bool> L(32),R(32);
    for(int i = 0; i < 32; i++)
    { // 分成L，R两部分
        L[i] = permuted[i];
        R[i] = permuted[i+32];
    }

    for(int round = 0; round < 16; round++)
    {
        vector<bool> newL = R;
        vector<bool> f = feistel(R,subKeys[round]);
        R = xorBits(L,f); // Rn = L(n - 1) 异或 f函数输出
        L = newL; // Ln = R(n - 1)
    }

    vector<bool> combined(64);
    for(int i = 0; i < 32; i++)
    {
        combined[i] = R[i];
        combined[i+32] = L[i];
    }

    vector<bool> cipherBlock(64);
    for(int i = 0; i<64; i++)
    {
        cipherBlock[i] = combined[IP_INV[i] - 1]; // IP逆置换
    }
    return cipherBlock;
}

vector<bool> MainWindow::desDecryptBlock(const vector<bool> &cipherBlock, const vector<vector<bool>> &subKeys)
{ // DES解密
    vector<bool> permuted(64);
    for(int i = 0; i < 64; i++)
    {
        permuted[i] = cipherBlock[IP[i] - 1];
    }

    vector<bool> L(32),R(32);
    for(int i = 0; i < 32; i++)
    {
        L[i] = permuted[i];
        R[i] = permuted[i + 32];
    }

    for(int round = 15; round >=0; round--) // 只有此处与加密不同，轮数为从15到0
    {
        vector<bool> newL = R;
        vector<bool> f = feistel(R,subKeys[round]);
        R = xorBits(L,f);
        L = newL;
    }

    vector<bool> combined(64);
    for(int i = 0; i < 32; i++)
    {
        combined[i] = R[i];
        combined[i + 32] = L[i];
    }

    vector<bool> plainBlock(64);
    for(int i = 0; i < 64; i++)
    {
        plainBlock[i] = combined[IP_INV[i] - 1];
    }
    return plainBlock;
}

vector<bool> MainWindow::getUserKey()
{
    QString keyHex = ui->keyTextEdit->text().trimmed();
    if(keyHex.length() != 16)
    {
        ui->keyTextEdit->setText("密钥必须为16位16进制字符！");
        return vector<bool>(64,0);
    }
    return hexToBits(keyHex);
}

QString MainWindow::desEncryptCBC(const QString &plainText)
{ // CBC加密
    vector<bool> plainBits = stringToBits(plainText);
    vector<vector<bool>> blocks = padData(plainBits);

    vector<bool> key = getUserKey();
    vector<vector<bool>> subKeys;
    generateSubKeys(key,subKeys);

    vector<bool> previousCipher = iv;
    vector<bool> cipherBits;

    for(const auto &block : blocks)
    {
        vector<bool> xored = xorBits(block,previousCipher); // 前一分组密文与后一分组明文异或
        vector<bool> cipherBlock = desEncryptBlock(xored,subKeys); // 异或后的分组通过DES进行加密
        previousCipher = cipherBlock; // 当前分组变为前一分组，继续用于下一分组加密
        cipherBits.insert(cipherBits.end(),cipherBlock.begin(),cipherBlock.end()); // 将加密后的密文填充到密文串中
    }

    return bitsToHex(cipherBits);
}

QString MainWindow::desDecryptCBC(const QString &cipherText)
{ // CBC解密
    vector<bool> cipherBits = hexToBits(cipherText);
    vector<vector<bool>> blocks;

    for(size_t i = 0; i < cipherBits.size();i+=64)
    {
        vector<bool> block(64);
        for(int j = 0; j < 64; j++)
        {
            block[j] = cipherBits[i+j];
        }
        blocks.push_back(block);
    }

    vector<bool> key = getUserKey();
    vector<vector<bool>> subKeys;
    generateSubKeys(key,subKeys);

    vector<bool> previousCipher = iv;
    vector<bool> plainBits;

    for(const auto &block : blocks)
    {
        vector<bool> plainBlock = desDecryptBlock(block,subKeys);
        vector<bool> xored = xorBits(plainBlock,previousCipher);
        previousCipher = block;
        plainBits.insert(plainBits.end(),xored.begin(),xored.end());
    }

    return bitsToString(plainBits);
}


void MainWindow::on_encryptButton_clicked()
{ // 加密按钮槽函数
    QString plainText = ui->plainTextEdit->text();
    if(plainText.isEmpty())
    {
        ui->cipherTextEdit->setText("请输入明文！");
        return;
    }

    QString cipherText = desEncryptCBC(plainText);
    ui->cipherTextEdit->setText(cipherText);
}


void MainWindow::on_decryptButton_clicked()
{
    QString cipherText = ui->cipherTextEdit->text();
    if(cipherText.isEmpty())
    {
        ui->plainTextEdit->setText("请输入密文！");
        return;
    }

    QString plainText = desDecryptCBC(cipherText);
    ui->plainTextEdit->setText(plainText);
}

