// Algorithm:铺砖问题.cpp
#include <iostream>   // 包含输入输出流库
using namespace std;  // 使用标准命名空间

#define MAX 1025   // 定义宏MAX为1025
int k;  // 棋盘都大小
int x, y;  // 特殊方格的位置
int board[MAX][MAX];  // 初始化一个大小为MAX*MAX的棋盘数组
int tile = 1;  // 初始瓷砖号为1

void ChessBoard(int tr, int tc, int dr, int dc, int size) {  // 定义一个棋盘分割函数，参数包括起始行列(tr, tc)、特殊方格位置(dr, dc)以及尺寸size
    if (size == 1) return;  // 当尺寸为1时，返回

    int t = tile++;  // 将瓷砖号赋值给t，并递增瓷砖号
    int s = size / 2;  // 棋盘分割尺寸取一半

    if (dr < tr + s && dc < tc + s) {  // 特殊方格在左上子棋盘中
        ChessBoard(tr, tc, dr, dc, s);
    }
    else {
        board[tr + s - 1][tc + s - 1] = t;  // 记录特殊方格所在位置
        ChessBoard(tr, tc, tr + s - 1, tc + s - 1, s);  // 递归处理左上子棋盘
    }

    if (dr < tr + s && dc >= tc + s) {  // 特殊方格在右上子棋盘中
        ChessBoard(tr, tc + s, dr, dc, s);
    }
    else {
        board[tr + s - 1][tc + s] = t;
        ChessBoard(tr, tc + s, tr + s - 1, tc + s, s);  // 递归处理右上子棋盘
    }

    if (dr >= tr + s && dc < tc + s) {  // 特殊方格在左下子棋盘中
        ChessBoard(tr + s, tc, dr, dc, s);
    }
    else {
        board[tr + s][tc + s - 1] = t;
        ChessBoard(tr + s, tc, tr + s, tc + s - 1, s);  // 递归处理左下子棋盘
    }

    if (dr >= tr + s && dc >= tc + s) {  // 特殊方格在右下子棋盘中
        ChessBoard(tr + s, tc + s, dr, dc, s);
    }
    else {
        board[tr + s][tc + s] = t;
        ChessBoard(tr + s, tc + s, tr + s, tc + s, s);  // 递归处理右下子棋盘
    }
}

int main() {
    k = 4;  // 设置棋盘大小为2^4
    srand((unsigned)time(NULL));
    x = rand() % 16;
    y = rand() % 16;
    // x = 1, y = 2;  // 特殊方格的位置(x, y)
    int size = 1 << k;  // 计算棋盘总尺寸
    ChessBoard(0, 0, x, y, size);  // 调用棋盘分割函数处理特殊方格
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%4d", board[i][j]);  // 打印棋盘中的数字
        }
        printf("\n");  // 换行
    }
    return 0;  // 返回执行成功
}

