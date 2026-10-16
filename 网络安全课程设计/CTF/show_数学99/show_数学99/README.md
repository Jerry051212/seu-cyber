# CTF_PWN，题目：show_数学99
Exploit 思路
二进制是 64 位 PIE，有 canary，主要逻辑是三道“数学题”，全部通过后才会进入 win（puts("You are so clever!"); system("cat flag");）。
三道题的突破点

a - b = 9 且 0 ≤ a,b < 9
代码用 strchr 禁止输入中出现 -，再用 atoi 后做有符号比较。
利用 atoi 对超大数字的 32 位截断/回绕：
a = 4294967304 → atoi 结果 = 8
b = 4294967295 → atoi 结果 = -1
8 - (-1) = 9，且字符串里没有 -。

a * b = 9 且 a > 9, b > 9
32 位有符号乘法溢出：
a = 9629
b = 446045
(int32_t)(9629 * 446045) == 9

a / b = ERROR, b ≠ 0
设置 signal(SIGFPE, win) 后执行有符号除法。
经典触发方式：INT_MIN / -1 产生有符号溢出，引发 SIGFPE，直接跳进 win 函数执行 system("cat flag")（不会回到原来的返回值检查）。

一键运行脚本show_99.py
