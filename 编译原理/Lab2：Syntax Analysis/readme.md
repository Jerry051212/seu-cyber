实验文法：
RE：空白忽略(“ ”)、数字(0~9)、四则运算符(+, -,*,/)、括号(“(”，“)”)、函数名(sin,cos,lg,ln)、非法字符。

词类定义：
NUM (数字)、PLUS (+)、MINUS (-)、STAR (*)、SLASH (/)、LPAREN ( ()、RPAREN ( ) )、SIN (sin)、COS (cos)、LG (lg)、LN (ln)、EOL (换行)、ERROR (非法字符)。

测试用例输入硬编码在LR1.cpp程序内部，可以输入正确或错误的语句。

LR1.cpp可以用visual studio打开或者直接使用g++(std=c++11)编译运行
