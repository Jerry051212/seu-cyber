该文件夹中为57123130刘懿仁编译方法实验一的代码和测试程序：
myflex.cpp：自己编程实现的flex，支持多种文法的.l文件编译自动生成.cpp文件
myflex.exe：由myflex.cpp编译得到的可执行程序
lexer.l：根据实验的文法编写的.l文件
lexer.cpp：由myflex.exe根据lexer.l自动生成的lexical analyzer程序代码
lexer.exe：由lexer.exe编译得到的可执行程序
new_lexer.l：为了验证flex普适性的新文法的.l文件
new_lexer.cpp、new_lexer.exe：同上，新文法的lexical analyzer程序。

实验文法：
RE：空白忽略(“ ”)、数字(0~9)、四则运算符(+, -,*,/)、括号(“(”，“)”)、函数名(sin,cos,lg,ln)、非法字符。
词类定义：
NUM (数字)、PLUS (+)、MINUS (-)、STAR (*)、SLASH (/)、LPAREN ( ()、RPAREN ( ) )、SIN (sin)、COS (cos)、LG (lg)、LN (ln)、EOL (换行)、ERROR (非法字符)。