# CTF_Misc，题目：moeCTF2019-Kokoro
moectf{d0_y0u_kn0w_br@1nfuck}
解题步骤：

1.文本由数字、.、!、? 组成。数字表示后面那个符号的重复次数（例如 20. 就是 20 个 .）。

2.展开重复后，得到的是 Ook! 编码（Brainfuck 的变种）。
符号对映射关系为：
.. → +
!! → -
.? → >
?. → <
!. → .
.! → ,
!? → [
?! → ]

3.转换成标准 Brainfuck 代码后直接解释执行，输出即为 flag：
moectf{d0_y0u_kn0w_br@1nfuck}
