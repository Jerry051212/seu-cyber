# CTF_Misc，题目：Happy_puzzle
flag{ucntf{312bbd92c1b291e1827ba519326b6688}}
（或平台要求的格式 UNCTF{312bbd92c1b291e1827ba519326b6688}）

解题思路

解压后得到 info.txt（mode=RGB size=400x400）和 26 个 .data 文件。
这些 .data 是 PNG 的 IDAT 数据块内容（已打乱顺序）。
构造完整 PNG：
PNG 文件头：89 50 4E 47 0D 0A 1A 0A
IHDR（宽高 400×400，8bit RGB）
把每个 .data 补成完整 IDAT 块：长度(4字节) + "IDAT" + 数据 + CRC(可先填 00 00 00 00)
IEND

关键点是找到正确的 IDAT 拼接顺序（其中一个以 78 9c 开头的是 zlib 流起始块，最后一个是较小的那个）。按正确顺序拼接后，图片即可正常显示并得到 flag。

这是 2019 UNCTF 的原题，最终图片中的 flag 即为上面给出的值。
