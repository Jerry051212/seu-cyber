# CTF_crypto，题目：告诉你个秘密
flag{TONGYUAN}
解题步骤：
文件内容是两段十六进制字符串，拼接后用 hex 解码得到：
cjV5RyBscDlJIEJqTSB0RmhCVDZ1aCB5N2lKIFFzWiBiaE0g
再对该字符串做 Base64 解码，得到带空格的分组：
r5yG lp9I BjM tFhBT6uh y7iJ QsZ bhM
这些分组对应 QWERTY 键盘上的按键位置，每组字母正好围住中间一个键：
r5yG → T
lp9I → O
BjM → N
tFhB → G
T6uh → Y
y7iJ → U
QsZ → A
bhM → N

拼起来即为 TONGYUAN。
