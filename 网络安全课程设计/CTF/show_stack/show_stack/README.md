# CTF_PWN，题目：show_stack
nc 211.65.197.50 21003
flag{a8vugkja4cxfqdpr7devumty}（每个人不同）
简要分析：

32 位无保护栈溢出。
pwnme() 中 fgets(buf, 0x32, stdin)，buf 位于 ebp-0x9。
到返回地址的偏移 = 9 + 4 = 13 字节。
程序自带 stack() 函数直接执行 system("/bin/sh")（地址 0x0804850f）。

Payload：
textb"A"*13 + p32(0x0804850f)
发送后拿到 shell，cat flag 即可。
