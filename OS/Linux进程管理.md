# 实验一 Linux进程管理及其扩展
## 一、实验内容
1. 阅读并分析Linux内核源代码，了解进程控制块、进程队列等数据结构；新增任意一个系统调用并测试。 

2. 实现一个系统调用，使得可以根据指定的参数隐藏进程，使用户无法使用ps或top观察到进程状态。具体要求如下： 

（1）实现系统调用int hide(pid_t pid, int on)，在进程pid有效的前提下，如果on置1，进程被隐藏，用户无法通过ps或top观察到进程状态；如果on置0且此前为隐藏状态，则恢复正常状态。 

（2）考虑权限问题，只有根用户才能隐藏进程。 

（3）设计一个新的系统调用int hide_user_processes(uid_t uid, char *binname)，参数 uid为用户ID号，当 binname参数为NULL时，隐藏该用户的所有进程；否则，隐藏二进制映像名为binname的用户进程。该系统调用应与hide系统调用共存。 

（4）在/proc 目录下创建一个文件/proc/hidden，该文件可读可写，对应一个全局变量
hidden_flag，当 hidden_flag 为 0 时，所有进程都无法隐藏，即便此前进程被hide系统调用要求隐藏。只有当hidden_flag为1时，此前通过hide调用要求被屏蔽的进程才隐藏起来。 

（5）在/proc目录下创建一个文件/proc/hidden_process，该文件的内容包含所有被隐藏进程的pid，各pid之间用空格分开。
## 二、实验目的
通过实验，加深理解进程控制块、进程队列等概念，了解进程管理的具体实施方法。
## 三、设计思路和流程图

1、	Linux内核代码分析，新增系统调用：

（1）解压内核

（2）生成内核配置文件

（3）编译安装内核

（4）新增系统调用
 
 4.1 在文件arch/i386/kernel/syscall_table.S的尾部加上要新增的系统调用函数的名称
 
 4.2 在include/linux目录下添加头文件在kernel目录下新建文件.c，在该文件中实现函数
 
 4.3 修改文件kernel/Makefile，使得.c在编译时可见\
 
 4.4 在include/asm-i386/unistd.h里加上系统调用号的宏定义
 
 4.5 修改include/linux/syscalls.h，加上函数系统调用的声明。在该文件的”#include <linux/types.h>之后添加
 
 4.6 重新编译内核

2、Linux进程管理及其扩展

（1）实现系统调用hide

1.1.	在include/linux/sched.h中修改task_struct，添加一个成员cloak，用来记录进程隐藏与否。

1.2.	在进程创建时，将task_struct的成员cloak初始化为未隐藏。

1.3.	添加hide系统调用

1.4.	修改proc_pid_readdir函数（在fs/proc/base.c文件中），其中使用for循环遍历进程，在遍历过程中添加判断，过滤掉被隐藏的进程。

1.5.	修改proc_pid_lookup函数，在进程查找完成前过滤掉被隐藏的进程。

（2）考虑权限问题，只有根用户才能隐藏进程

（3）实现系统调用hide_user_processes

（4） hidden文件

4.1. 在/fs/proc/proc_misc.c中添加回调函数，首先在全局作用域定义变量hidden_flag。然后定义hidden文件的读写回调函数。

4.2. 在/fs/proc/proc_misc.c中proc_misc_init函数的最后添加创建hidden文件的代码，并指定其回调函数。

4.3. hidden文件创建成功后，需要实现通过全局变量hidden_flag来约束隐藏进程的函数

（5）hidden_process文件，方法和创建hidden文件一样，hidden_process文件只需要设置读的回调函数即可。输出所有被隐藏进程的pid只要输出所有cloak为1的进程pid即可。用遍历所有进程的方法，判断cloak的值来决定是否在回调函数中输出。

## 四、主要数据结构及其说明

1、使用了结构体task_struct，通过对结构体里的cloak等变量进行赋值，改变进程的状态。

2、使用了结构体task_struct，通过其中的cloak等变量，控制进程的状态。

3、使用到了结构体、进程控制块等数据结构。

4、使用了结构体、进程控制块、进程队列等数据结构。

## 五、源程序并附上注释（关键部分）
```
// psta.h
#ifndef _LINUX_PSTA_H
#define _LINUX_PSTA_H
struct stu
{
int ID;
char name;
};
#endif

// psta.c
#include <linux/linkage.h>
#include<linux/types.h>
#include<linux/psta.h>
#include<linux/kernel.h>

asmlinkage int sys_psta(struct stu *buf)
{
return buf->ID;

}

// hide.c
#include<stdio.h>
#include<sys/syscall.h>
#include<unistd.h>
int main()
{
    int syscallNum=322;
    uid_t uid=0;
    char *binname="init";
    int recover=0;
    syscall(syscallNum,uid,binname,recover);
    return 0;
}

// hide_user_process.c
#include<stdio.h>
#include<sys/syscall.h>
#include<unistd.h>
int main()
{
    int syscallNum=322;
    uid_t uid=0;
    char *binname="init";
    int recover=0;
    syscall(syscallNum,uid,binname,recover);
    return 0;
}
```
## 六、实验运行结果及分析
新增任意系统调用：
1、	设置新增系统调用的名称及编号，修改文件 arch/i386/kernel/syscall_table.S
 
2、设置系统调用的具体内容
在include/linux目录下添加添加头文件psta.h。在kernel目录下新建文件 psta.c ，实现函数 psta。 宏asmlinkage定义在linux/linkage.h中，表示函数的参数通过栈传递，而不是寄存器，所有的系统调用都遵循这种参数传递方式。

3、使该系统调用在编译时可见，修改文件kernel/Makefile
 
4、加上系统调用号的宏定义，修改文件include/asm-i386/unistd.h

5、加上函数sys_select_sort的声明，修改文件include/linux/syscalls.h
 
6、重新编译内核
linux进程管理及扩展：

1、初始状态，进程都是处于显示的状态，我们的目的是隐藏1号进程。 非root用户下执行测试程序，进程未被隐藏，用 dmesg 命令查看输出。 切换到root用户，再次执行程序，结果如下，1号进程被隐藏了。 更改参数on=0，被隐藏的进程将再次出现
 
2、	新增系统调用hide_user_processes

（1）与上个实验一样，如果非root用户，则没有隐藏进程的权限
 
（2）隐藏uid为0，进程名为init的进程，修改参数uid_t uid=0; char *binname="init";结果如下，对应的进程被隐藏了
 
（3）隐藏uid=500（对应用户名为seu）的所有进程，修改参数uid_t uid=500; char *binname=NULL;结果如下，seu用户的所有进程被隐藏了

 
(4)更改参数recover=1，所有进程将恢复为显示状态
 
3、	在/proc目录下创建一个文件/proc/hidden

（1）首先默认设置hidden_flag=1，使用hide_user_processes隐藏uid=500，即seu用户的所有进程。
 
（2）hidden_flag的值改为0，这时再查看进程，所有被隐藏的进程又出现了。
 
 
（3）将hidden_flag改回为1，查看进程，seu用户的进程又处于隐藏状态了。
 
 
4、	在/proc目录下创建一个文件/proc/hidden_process

（1）首先隐藏uid=500，即seu用户的所有进程。
 
（2）恢复所有进程为显示状态，这时没有被隐藏的进程，hidden_process文件里的内容也为空。
