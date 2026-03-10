# 实验一 Shell的实现
## 一、实验内容
实现具有管道、重定向功能的shell

能够执行一些简单的基本命令，如进程执行、列目录等
## 二、实验目的
1.学习并理解linux中shell的知识；

2.重点学会编程实现管道和重定向的功能；

3.实现自己的shell
## 三、设计思路和流程图
1.对输入的命令进行解析 实验内容主要是管道和重定向，这两个功能涉及shell“|”和“<”以及“>”等不同符号，所以要对输入的命令进行解析。初步按照空格来分，之后再按照<、>、|这些涉及管道和重定向的符号来分。

2.简单命令的执行 使用函数execvp可以实现简单的命令，这些命令暂时不涉及管道和重定向，函数原型为int execvp(const char *file ,char * const argv []);，execvp()会从PATH 环境变量所指的目录中查找符合参数file 的文件名，找到后便执行该文件，然后将第二个参数argv传给该欲执行的文件。为了不造成阻塞，这里启用了一个新线程实现它，最后父进程需等待子进程，以回收分配给它的资源。下面有些地方也用到这种方法。

3.输入输出重定向的实现 实现重定向的主要函数是freopen，FILE *freopen( const char *path, const char *mode, FILE *stream );path: 文件名，用于存储输入输出的自定义文件名。 mode: 文件打开的模式。和fopen中的模式（如r-只读, w-写）相同。 stream: 一个文件，通常使用标准流文件。函数实现重定向，把预定义的标准流文件定向到由path指定的文件中。要注意的是第二个参数，刚开始我是用的a+，结果每次输出都加到文件末尾。后来查了一下，改成w+可以先清空再写入文件。

4.管道功能的实现 命令之间通过|符号来分隔，使用pipe函数来建立管道。如何分隔这些命令呢？上面是写一个函数通过空格来分离每个字符串，这里通过strtok_r函数来分隔命令。利用pipe函数生成的的读取端和写入端，第一条命令的输出作为第二条命令的输入，从而实现管道的功能。四、主要数据结构及其说明
主要使用了数组和指针，存放相关的命令，通过字符串操作实现一些基本的逻辑。
## 五、源程序并附上注释（关键部分）
```
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <sys/stat.h>
#include <signal.h>
#include <fcntl.h>

#define hist_size 1024  // 历史记录容量（最多存 1024 条命令）
char *hist[hist_size]; 
int f = 0;              // 标记是否执行过 cd 命令
int head = 0, filled = 0; // 历史记录指针

/***********************
 * 1. parse: 解析用户输入
 * 功能：将用户输入的字符串按空格切分成命令和参数
 * 例如输入 "ls -l /home" -> argv = {"ls", "-l", "/home", NULL}
 ***********************/
void parse(char *word, char **argv)
{
    int count = 0;
    memset(argv, 0, sizeof(char*) * 64); // 初始化 argv 数组为 0

    char *lefts = NULL;
    const char *split = " ";  // 分隔符：空格

    while (1) {
        char *p = strtok_r(word, split, &lefts); // 逐个分词
        if (p == NULL) break;
        argv[count] = p;  // 存储分词结果
        word = lefts;     // 更新剩余字符串
        count++;
    }

    // 内置命令：exit
    if (strcmp(argv[0], "exit") == 0) {
        exit(0);
    }
    // 内置命令：cd
    else if (strcmp(argv[0], "cd") == 0) {
        int ch = chdir(argv[1]); // 切换目录
        f = 1;                   // 标记已执行 cd
    }
}

/***********************
 * 2. trim: 去掉字符串中的多余空格
 * 功能：返回去掉空格的新字符串
 ***********************/
char *trim(char *string)
{
    int i = 0, j = 0;
    char *ptr = malloc(sizeof(char*) * strlen(string));
    for (i = 0; string[i] != '\0'; i++) {
        if (string[i] != ' ') {
            ptr[j] = string[i];
            j++;
        }
    }
    ptr[j] = '\0';
    return ptr; // 返回新字符串（无空格）
}

/***********************
 * 3. execute: 执行普通命令
 * 使用 fork + execvp 执行命令
 ***********************/
void execute(char **argv)
{
    pid_t pid;
    int status;

    if ((pid = fork()) < 0) {  // 创建子进程
        printf("error: fork failed.\n");
        exit(1);
    }
    else if (pid == 0) { // 子进程执行命令
        if (execvp(argv[0], argv) < 0 && strcmp(argv[0], "cd")) {
            printf("error: invalid command.\n");
        }
        exit(0);
    }
    else { // 父进程等待子进程结束
        while (wait(&status) != pid);
    }
}

/***********************
 * 4. execute_file: 输出重定向 >
 * 功能：将命令的输出写入指定文件
 * 例如：ls > out.txt
 ***********************/
void execute_file(char **argv, char *output)
{
    pid_t pid;
    int status, flag;
    char *file = NULL;

    if ((pid = fork()) < 0) {
        printf("error: fork failed.\n");
        exit(1);
    }
    else if (pid == 0) {
        // 检测是否有 >
        if (strstr(output, ">") > 0) {
            char *p = strtok_r(output, ">", &file);
            output += 1;   
            file = trim(file);  // 获取文件名

            // 保存原 stdout
            int old_stdout = dup(1);
            // 将 stdout 重定向到文件
            FILE *fp1 = freopen(output, "w+", stdout);

            execute_file(argv, file);

            // 恢复 stdout
            fclose(stdout);
            FILE *fp2 = fdopen(old_stdout, "w");
            *stdout = *fp2;
            exit(0);
        }

        // 普通 >
        int old_stdout = dup(1);
        FILE *fp1 = freopen(output, "w+", stdout);
        if (execvp(argv[0], argv) < 0)
            printf("error: in exec");
        fclose(stdout);
        FILE *fp2 = fdopen(old_stdout, "w");
        *stdout = *fp2;
        exit(0);
    }
    else {
        while (wait(&status) != pid);
    }
}

/***********************
 * 5. execute_input: 输入重定向 <
 * 功能：将文件作为命令的输入
 * 例如：sort < in.txt
 ***********************/
void execute_input(char **argv, char *output)
{
    pid_t pid;
    int fd;
    char *file;
    int flag = 0;
    int status;

    if ((pid = fork()) < 0) {
        printf("error: fork failed\n");
        exit(1);
    }
    else if (pid == 0) {
        if (strstr(output, "<") > 0) {
            char *p = strtok_r(output, "<", &file);
            file = trim(file);
            fd = open(output, O_RDONLY);
            if (fd < 0) {
                printf("No such file or directory.");
                exit(0);
            }
            close(0);
            dup(fd);  // 重定向 stdin
        }
        if (execvp(argv[0], argv) < 0) {
            printf("error: in exec");
        }
        exit(0);
    }
    else {
        while (wait(&status) != pid);
    }
}

/***********************
 * 6. execute_pipe: 单管道 |
 * 功能：实现两个命令之间的管道
 * 例如：ls | grep txt
 ***********************/
void execute_pipe(char **argv, char *output)
{
    int pfds[2];
    pid_t pid, pid2;
    int status, status2;

    pipe(pfds); // 创建管道

    if ((pid = fork()) < 0) { printf("error: fork failed\n"); exit(1); }
    if ((pid2 = fork()) < 0) { printf("error: fork failed\n"); exit(1); }

    if (pid == 0 && pid2 != 0) { // 第一个子进程：命令1
        close(1);      // 关闭标准输出
        dup(pfds[1]);  // 将 stdout 重定向到管道写端
        close(pfds[0]); close(pfds[1]);
        execvp(argv[0], argv);
        exit(0);
    }
    else if (pid2 == 0 && pid != 0) { // 第二个子进程：命令2
        char *args[64];
        parse(output, args);   // 解析管道右边命令
        close(0);      // 关闭标准输入
        dup(pfds[0]);  // stdin <- 管道读端
        close(pfds[1]); close(pfds[0]);
        execvp(args[0], args);
        exit(0);
    }
    else { // 父进程等待两个子进程
        close(pfds[0]); close(pfds[1]);
        while (wait(&status) != pid);
        while (wait(&status2) != pid2);
    }
}

/***********************
 * 7. execute_pipe2: 双管道 ||
 * 功能：支持三个命令的管道
 * 例如：cmd1 | cmd2 | cmd3
 ***********************/
void execute_pipe2(char **argv, char **args, char **argp)
{
    int status;
    int i;
    int pipes[4];
    pipe(pipes);
    pipe(pipes + 2);

    if (fork() == 0) {
        dup2(pipes[1], 1); // cmd1 输出 -> pipes[1]
        close(pipes[0]); close(pipes[1]); close(pipes[2]); close(pipes[3]);
        execvp(argv[0], argv);
        exit(1);
    }
    else {
        if (fork() == 0) {
            dup2(pipes[0], 0); // cmd2 输入 <- pipes[0]
            dup2(pipes[3], 1); // cmd2 输出 -> pipes[3]
            close(pipes[0]); close(pipes[1]); close(pipes[2]); close(pipes[3]);
            execvp(args[0], args);
            exit(1);
        }
        else {
            if (fork() == 0) {
                dup2(pipes[2], 0); // cmd3 输入 <- pipes[2]
                close(pipes[0]); close(pipes[1]); close(pipes[2]); close(pipes[3]);
                execvp(argp[0], argp);
                exit(1);
            }
        }
    }
    close(pipes[0]); close(pipes[1]); close(pipes[2]); close(pipes[3]);
    for (i = 0; i < 3; i++) wait(&status);
}

/***********************
 * main: Shell 主循环
 * 功能：
 * - 显示提示符
 * - 读取用户输入
 * - 分析是否包含 > < |
 * - 调用相应的执行函数
 ***********************/
int main()
{
    char line[1024];
    char *argv[64];
    char *args[64];
    char *left;
    size_t size = 0;
    int count = 0;

    // 初始化历史记录
    for (int i = 0; i < hist_size; i++) {
        hist[i] = (char *)malloc(150);
    }

    while (1) {
        printf("SHELL~");
        char *dire[] = { "pwd" }; // 显示当前路径
        execute(dire);
        printf("$");

        char *word = NULL;
        int len = getline(&word, &size, stdin);
        if (*word == '\n') continue;
        word[len - 1] = '\0'; // 去掉换行符

        char *file = NULL;
        char *temp = (char *)malloc(150);
        strcpy(temp, word);
        parse(temp, argv);

        // 存入历史记录
        strcpy(hist[(head + 1) % hist_size], word);
        head = (head + 1) % hist_size;
        filled++;

        // 判断是否有重定向或管道
        int flag = 0;
        for (int i = 0; word[i] != '\0'; i++) {
            if (word[i] == '>') {
                strtok_r(word, ">", &file);
                file = trim(file);
                flag = 1; break;
            }
            else if (word[i] == '<') {
                strtok_r(word, "<", &file);
                file = trim(file);
                flag = 2; break;
            }
            else if (word[i] == '|') {
                strtok_r(word, "|", &left);
                flag = 3; break;
            }
        }

        if (strcmp(word, "exit") == 0) exit(0);

        if (flag == 1) {
            parse(word, argv);
            execute_file(argv, file);
        }
        else if (flag == 2) {
            parse(word, argv);
            execute_input(argv, file);
        }
        else if (flag == 3) {
            char *argp[64];
            if (strstr(left, "|") > 0) {
                char *file;
                strtok_r(left, "|", &file);
                parse(word, argv);
                parse(left, args);
                parse(file, argp);
                execute_pipe2(argv, args, argp);
            }
            else {
                parse(word, argv);
                execute_pipe(argv, left);
            }
        }
        else {
            parse(word, argv);
            execute(argv);
        }
    }
}
```

## 六、实验运行结果及分析
首先，编译myshell.c文件，然后运行，分别运行ls指令,pwd指令和ls -al|wc指令查看结果，然后将ls -al|wc写入a.txt，通过cat命令将a.txt的内容写入b.txt，随后运行ps aux | grep notification >a.txt命令改写a.txt内容，最后用cat命令查看a.txt和b.txt中的内容，结果如图所示：
 ![image](https://github.com/Jerry051212/seu-cyber/blob/main/OS/shell.png)
