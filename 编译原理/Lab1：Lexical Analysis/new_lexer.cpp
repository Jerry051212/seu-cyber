#include <iostream>
#include <string>
#include <cctype>
#include <cstdio>  // for printf, fprintf

std::string yytext;
size_t yypos = 0;
std::string yyinput;

int yylex();

int main() {
    std::string line;
    std::string fullInput;
    std::cout << "输入算术表达式（多行输入，以Ctrl+Z或空行结束）:\n";
    while (std::getline(std::cin, line)) {
        if (line.empty()) break;
        fullInput += line + "\n";
    }
    yyinput = fullInput;
    yypos = 0;
    yylex();
    return 0;
}

int yywrap() { return 1; }

int yylex() {
    while (yypos < yyinput.size()) {
        yytext.clear();
        size_t start = yypos;

        // 忽略空白（不包括换行）
        while (yypos < yyinput.size() && std::isspace(yyinput[yypos]) && yyinput[yypos] != '\n') ++yypos;

        if (yypos >= yyinput.size()) return 0;

        start = yypos;
        char c = yyinput[yypos];
        // 规则: [
        // 规则: [0-9]+(\\.[0-9]+)?
        if (std::isdigit(yyinput[yypos])) {
            start = yypos;
            bool hasDot = false;
            while (yypos < yyinput.size() && (std::isdigit(yyinput[yypos]) || (yyinput[yypos] == '.' && !hasDot))) {
                if (yyinput[yypos] == '.') hasDot = true;
                ++yypos;
            }
            yytext = yyinput.substr(start, yypos - start);
            { printf("NUM (%s)\n", yytext.c_str()); }
            continue;
        }
        // 规则: \"sin\"
        if (yypos + 3 <= yyinput.size() && yyinput.substr(yypos, 3) == "sin") {
            yytext = "sin";
            yypos += 3;
            { printf("SIN (sin)\n"); }
            continue;
        }
        // 规则: \"cos\"
        if (yypos + 3 <= yyinput.size() && yyinput.substr(yypos, 3) == "cos") {
            yytext = "cos";
            yypos += 3;
            { printf("COS (cos)\n"); }
            continue;
        }
        // 规则: \"lg\"
        if (yypos + 2 <= yyinput.size() && yyinput.substr(yypos, 2) == "lg") {
            yytext = "lg";
            yypos += 2;
            { printf("LG (lg)\n"); }
            continue;
        }
        // 规则: \"ln\"
        if (yypos + 2 <= yyinput.size() && yyinput.substr(yypos, 2) == "ln") {
            yytext = "ln";
            yypos += 2;
            { printf("LN (ln)\n"); }
            continue;
        }
        // 规则: \"+\"
        if (yypos + 1 <= yyinput.size() && yyinput.substr(yypos, 1) == "+") {
            yytext = "+";
            yypos += 1;
            { printf("PLUS (+)\n"); }
            continue;
        }
        // 规则: \"-\"
        if (yypos + 1 <= yyinput.size() && yyinput.substr(yypos, 1) == "-") {
            yytext = "-";
            yypos += 1;
            { printf("MINUS (-)\n"); }
            continue;
        }
        // 规则: \"*\"
        if (yypos + 1 <= yyinput.size() && yyinput.substr(yypos, 1) == "*") {
            yytext = "*";
            yypos += 1;
            { printf("STAR (*)\n"); }
            continue;
        }
        // 规则: \"/\"
        if (yypos + 1 <= yyinput.size() && yyinput.substr(yypos, 1) == "/") {
            yytext = "/";
            yypos += 1;
            { printf("SLASH (/)\n"); }
            continue;
        }
        // 规则: \"(\"
        if (yypos + 1 <= yyinput.size() && yyinput.substr(yypos, 1) == "(") {
            yytext = "(";
            yypos += 1;
            { printf("LPAREN (()\n"); }
            continue;
        }
        // 规则: \")\"
        if (yypos + 1 <= yyinput.size() && yyinput.substr(yypos, 1) == ")") {
            yytext = ")";
            yypos += 1;
            { printf("RPAREN ())\n"); }
            continue;
        }
        // 规则: \\n
        if (yyinput[yypos] == '\n') {
            yytext = "\n";
            ++yypos;
            { printf("EOL\n"); }
            continue;
        }
        // 规则: .
        else {
            yytext = yyinput.substr(yypos, 1);
            ++yypos;
            { fprintf(stderr, "非法字符: %s\n", yytext.c_str()); }
            continue;
        }
        // 默认错误
        std::cerr << "未匹配字符: " << yyinput[yypos] << std::endl;
        ++yypos;
    }
    return 0;
}


