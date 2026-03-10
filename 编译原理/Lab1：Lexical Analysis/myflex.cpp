#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cctype>
#include <map>

// 规则结构
struct Rule {
    std::string pattern;
    std::string action;
    bool isIgnore = false;
};

std::vector<Rule> rules;
std::string userCode;  // 用户提供的额外代码，如 main

// 转义字符串
std::string escapeString(const std::string& s) {
    std::string res;
    for (char c : s) {
        if (c == '"') res += "\\\"";
        else if (c == '\\') res += "\\\\";
        else res += c;
    }
    return res;
}

// 替换 action 中的 yytext 为 yytext.c_str()
std::string replaceYytext(const std::string& action) {
    std::string res = action;
    size_t pos = 0;
    while ((pos = res.find("yytext", pos)) != std::string::npos) {
        res.insert(pos + 6, ".c_str()");
        pos += 14;  // 跳过插入部分
    }
    return res;
}

// 生成 lexer.cpp 代码
void generateLexer() {
    std::cout << R"(#include <iostream>
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
)";

    // 生成规则匹配
    for (const auto& rule : rules) {
        if (rule.isIgnore) continue;

        std::string act = replaceYytext(rule.action);

        std::cout << "        // 规则: " << escapeString(rule.pattern) << "\n";

        if (rule.pattern == "\\n") {
            std::cout << "        if (yyinput[yypos] == '\\n') {\n";
            std::cout << "            yytext = \"\\n\";\n";
            std::cout << "            ++yypos;\n";
            std::cout << "            " << act << "\n";
            std::cout << "            continue;\n";
            std::cout << "        }\n";
        }
        else if (rule.pattern == ".") {
            std::cout << "        else {\n";
            std::cout << "            yytext = yyinput.substr(yypos, 1);\n";
            std::cout << "            ++yypos;\n";
            std::cout << "            " << act << "\n";
            std::cout << "            continue;\n";
            std::cout << "        }\n";
        }
        else if (rule.pattern[0] == '"' && rule.pattern.back() == '"') {
            std::string pat = rule.pattern.substr(1, rule.pattern.size() - 2);
            std::cout << "        if (yypos + " << pat.size() << " <= yyinput.size() && "
                << "yyinput.substr(yypos, " << pat.size() << ") == \"" << escapeString(pat) << "\") {\n";
            std::cout << "            yytext = \"" << escapeString(pat) << "\";\n";
            std::cout << "            yypos += " << pat.size() << ";\n";
            std::cout << "            " << act << "\n";
            std::cout << "            continue;\n";
            std::cout << "        }\n";
        }
        else if (rule.pattern.find("[0-9]") != std::string::npos) {
            // 硬编码数字匹配 [0-9]+(\.[0-9]+)?
            std::cout << "        if (std::isdigit(yyinput[yypos])) {\n";
            std::cout << "            start = yypos;\n";
            std::cout << "            bool hasDot = false;\n";
            std::cout << "            while (yypos < yyinput.size() && (std::isdigit(yyinput[yypos]) || (yyinput[yypos] == '.' && !hasDot))) {\n";
            std::cout << "                if (yyinput[yypos] == '.') hasDot = true;\n";
            std::cout << "                ++yypos;\n";
            std::cout << "            }\n";
            std::cout << "            yytext = yyinput.substr(start, yypos - start);\n";
            std::cout << "            " << act << "\n";
            std::cout << "            continue;\n";
            std::cout << "        }\n";
        }
    }

    std::cout << R"(        // 默认错误
        std::cerr << "未匹配字符: " << yyinput[yypos] << std::endl;
        ++yypos;
    }
    return 0;
}
)";
}

// 解析 .l 文件
void parseLFile() {
    std::string line;
    bool inRules = false;
    bool inUserCode = false;

    while (std::getline(std::cin, line)) {
        if (line == "%%") {
            if (!inRules) {
                inRules = true;
            }
            else {
                inRules = false;
                inUserCode = true;
            }
            continue;
        }

        if (inUserCode) {
            userCode += line + "\n";
            continue;
        }

        if (!inRules) continue;

        // 解析规则行
        std::istringstream iss(line);
        std::string pattern, actionStr;
        iss >> pattern;
        std::getline(iss, actionStr);  // 获取剩余作为 action

        // 去掉前导空白
        size_t start = actionStr.find_first_not_of(" \t");
        if (start != std::string::npos) {
            actionStr = actionStr.substr(start);
        }
        else {
            actionStr = "";
        }

        bool ignore = false;
        if (pattern == "[ \t]+" || actionStr.find("/* ignore */") != std::string::npos || actionStr.empty()) {
            ignore = true;
        }

        rules.push_back({ pattern, actionStr, ignore });
    }
}

int main() {
    std::cout << "=== 我的 Lex 生成器 ===\n";
    std::cout << "请粘贴 .l 文件内容，以 %% 分隔区段，最后以 Ctrl+Z 结束\n\n";

    parseLFile();

    std::cout << "\n生成 lexer.cpp ...\n\n";
    generateLexer();

    // 添加用户代码，如果有
    if (!userCode.empty()) {
        std::cout << "\n// 用户代码\n" << userCode;
    }

    std::cout << "\n生成完成！保存为 lexer.cpp 并编译 g++ lexer.cpp -o lexer.exe\n";

    return 0;
}