/*#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <map>
#include <set>
#include <sstream>

// Token types based on the provided .l file
enum TokenType {
    TOK_NUM, TOK_SIN, TOK_COS, TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
    TOK_LPAREN, TOK_RPAREN, TOK_EOL, TOK_ILLEGAL, TOK_EOF
};

struct Token {
    TokenType type;
    std::string value;  // for NUM
};

std::string tokenToString(TokenType t) {
    switch (t) {
    case TOK_NUM: return "NUM";
    case TOK_SIN: return "SIN";
    case TOK_COS: return "COS";
    case TOK_PLUS: return "+";
    case TOK_MINUS: return "-";
    case TOK_STAR: return "*";
    case TOK_SLASH: return "/";
    case TOK_LPAREN: return "(";
    case TOK_RPAREN: return ")";
    case TOK_EOF: return "$";
    default: return "UNKNOWN";
    }
}

class Lexer {
private:
    std::string input;
    size_t pos = 0;

    void skipWhitespace() {
        while (pos < input.size() && std::isspace(input[pos])) ++pos;
    }

public:
    Lexer(const std::string& s) : input(s) {}

    Token nextToken() {
        skipWhitespace();
        if (pos >= input.size()) return { TOK_EOF, "$" };

        char c = input[pos];
        if (isdigit(c) || (c == '.' && pos + 1 < input.size() && isdigit(input[pos + 1]))) {
            size_t start = pos;
            while (pos < input.size() && (isdigit(input[pos]) || input[pos] == '.')) ++pos;
            return { TOK_NUM, input.substr(start, pos - start) };
        }

        ++pos;
        switch (c) {
        case '+': return { TOK_PLUS, "+" };
        case '-': return { TOK_MINUS, "-" };
        case '*': return { TOK_STAR, "*" };
        case '/': return { TOK_SLASH, "/" };
        case '(': return { TOK_LPAREN, "(" };
        case ')': return { TOK_RPAREN, ")" };
        default:
            if (isalpha(c)) {
                size_t start = pos - 1;
                while (pos < input.size() && isalpha(input[pos])) ++pos;
                std::string id = input.substr(start, pos - start);
                if (id == "sin") return { TOK_SIN, "sin" };
                if (id == "cos") return { TOK_COS, "cos" };
                return { TOK_ILLEGAL, id };
            }
            return { TOK_ILLEGAL, std::string(1, c) };
        }
    }
};

// Grammar symbols
const int START = 0;  // S
const int E = 1;
const int T = 2;
const int F = 3;
const int A = 4;

// Terminals
const int NUM = -1;
const int SIN = -2;
const int COS = -3;
const int PLUS = -4;
const int MINUS = -5;
const int STAR = -6;
const int SLASH = -7;
const int LPAREN = -8;
const int RPAREN = -9;
const int END = -10;  // $

struct Action {
    char type;  // 's' shift, 'r' reduce, 'a' accept, 'e' error
    int value;  // state or production number
};

class LR1Parser {
private:
    Lexer lexer;
    Token current;
    std::vector<int> stack;
    std::vector<Action> actionTable;
    std::vector<int> gotoTable;

    // Productions (lhs, length, prod index matches python)
    std::vector<std::pair<int, int>> prods = {
        {START, 1},  // 0: S -> E
        {E, 3},      // 1: E -> E + T
        {E, 3},      // 2: E -> E - T
        {E, 1},      // 3: E -> T
        {T, 3},      // 4: T -> T * F
        {T, 3},      // 5: T -> T / F
        {T, 1},      // 6: T -> F
        {F, 1},      // 7: F -> A
        {A, 1},      // 8: A -> NUM
        {A, 3},      // 9: A -> ( E )
        {A, 4},      // 10: A -> sin ( E )
        {A, 4}       // 11: A -> cos ( E )
    };

    int termIndex(TokenType tt) {
        switch (tt) {
        case TOK_NUM: return 0;
        case TOK_SIN: return 1;
        case TOK_COS: return 2;
        case TOK_PLUS: return 3;
        case TOK_MINUS: return 4;
        case TOK_STAR: return 5;
        case TOK_SLASH: return 6;
        case TOK_LPAREN: return 7;
        case TOK_RPAREN: return 8;
        case TOK_EOF: return 9;
        default: return -1;
        }
    }

    void setAction(int state, int t_idx, char typ, int val) {
        actionTable[state * 10 + t_idx] = { typ, val };
    }

    void setGoto(int state, int nt_idx, int val) {  // nt_idx 0=E,1=T,2=F,3=A
        gotoTable[state * 4 + nt_idx] = val;
    }

    void initTables() {
        const int num_states = 48;
        actionTable.assign(num_states * 10, { 'e', 0 });
        gotoTable.assign(num_states * 4, -1);

        // State 0
        setAction(0, 7, 's', 8);  // (
        setAction(0, 2, 's', 7);  // COS
        setAction(0, 0, 's', 5);  // NUM
        setAction(0, 1, 's', 6);  // SIN
        setGoto(0, 3, 4);  // A
        setGoto(0, 0, 1);  // E
        setGoto(0, 2, 3);  // F
        setGoto(0, 1, 2);  // T

        // State 1
        setAction(1, 9, 'a', 0);  // $
        setAction(1, 3, 's', 9);  // +
        setAction(1, 4, 's', 10);  // -

        // State 2
        setAction(2, 9, 'r', 3);  // $
        setAction(2, 5, 's', 11);  // *
        setAction(2, 3, 'r', 3);  // +
        setAction(2, 4, 'r', 3);  // -
        setAction(2, 6, 's', 12);  // /

        // State 3
        setAction(3, 9, 'r', 6);  // $
        setAction(3, 5, 'r', 6);  // *
        setAction(3, 3, 'r', 6);  // +
        setAction(3, 4, 'r', 6);  // -
        setAction(3, 6, 'r', 6);  // /

        // State 4
        setAction(4, 9, 'r', 7);  // $
        setAction(4, 5, 'r', 7);  // *
        setAction(4, 3, 'r', 7);  // +
        setAction(4, 4, 'r', 7);  // -
        setAction(4, 6, 'r', 7);  // /

        // State 5
        setAction(5, 9, 'r', 8);  // $
        setAction(5, 5, 'r', 8);  // *
        setAction(5, 3, 'r', 8);  // +
        setAction(5, 4, 'r', 8);  // -
        setAction(5, 6, 'r', 8);  // /

        // State 6
        setAction(6, 7, 's', 13);  // (

        // State 7
        setAction(7, 7, 's', 14);  // (

        // State 8
        setAction(8, 7, 's', 22);  // (
        setAction(8, 2, 's', 21);  // COS
        setAction(8, 0, 's', 19);  // NUM
        setAction(8, 1, 's', 20);  // SIN
        setGoto(8, 3, 18);  // A
        setGoto(8, 0, 15);  // E
        setGoto(8, 2, 17);  // F
        setGoto(8, 1, 16);  // T

        // State 9
        setAction(9, 7, 's', 8);  // (
        setAction(9, 2, 's', 7);  // COS
        setAction(9, 0, 's', 5);  // NUM
        setAction(9, 1, 's', 6);  // SIN
        setGoto(9, 3, 4);  // A
        setGoto(9, 2, 3);  // F
        setGoto(9, 1, 23);  // T

        // State 10
        setAction(10, 7, 's', 8);  // (
        setAction(10, 2, 's', 7);  // COS
        setAction(10, 0, 's', 5);  // NUM
        setAction(10, 1, 's', 6);  // SIN
        setGoto(10, 3, 4);  // A
        setGoto(10, 2, 3);  // F
        setGoto(10, 1, 24);  // T

        // State 11
        setAction(11, 7, 's', 8);  // (
        setAction(11, 2, 's', 7);  // COS
        setAction(11, 0, 's', 5);  // NUM
        setAction(11, 1, 's', 6);  // SIN
        setGoto(11, 3, 4);  // A
        setGoto(11, 2, 25);  // F

        // State 12
        setAction(12, 7, 's', 8);  // (
        setAction(12, 2, 's', 7);  // COS
        setAction(12, 0, 's', 5);  // NUM
        setAction(12, 1, 's', 6);  // SIN
        setGoto(12, 3, 4);  // A
        setGoto(12, 2, 26);  // F

        // State 13
        setAction(13, 7, 's', 22);  // (
        setAction(13, 2, 's', 21);  // COS
        setAction(13, 0, 's', 19);  // NUM
        setAction(13, 1, 's', 20);  // SIN
        setGoto(13, 3, 18);  // A
        setGoto(13, 0, 27);  // E
        setGoto(13, 2, 17);  // F
        setGoto(13, 1, 16);  // T

        // State 14
        setAction(14, 7, 's', 22);  // (
        setAction(14, 2, 's', 21);  // COS
        setAction(14, 0, 's', 19);  // NUM
        setAction(14, 1, 's', 20);  // SIN
        setGoto(14, 3, 18);  // A
        setGoto(14, 0, 28);  // E
        setGoto(14, 2, 17);  // F
        setGoto(14, 1, 16);  // T

        // State 15
        setAction(15, 8, 's', 31);  // )
        setAction(15, 3, 's', 29);  // +
        setAction(15, 4, 's', 30);  // -

        // State 16
        setAction(16, 8, 'r', 3);  // )
        setAction(16, 5, 's', 32);  // *
        setAction(16, 3, 'r', 3);  // +
        setAction(16, 4, 'r', 3);  // -
        setAction(16, 6, 's', 33);  // /

        // State 17
        setAction(17, 8, 'r', 6);  // )
        setAction(17, 5, 'r', 6);  // *
        setAction(17, 3, 'r', 6);  // +
        setAction(17, 4, 'r', 6);  // -
        setAction(17, 6, 'r', 6);  // /

        // State 18
        setAction(18, 8, 'r', 7);  // )
        setAction(18, 5, 'r', 7);  // *
        setAction(18, 3, 'r', 7);  // +
        setAction(18, 4, 'r', 7);  // -
        setAction(18, 6, 'r', 7);  // /

        // State 19
        setAction(19, 8, 'r', 8);  // )
        setAction(19, 5, 'r', 8);  // *
        setAction(19, 3, 'r', 8);  // +
        setAction(19, 4, 'r', 8);  // -
        setAction(19, 6, 'r', 8);  // /

        // State 20
        setAction(20, 7, 's', 34);  // (

        // State 21
        setAction(21, 7, 's', 35);  // (

        // State 22
        setAction(22, 7, 's', 22);  // (
        setAction(22, 2, 's', 21);  // COS
        setAction(22, 0, 's', 19);  // NUM
        setAction(22, 1, 's', 20);  // SIN
        setGoto(22, 3, 18);  // A
        setGoto(22, 0, 36);  // E
        setGoto(22, 2, 17);  // F
        setGoto(22, 1, 16);  // T

        // State 23
        setAction(23, 9, 'r', 1);  // $
        setAction(23, 5, 's', 11);  // *
        setAction(23, 3, 'r', 1);  // +
        setAction(23, 4, 'r', 1);  // -
        setAction(23, 6, 's', 12);  // /

        // State 24
        setAction(24, 9, 'r', 2);  // $
        setAction(24, 5, 's', 11);  // *
        setAction(24, 3, 'r', 2);  // +
        setAction(24, 4, 'r', 2);  // -
        setAction(24, 6, 's', 12);  // /

        // State 25
        setAction(25, 9, 'r', 4);  // $
        setAction(25, 5, 'r', 4);  // *
        setAction(25, 3, 'r', 4);  // +
        setAction(25, 4, 'r', 4);  // -
        setAction(25, 6, 'r', 4);  // /

        // State 26
        setAction(26, 9, 'r', 5);  // $
        setAction(26, 5, 'r', 5);  // *
        setAction(26, 3, 'r', 5);  // +
        setAction(26, 4, 'r', 5);  // -
        setAction(26, 6, 'r', 5);  // /

        // State 27
        setAction(27, 8, 's', 37);  // )
        setAction(27, 3, 's', 29);  // +
        setAction(27, 4, 's', 30);  // -

        // State 28
        setAction(28, 8, 's', 38);  // )
        setAction(28, 3, 's', 29);  // +
        setAction(28, 4, 's', 30);  // -

        // State 29
        setAction(29, 7, 's', 22);  // (
        setAction(29, 2, 's', 21);  // COS
        setAction(29, 0, 's', 19);  // NUM
        setAction(29, 1, 's', 20);  // SIN
        setGoto(29, 3, 18);  // A
        setGoto(29, 2, 17);  // F
        setGoto(29, 1, 39);  // T

        // State 30
        setAction(30, 7, 's', 22);  // (
        setAction(30, 2, 's', 21);  // COS
        setAction(30, 0, 's', 19);  // NUM
        setAction(30, 1, 's', 20);  // SIN
        setGoto(30, 3, 18);  // A
        setGoto(30, 2, 17);  // F
        setGoto(30, 1, 40);  // T

        // State 31
        setAction(31, 9, 'r', 9);  // $
        setAction(31, 5, 'r', 9);  // *
        setAction(31, 3, 'r', 9);  // +
        setAction(31, 4, 'r', 9);  // -
        setAction(31, 6, 'r', 9);  // /

        // State 32
        setAction(32, 7, 's', 22);  // (
        setAction(32, 2, 's', 21);  // COS
        setAction(32, 0, 's', 19);  // NUM
        setAction(32, 1, 's', 20);  // SIN
        setGoto(32, 3, 18);  // A
        setGoto(32, 2, 41);  // F

        // State 33
        setAction(33, 7, 's', 22);  // (
        setAction(33, 2, 's', 21);  // COS
        setAction(33, 0, 's', 19);  // NUM
        setAction(33, 1, 's', 20);  // SIN
        setGoto(33, 3, 18);  // A
        setGoto(33, 2, 42);  // F

        // State 34
        setAction(34, 7, 's', 22);  // (
        setAction(34, 2, 's', 21);  // COS
        setAction(34, 0, 's', 19);  // NUM
        setAction(34, 1, 's', 20);  // SIN
        setGoto(34, 3, 18);  // A
        setGoto(34, 0, 43);  // E
        setGoto(34, 2, 17);  // F
        setGoto(34, 1, 16);  // T

        // State 35
        setAction(35, 7, 's', 22);  // (
        setAction(35, 2, 's', 21);  // COS
        setAction(35, 0, 's', 19);  // NUM
        setAction(35, 1, 's', 20);  // SIN
        setGoto(35, 3, 18);  // A
        setGoto(35, 0, 44);  // E
        setGoto(35, 2, 17);  // F
        setGoto(35, 1, 16);  // T

        // State 36
        setAction(36, 8, 's', 45);  // )
        setAction(36, 3, 's', 29);  // +
        setAction(36, 4, 's', 30);  // -

        // State 37
        setAction(37, 9, 'r', 10);  // $
        setAction(37, 5, 'r', 10);  // *
        setAction(37, 3, 'r', 10);  // +
        setAction(37, 4, 'r', 10);  // -
        setAction(37, 6, 'r', 10);  // /

        // State 38
        setAction(38, 9, 'r', 11);  // $
        setAction(38, 5, 'r', 11);  // *
        setAction(38, 3, 'r', 11);  // +
        setAction(38, 4, 'r', 11);  // -
        setAction(38, 6, 'r', 11);  // /

        // State 39
        setAction(39, 8, 'r', 1);  // )
        setAction(39, 5, 's', 32);  // *
        setAction(39, 3, 'r', 1);  // +
        setAction(39, 4, 'r', 1);  // -
        setAction(39, 6, 's', 33);  // /

        // State 40
        setAction(40, 8, 'r', 2);  // )
        setAction(40, 5, 's', 32);  // *
        setAction(40, 3, 'r', 2);  // +
        setAction(40, 4, 'r', 2);  // -
        setAction(40, 6, 's', 33);  // /

        // State 41
        setAction(41, 8, 'r', 4);  // )
        setAction(41, 5, 'r', 4);  // *
        setAction(41, 3, 'r', 4);  // +
        setAction(41, 4, 'r', 4);  // -
        setAction(41, 6, 'r', 4);  // /

        // State 42
        setAction(42, 8, 'r', 5);  // )
        setAction(42, 5, 'r', 5);  // *
        setAction(42, 3, 'r', 5);  // +
        setAction(42, 4, 'r', 5);  // -
        setAction(42, 6, 'r', 5);  // /

        // State 43
        setAction(43, 8, 's', 46);  // )
        setAction(43, 3, 's', 29);  // +
        setAction(43, 4, 's', 30);  // -

        // State 44
        setAction(44, 8, 's', 47);  // )
        setAction(44, 3, 's', 29);  // +
        setAction(44, 4, 's', 30);  // -

        // State 45
        setAction(45, 8, 'r', 9);  // )
        setAction(45, 5, 'r', 9);  // *
        setAction(45, 3, 'r', 9);  // +
        setAction(45, 4, 'r', 9);  // -
        setAction(45, 6, 'r', 9);  // /

        // State 46
        setAction(46, 8, 'r', 10);  // )
        setAction(46, 5, 'r', 10);  // *
        setAction(46, 3, 'r', 10);  // +
        setAction(46, 4, 'r', 10);  // -
        setAction(46, 6, 'r', 10);  // /

        // State 47
        setAction(47, 8, 'r', 11);  // )
        setAction(47, 5, 'r', 11);  // *
        setAction(47, 3, 'r', 11);  // +
        setAction(47, 4, 'r', 11);  // -
        setAction(47, 6, 'r', 11);  // /
    }

public:
    LR1Parser(const std::string& input) : lexer(input) {
        initTables();
        stack.push_back(0);
        current = lexer.nextToken();
    }

    void printStack() {
        std::cout << "Stack: ";
        for (auto s : stack) std::cout << s << " ";
        std::cout << std::endl;
    }

    void printReduction(int prod) {
        std::cout << "使用产生式 " << prod << " 归约" << ": ";
        int lhs = prods[prod].first;
        std::string lhs_str = (lhs == 0 ? "S" : lhs == 1 ? "E" : lhs == 2 ? "T" : lhs == 3 ? "F" : "A");
        std::cout << lhs_str << " -> ";
        // For simplicity, print the prod description
        switch (prod) {
        case 0: std::cout << "E"; break;
        case 1: std::cout << "E + T"; break;
        case 2: std::cout << "E - T"; break;
        case 3: std::cout << "T"; break;
        case 4: std::cout << "T * F"; break;
        case 5: std::cout << "T / F"; break;
        case 6: std::cout << "F"; break;
        case 7: std::cout << "A"; break;
        case 8: std::cout << "NUM"; break;
        case 9: std::cout << "( E )"; break;
        case 10: std::cout << "sin ( E )"; break;
        case 11: std::cout << "cos ( E )"; break;
        }
        std::cout << std::endl;
    }

    bool parse() {
        while (true) {
            printStack();
            std::cout << "当前token: " /* << tokenToString(current.type) << " " << current.value << std::endl;

            int state = stack.back();
            int t_idx = termIndex(current.type);
            if (t_idx == -1) {
                std::cout << "非法token" << std::endl;
                return false;
            }

            Action act = actionTable[state * 10 + t_idx];

            if (act.type == 's') {
                std::cout << "Shift 到状态 " << act.value << std::endl;
                stack.push_back(act.value);
                current = lexer.nextToken();
            }
            else if (act.type == 'r') {
                printReduction(act.value);
                int len = prods[act.value].second;
                for (int i = 0; i < len; ++i) stack.pop_back();
                int top = stack.back();
                int lhs = prods[act.value].first;
                int nt_idx = lhs - 1;  // START=0 -> -1? Wait, START=0 not pushed, but reduces to E etc.
                // START=0, but reduce 0 only at end.
                if (lhs == 0) nt_idx = -1; // not happen until accept
                else nt_idx = lhs - 1; // E1->0, T2->1 etc
                int newState = gotoTable[top * 4 + nt_idx];
                if (newState == -1) {
                    std::cout << "Goto error" << std::endl;
                    return false;
                }
                std::cout << "Goto 状态 " << newState << std::endl;
                stack.push_back(newState);
            }
            else if (act.type == 'a') {
                std::cout << "Accept" << std::endl;
                return true;
            }
            else {
                std::cout << "Syntax error" << std::endl;
                return false;
            }
        }
    }
};

int main() {
    std::string input = "(sin(3 + 4) * 2) * cos(2)";
    LR1Parser parser(input);
    parser.parse();
    return 0;
}*/

#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <map>
#include <set>
#include <sstream>

// Token types based on the provided .l file
enum TokenType {
    TOK_NUM, TOK_SIN, TOK_COS, TOK_LG, TOK_LN, TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_SLASH,
    TOK_LPAREN, TOK_RPAREN, TOK_EOL, TOK_ILLEGAL, TOK_EOF
};

struct Token {
    TokenType type;
    std::string value;  // for NUM
};

std::string tokenToString(TokenType t) {
    switch (t) {
    case TOK_NUM: return "NUM";
    case TOK_SIN: return "SIN";
    case TOK_COS: return "COS";
    case TOK_LG: return "LG";
    case TOK_LN: return "LN";
    case TOK_PLUS: return "+";
    case TOK_MINUS: return "-";
    case TOK_STAR: return "*";
    case TOK_SLASH: return "/";
    case TOK_LPAREN: return "(";
    case TOK_RPAREN: return ")";
    case TOK_EOF: return "$";
    default: return "UNKNOWN";
    }
}

class Lexer {
private:
    std::string input;
    size_t pos = 0;

    void skipWhitespace() {
        while (pos < input.size() && std::isspace(input[pos])) ++pos;
    }

public:
    Lexer(const std::string& s) : input(s) {}

    Token nextToken() {
        skipWhitespace();
        if (pos >= input.size()) return { TOK_EOF, "$" };

        char c = input[pos];
        if (isdigit(c) || (c == '.' && pos + 1 < input.size() && isdigit(input[pos + 1]))) {
            size_t start = pos;
            while (pos < input.size() && (isdigit(input[pos]) || input[pos] == '.')) ++pos;
            return { TOK_NUM, input.substr(start, pos - start) };
        }

        ++pos;
        switch (c) {
        case '+': return { TOK_PLUS, "+" };
        case '-': return { TOK_MINUS, "-" };
        case '*': return { TOK_STAR, "*" };
        case '/': return { TOK_SLASH, "/" };
        case '(': return { TOK_LPAREN, "(" };
        case ')': return { TOK_RPAREN, ")" };
        default:
            if (isalpha(c)) {
                size_t start = pos - 1;
                while (pos < input.size() && isalpha(input[pos])) ++pos;
                std::string id = input.substr(start, pos - start);
                if (id == "sin") return { TOK_SIN, "sin" };
                if (id == "cos") return { TOK_COS, "cos" };
                if (id == "lg") return { TOK_LG, "lg" };
                if (id == "ln") return { TOK_LN, "ln" };
                return { TOK_ILLEGAL, id };
            }
            return { TOK_ILLEGAL, std::string(1, c) };
        }
    }
};

// Grammar symbols
const int START = 0;  // S
const int E = 1;
const int T = 2;
const int F = 3;
const int A = 4;

// Terminals
const int NUM = -1;
const int SIN = -2;
const int COS = -3;
const int LG = -11;
const int LN = -12;
const int PLUS = -4;
const int MINUS = -5;
const int STAR = -6;
const int SLASH = -7;
const int LPAREN = -8;
const int RPAREN = -9;
const int END = -10;  // $

struct Action {
    char type;  // 's' shift, 'r' reduce, 'a' accept, 'e' error
    int value;  // state or production number
};

class LR1Parser {
private:
    Lexer lexer;
    Token current;
    std::vector<int> stack;
    std::vector<Action> actionTable;
    std::vector<int> gotoTable;

    // Productions (lhs, length)
    std::vector<std::pair<int, int>> prods = {
        {START, 1},  // 0: S -> E
        {E, 3},      // 1: E -> E + T
        {E, 3},      // 2: E -> E - T
        {E, 1},      // 3: E -> T
        {T, 3},      // 4: T -> T * F
        {T, 3},      // 5: T -> T / F
        {T, 1},      // 6: T -> F
        {F, 1},      // 7: F -> A
        {A, 1},      // 8: A -> NUM
        {A, 3},      // 9: A -> ( E )
        {A, 4},      // 10: A -> sin ( E )
        {A, 4},      // 11: A -> cos ( E )
        {A, 4},      // 12: A -> lg ( E )
        {A, 4}       // 13: A -> ln ( E )
    };

    int termIndex(TokenType tt) {
        switch (tt) {
        case TOK_LN: return 0;
        case TOK_LG: return 1;
        case TOK_EOF: return 2;
        case TOK_RPAREN: return 3;
        case TOK_LPAREN: return 4;
        case TOK_SLASH: return 5;
        case TOK_STAR: return 6;
        case TOK_MINUS: return 7;
        case TOK_PLUS: return 8;
        case TOK_COS: return 9;
        case TOK_SIN: return 10;
        case TOK_NUM: return 11;
        default: return -1;
        }
    }

    void setAction(int state, int t_idx, char typ, int val) {
        actionTable[state * 12 + t_idx] = { typ, val };
    }

    void setGoto(int state, int nt_idx, int val) {  // nt_idx 0=E,1=T,2=F,3=A
        gotoTable[state * 4 + nt_idx] = val;
    }

    void initTables() {
        const int num_states = 64;
        actionTable.assign(num_states * 12, { 'e', 0 });
        gotoTable.assign(num_states * 4, -1);
        // State 0
        setAction(0, 11, 's', 3);
        setAction(0, 4, 's', 4);
        setAction(0, 10, 's', 6);
        setAction(0, 1, 's', 7);
        setAction(0, 9, 's', 8);
        setAction(0, 0, 's', 9);
        setGoto(0, 1, 1);
        setGoto(0, 0, 2);
        setGoto(0, 3, 5);
        setGoto(0, 2, 10);
        // State 1
        setAction(1, 7, 'r', 3);
        setAction(1, 2, 'r', 3);
        setAction(1, 8, 'r', 3);
        setAction(1, 5, 's', 11);
        setAction(1, 6, 's', 12);
        // State 2
        setAction(2, 2, 'a', 0);
        setAction(2, 8, 's', 13);
        setAction(2, 7, 's', 14);
        // State 3
        setAction(3, 2, 'r', 8);
        setAction(3, 8, 'r', 8);
        setAction(3, 5, 'r', 8);
        setAction(3, 7, 'r', 8);
        setAction(3, 6, 'r', 8);
        // State 4
        setAction(4, 11, 's', 18);
        setAction(4, 4, 's', 20);
        setAction(4, 10, 's', 21);
        setAction(4, 1, 's', 22);
        setAction(4, 9, 's', 23);
        setAction(4, 0, 's', 24);
        setGoto(4, 1, 15);
        setGoto(4, 0, 16);
        setGoto(4, 2, 17);
        setGoto(4, 3, 19);
        // State 5
        setAction(5, 7, 'r', 7);
        setAction(5, 6, 'r', 7);
        setAction(5, 2, 'r', 7);
        setAction(5, 5, 'r', 7);
        setAction(5, 8, 'r', 7);
        // State 6
        setAction(6, 4, 's', 25);
        // State 7
        setAction(7, 4, 's', 26);
        // State 8
        setAction(8, 4, 's', 27);
        // State 9
        setAction(9, 4, 's', 28);
        // State 10
        setAction(10, 5, 'r', 6);
        setAction(10, 8, 'r', 6);
        setAction(10, 7, 'r', 6);
        setAction(10, 6, 'r', 6);
        setAction(10, 2, 'r', 6);
        // State 11
        setAction(11, 11, 's', 3);
        setAction(11, 4, 's', 4);
        setAction(11, 10, 's', 6);
        setAction(11, 1, 's', 7);
        setAction(11, 9, 's', 8);
        setAction(11, 0, 's', 9);
        setGoto(11, 2, 29);
        setGoto(11, 3, 5);
        // State 12
        setAction(12, 11, 's', 3);
        setAction(12, 4, 's', 4);
        setAction(12, 10, 's', 6);
        setAction(12, 1, 's', 7);
        setAction(12, 9, 's', 8);
        setAction(12, 0, 's', 9);
        setGoto(12, 2, 30);
        setGoto(12, 3, 5);
        // State 13
        setAction(13, 11, 's', 3);
        setAction(13, 4, 's', 4);
        setAction(13, 10, 's', 6);
        setAction(13, 1, 's', 7);
        setAction(13, 9, 's', 8);
        setAction(13, 0, 's', 9);
        setGoto(13, 1, 31);
        setGoto(13, 3, 5);
        setGoto(13, 2, 10);
        // State 14
        setAction(14, 11, 's', 3);
        setAction(14, 4, 's', 4);
        setAction(14, 10, 's', 6);
        setAction(14, 1, 's', 7);
        setAction(14, 9, 's', 8);
        setAction(14, 0, 's', 9);
        setGoto(14, 1, 32);
        setGoto(14, 3, 5);
        setGoto(14, 2, 10);
        // State 15
        setAction(15, 7, 'r', 3);
        setAction(15, 8, 'r', 3);
        setAction(15, 3, 'r', 3);
        setAction(15, 5, 's', 33);
        setAction(15, 6, 's', 34);
        // State 16
        setAction(16, 3, 's', 35);
        setAction(16, 7, 's', 36);
        setAction(16, 8, 's', 37);
        // State 17
        setAction(17, 5, 'r', 6);
        setAction(17, 8, 'r', 6);
        setAction(17, 7, 'r', 6);
        setAction(17, 3, 'r', 6);
        setAction(17, 6, 'r', 6);
        // State 18
        setAction(18, 8, 'r', 8);
        setAction(18, 5, 'r', 8);
        setAction(18, 7, 'r', 8);
        setAction(18, 3, 'r', 8);
        setAction(18, 6, 'r', 8);
        // State 19
        setAction(19, 7, 'r', 7);
        setAction(19, 3, 'r', 7);
        setAction(19, 6, 'r', 7);
        setAction(19, 5, 'r', 7);
        setAction(19, 8, 'r', 7);
        // State 20
        setAction(20, 11, 's', 18);
        setAction(20, 4, 's', 20);
        setAction(20, 10, 's', 21);
        setAction(20, 1, 's', 22);
        setAction(20, 9, 's', 23);
        setAction(20, 0, 's', 24);
        setGoto(20, 1, 15);
        setGoto(20, 0, 38);
        setGoto(20, 2, 17);
        setGoto(20, 3, 19);
        // State 21
        setAction(21, 4, 's', 39);
        // State 22
        setAction(22, 4, 's', 40);
        // State 23
        setAction(23, 4, 's', 41);
        // State 24
        setAction(24, 4, 's', 42);
        // State 25
        setAction(25, 11, 's', 18);
        setAction(25, 4, 's', 20);
        setAction(25, 10, 's', 21);
        setAction(25, 1, 's', 22);
        setAction(25, 9, 's', 23);
        setAction(25, 0, 's', 24);
        setGoto(25, 1, 15);
        setGoto(25, 0, 43);
        setGoto(25, 2, 17);
        setGoto(25, 3, 19);
        // State 26
        setAction(26, 11, 's', 18);
        setAction(26, 4, 's', 20);
        setAction(26, 10, 's', 21);
        setAction(26, 1, 's', 22);
        setAction(26, 9, 's', 23);
        setAction(26, 0, 's', 24);
        setGoto(26, 0, 44);
        setGoto(26, 2, 17);
        setGoto(26, 1, 15);
        setGoto(26, 3, 19);
        // State 27
        setAction(27, 11, 's', 18);
        setAction(27, 4, 's', 20);
        setAction(27, 10, 's', 21);
        setAction(27, 1, 's', 22);
        setAction(27, 9, 's', 23);
        setAction(27, 0, 's', 24);
        setGoto(27, 1, 15);
        setGoto(27, 0, 45);
        setGoto(27, 2, 17);
        setGoto(27, 3, 19);
        // State 28
        setAction(28, 11, 's', 18);
        setAction(28, 4, 's', 20);
        setAction(28, 10, 's', 21);
        setAction(28, 1, 's', 22);
        setAction(28, 9, 's', 23);
        setAction(28, 0, 's', 24);
        setGoto(28, 1, 15);
        setGoto(28, 0, 46);
        setGoto(28, 2, 17);
        setGoto(28, 3, 19);
        // State 29
        setAction(29, 5, 'r', 5);
        setAction(29, 8, 'r', 5);
        setAction(29, 7, 'r', 5);
        setAction(29, 6, 'r', 5);
        setAction(29, 2, 'r', 5);
        // State 30
        setAction(30, 8, 'r', 4);
        setAction(30, 7, 'r', 4);
        setAction(30, 6, 'r', 4);
        setAction(30, 2, 'r', 4);
        setAction(30, 5, 'r', 4);
        // State 31
        setAction(31, 7, 'r', 1);
        setAction(31, 2, 'r', 1);
        setAction(31, 8, 'r', 1);
        setAction(31, 5, 's', 11);
        setAction(31, 6, 's', 12);
        // State 32
        setAction(32, 2, 'r', 2);
        setAction(32, 8, 'r', 2);
        setAction(32, 7, 'r', 2);
        setAction(32, 5, 's', 11);
        setAction(32, 6, 's', 12);
        // State 33
        setAction(33, 11, 's', 18);
        setAction(33, 4, 's', 20);
        setAction(33, 10, 's', 21);
        setAction(33, 1, 's', 22);
        setAction(33, 9, 's', 23);
        setAction(33, 0, 's', 24);
        setGoto(33, 3, 19);
        setGoto(33, 2, 47);
        // State 34
        setAction(34, 11, 's', 18);
        setAction(34, 4, 's', 20);
        setAction(34, 10, 's', 21);
        setAction(34, 1, 's', 22);
        setAction(34, 9, 's', 23);
        setAction(34, 0, 's', 24);
        setGoto(34, 2, 48);
        setGoto(34, 3, 19);
        // State 35
        setAction(35, 6, 'r', 9);
        setAction(35, 2, 'r', 9);
        setAction(35, 5, 'r', 9);
        setAction(35, 8, 'r', 9);
        setAction(35, 7, 'r', 9);
        // State 36
        setAction(36, 11, 's', 18);
        setAction(36, 4, 's', 20);
        setAction(36, 10, 's', 21);
        setAction(36, 1, 's', 22);
        setAction(36, 9, 's', 23);
        setAction(36, 0, 's', 24);
        setGoto(36, 1, 49);
        setGoto(36, 2, 17);
        setGoto(36, 3, 19);
        // State 37
        setAction(37, 11, 's', 18);
        setAction(37, 4, 's', 20);
        setAction(37, 10, 's', 21);
        setAction(37, 1, 's', 22);
        setAction(37, 9, 's', 23);
        setAction(37, 0, 's', 24);
        setGoto(37, 1, 50);
        setGoto(37, 2, 17);
        setGoto(37, 3, 19);
        // State 38
        setAction(38, 3, 's', 51);
        setAction(38, 7, 's', 36);
        setAction(38, 8, 's', 37);
        // State 39
        setAction(39, 11, 's', 18);
        setAction(39, 4, 's', 20);
        setAction(39, 10, 's', 21);
        setAction(39, 1, 's', 22);
        setAction(39, 9, 's', 23);
        setAction(39, 0, 's', 24);
        setGoto(39, 1, 15);
        setGoto(39, 0, 52);
        setGoto(39, 2, 17);
        setGoto(39, 3, 19);
        // State 40
        setAction(40, 11, 's', 18);
        setAction(40, 4, 's', 20);
        setAction(40, 10, 's', 21);
        setAction(40, 1, 's', 22);
        setAction(40, 9, 's', 23);
        setAction(40, 0, 's', 24);
        setGoto(40, 0, 53);
        setGoto(40, 2, 17);
        setGoto(40, 1, 15);
        setGoto(40, 3, 19);
        // State 41
        setAction(41, 11, 's', 18);
        setAction(41, 4, 's', 20);
        setAction(41, 10, 's', 21);
        setAction(41, 1, 's', 22);
        setAction(41, 9, 's', 23);
        setAction(41, 0, 's', 24);
        setGoto(41, 1, 15);
        setGoto(41, 0, 54);
        setGoto(41, 2, 17);
        setGoto(41, 3, 19);
        // State 42
        setAction(42, 11, 's', 18);
        setAction(42, 4, 's', 20);
        setAction(42, 10, 's', 21);
        setAction(42, 1, 's', 22);
        setAction(42, 9, 's', 23);
        setAction(42, 0, 's', 24);
        setGoto(42, 1, 15);
        setGoto(42, 0, 55);
        setGoto(42, 2, 17);
        setGoto(42, 3, 19);
        // State 43
        setAction(43, 7, 's', 36);
        setAction(43, 3, 's', 56);
        setAction(43, 8, 's', 37);
        // State 44
        setAction(44, 3, 's', 57);
        setAction(44, 7, 's', 36);
        setAction(44, 8, 's', 37);
        // State 45
        setAction(45, 3, 's', 58);
        setAction(45, 7, 's', 36);
        setAction(45, 8, 's', 37);
        // State 46
        setAction(46, 7, 's', 36);
        setAction(46, 3, 's', 59);
        setAction(46, 8, 's', 37);
        // State 47
        setAction(47, 5, 'r', 5);
        setAction(47, 8, 'r', 5);
        setAction(47, 7, 'r', 5);
        setAction(47, 3, 'r', 5);
        setAction(47, 6, 'r', 5);
        // State 48
        setAction(48, 8, 'r', 4);
        setAction(48, 7, 'r', 4);
        setAction(48, 6, 'r', 4);
        setAction(48, 3, 'r', 4);
        setAction(48, 5, 'r', 4);
        // State 49
        setAction(49, 8, 'r', 2);
        setAction(49, 7, 'r', 2);
        setAction(49, 3, 'r', 2);
        setAction(49, 5, 's', 33);
        setAction(49, 6, 's', 34);
        // State 50
        setAction(50, 7, 'r', 1);
        setAction(50, 3, 'r', 1);
        setAction(50, 8, 'r', 1);
        setAction(50, 5, 's', 33);
        setAction(50, 6, 's', 34);
        // State 51
        setAction(51, 6, 'r', 9);
        setAction(51, 3, 'r', 9);
        setAction(51, 5, 'r', 9);
        setAction(51, 8, 'r', 9);
        setAction(51, 7, 'r', 9);
        // State 52
        setAction(52, 7, 's', 36);
        setAction(52, 3, 's', 60);
        setAction(52, 8, 's', 37);
        // State 53
        setAction(53, 3, 's', 61);
        setAction(53, 7, 's', 36);
        setAction(53, 8, 's', 37);
        // State 54
        setAction(54, 7, 's', 36);
        setAction(54, 3, 's', 62);
        setAction(54, 8, 's', 37);
        // State 55
        setAction(55, 7, 's', 36);
        setAction(55, 3, 's', 63);
        setAction(55, 8, 's', 37);
        // State 56
        setAction(56, 6, 'r', 10);
        setAction(56, 5, 'r', 10);
        setAction(56, 2, 'r', 10);
        setAction(56, 8, 'r', 10);
        setAction(56, 7, 'r', 10);
        // State 57
        setAction(57, 5, 'r', 12);
        setAction(57, 2, 'r', 12);
        setAction(57, 8, 'r', 12);
        setAction(57, 7, 'r', 12);
        setAction(57, 6, 'r', 12);
        // State 58
        setAction(58, 7, 'r', 11);
        setAction(58, 6, 'r', 11);
        setAction(58, 2, 'r', 11);
        setAction(58, 5, 'r', 11);
        setAction(58, 8, 'r', 11);
        // State 59
        setAction(59, 6, 'r', 13);
        setAction(59, 2, 'r', 13);
        setAction(59, 5, 'r', 13);
        setAction(59, 8, 'r', 13);
        setAction(59, 7, 'r', 13);
        // State 60
        setAction(60, 3, 'r', 10);
        setAction(60, 6, 'r', 10);
        setAction(60, 5, 'r', 10);
        setAction(60, 8, 'r', 10);
        setAction(60, 7, 'r', 10);
        // State 61
        setAction(61, 5, 'r', 12);
        setAction(61, 8, 'r', 12);
        setAction(61, 7, 'r', 12);
        setAction(61, 3, 'r', 12);
        setAction(61, 6, 'r', 12);
        // State 62
        setAction(62, 7, 'r', 11);
        setAction(62, 6, 'r', 11);
        setAction(62, 3, 'r', 11);
        setAction(62, 5, 'r', 11);
        setAction(62, 8, 'r', 11);
        // State 63
        setAction(63, 6, 'r', 13);
        setAction(63, 5, 'r', 13);
        setAction(63, 8, 'r', 13);
        setAction(63, 7, 'r', 13);
        setAction(63, 3, 'r', 13);
    }

public:
    LR1Parser(const std::string& input) : lexer(input) {
        initTables();
        stack.push_back(0);
        current = lexer.nextToken();
    }

    void printStack() {
        std::cout << "栈: ";
        for (auto s : stack) std::cout << s << " ";
        std::cout << "     ";
    }

    void printReduction(int prod) {
        std::cout << "使用产生式 " << prod << " 归约: ";
        int lhs = prods[prod].first;
        std::string lhs_str = (lhs == 0 ? "S" : lhs == 1 ? "E" : lhs == 2 ? "T" : lhs == 3 ? "F" : "A");
        std::cout << lhs_str << " -> ";
        switch (prod) {
        case 0: std::cout << "E"; break;
        case 1: std::cout << "E + T"; break;
        case 2: std::cout << "E - T"; break;
        case 3: std::cout << "T"; break;
        case 4: std::cout << "T * F"; break;
        case 5: std::cout << "T / F"; break;
        case 6: std::cout << "F"; break;
        case 7: std::cout << "A"; break;
        case 8: std::cout << "NUM"; break;
        case 9: std::cout << "( E )"; break;
        case 10: std::cout << "sin ( E )"; break;
        case 11: std::cout << "cos ( E )"; break;
        case 12: std::cout << "lg ( E )"; break;
        case 13: std::cout << "ln ( E )"; break;
        }
        std::cout << "     ";
    }

    bool parse() {
        while (true) {
            printStack();
            std::cout << "当前token: "  << current.value << "     ";

            int state = stack.back();
            int t_idx = termIndex(current.type);
            if (t_idx == -1) {
                std::cout << "非法token" << std::endl;
                return false;
            }

            Action act = actionTable[state * 12 + t_idx];

            if (act.type == 's') {
                std::cout << "移进到状态 " << act.value << std::endl;
                stack.push_back(act.value);
                current = lexer.nextToken();
            }
            else if (act.type == 'r') {
                printReduction(act.value);
                int len = prods[act.value].second;
                for (int i = 0; i < len; ++i) stack.pop_back();
                int top = stack.back();
                int lhs = prods[act.value].first;
                int nt_idx = lhs - 1;  // E=1 -> 0, etc.
                int newState = gotoTable[top * 4 + nt_idx];
                if (newState == -1) {
                    std::cout << "Goto error" << std::endl;
                    return false;
                }
                std::cout << "Goto状态 " << newState << std::endl;
                stack.push_back(newState);
            }
            else if (act.type == 'a') {
                std::cout << "Accept" << std::endl;
                return true;
            }
            else {
                std::cout << "Syntax error" << std::endl;
                return false;
            }
        }
    }
};

int main() {
    std::string input = "sin(3 - 4)";
    LR1Parser parser(input);
    parser.parse();
    return 0;
}