%%
[ \t\r]+                ; /* 忽略空白 */

[0-9]+(\.[0-9]+)?       { printf("NUM (%s)\n", yytext); }

"sin"                   { printf("SIN (sin)\n"); }
"cos"                   { printf("COS (cos)\n"); }

"+"                     { printf("PLUS (+)\n"); }
"-"                     { printf("MINUS (-)\n"); }
"*"                     { printf("STAR (*)\n"); }
"/"                     { printf("SLASH (/)\n"); }
"("                     { printf("LPAREN (()\n"); }
")"                     { printf("RPAREN ())\n"); }

\n                      { printf("EOL\n"); }

.                       { fprintf(stderr, "非法字符: %s\n", yytext); }

%%
int main() {
    yylex();
    return 0;
}
int yywrap() { return 1; }