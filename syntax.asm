; Streaming C++ syntax highlighter, loaded with the model above 0x8000.
; Position-independent: BP is the runtime address of this code. The MBR
; tail-jumps here with AL=character; PUSHA/POPA preserves all callers.
; Only the standard 16 VGA text attributes are needed for the GitHub palette.

[org 0]
[bits 16]

COLOR_TEXT      equ 0x07        ; #c9d1d9 -> light gray
COLOR_KEYWORD   equ 0x0c        ; #ff7b72 -> light red
COLOR_FUNCTION  equ 0x0d        ; #d2a8ff -> light magenta
COLOR_CONSTANT  equ 0x0b        ; #79c0ff / #a5d6ff -> light cyan
COLOR_STRING    equ 0x0b
COLOR_BUILTIN   equ 0x0e        ; #ffa657 -> yellow
COLOR_COMMENT   equ 0x08        ; #8b949e -> dark gray
COLOR_ENTITY    equ 0x0a        ; #7ee787 -> light green

ST_DEFAULT      equ 0
ST_COMMENT      equ 1
ST_STRING       equ 2
ST_CHAR         equ 3
ST_HEADER       equ 4
ST_SLASH        equ 5
ST_IDENT        equ 6
ST_NUMBER       equ 7

F_INCLUDE       equ 0x40
F_DECL          equ 0x80

STATE           equ 0x7b86
ESCAPE          equ 0x7b87
TOKENLEN        equ 0x7b88
FLAGS           equ 0x7b89
INCHAR          equ 0x7b8a
TOKEN           equ 0x7b90        ; 32-byte, NUL-terminated token buffer

highlight:
        mov [INCHAR], al
        pusha
        call .position
.position:
        pop bp
        sub bp, .position         ; BP = load base (labels assembled at 0)
        mov bl, COLOR_TEXT
        mov al, [INCHAR]
        mov cl, [STATE]
        test cl, cl
        jz .default
        cmp cl, ST_COMMENT
        je .comment
        cmp cl, ST_STRING
        je .string
        cmp cl, ST_CHAR
        je .char
        cmp cl, ST_HEADER
        je .header
        cmp cl, ST_SLASH
        je .slash
        cmp cl, ST_IDENT
        je .ident
        jmp .number               ; ST_NUMBER

.default:
        cmp al, 10
        jne .not_newline
        mov byte [FLAGS], 0
.not_newline:
        cmp al, '/'
        je .start_slash
        cmp al, '"'
        je .start_string
        cmp al, 39                ; single quote
        je .start_char
        cmp al, '<'
        jne .not_header
        test byte [FLAGS], F_INCLUDE
        jz .emit                  ; template/operator angle bracket
        and byte [FLAGS], ~F_INCLUDE
        mov byte [STATE], ST_HEADER
        mov bl, COLOR_STRING
        jmp .emit
.not_header:
        cmp al, '#'
        jne .not_preprocessor
        mov bl, COLOR_KEYWORD
        jmp .emit
.not_preprocessor:
        cmp al, '0'
        jb .not_number
        cmp al, '9'
        jbe .start_number
.not_number:
        cmp al, '_'
        je .start_ident
        cmp al, 'A'
        jb .emit
        cmp al, 'Z'
        jbe .start_ident
        cmp al, 'a'
        jb .emit
        cmp al, 'z'
        ja .emit
.start_ident:
        mov byte [STATE], ST_IDENT
        mov [TOKEN], al
        mov byte [TOKEN+1], 0
        mov byte [TOKENLEN], 1
        jmp .done

.start_slash:
        mov byte [STATE], ST_SLASH
        jmp .done                 ; delay '/' until we know it is not '//'
.start_string:
        mov byte [STATE], ST_STRING
        mov byte [ESCAPE], 0
        mov bl, COLOR_STRING
        jmp .emit
.start_char:
        mov byte [STATE], ST_CHAR
        mov byte [ESCAPE], 0
        mov bl, COLOR_STRING
        jmp .emit
.start_number:
        mov byte [STATE], ST_NUMBER
        mov bl, COLOR_CONSTANT
        jmp .emit

.slash:
        mov byte [STATE], ST_DEFAULT
        cmp al, '/'
        jne .single_slash
        mov byte [STATE], ST_COMMENT
        mov bl, COLOR_COMMENT
        mov al, '/'
        call emit
        jmp .emit                 ; second slash, already COMMENT
.single_slash:
        push ax
        mov al, '/'
        call emit                 ; ordinary division operator
        pop ax
        jmp .default

.comment:
        cmp al, 10
        jne .comment_color
        mov byte [STATE], ST_DEFAULT
        mov byte [FLAGS], 0
.comment_color:
        mov bl, COLOR_COMMENT
        jmp .emit

.string:
        mov bl, COLOR_STRING
        call emit
        jmp .quoted_tail
.char:
        mov bl, COLOR_STRING
        call emit
        mov ah, 39
        jmp .quoted_test
.quoted_tail:
        mov ah, '"'
.quoted_test:
        cmp byte [ESCAPE], 0
        je .not_escaped
        mov byte [ESCAPE], 0
        jmp .done
.not_escaped:
        cmp al, '\'
        jne .not_backslash
        mov byte [ESCAPE], 1
        jmp .done
.not_backslash:
        cmp al, ah
        jne .quoted_newline
        mov byte [STATE], ST_DEFAULT
        jmp .done
.quoted_newline:
        cmp al, 10
        jne .done
        mov byte [STATE], ST_DEFAULT
        jmp .done

.header:
        mov bl, COLOR_STRING
        cmp al, '>'
        jne .emit
        mov byte [STATE], ST_DEFAULT
        jmp .emit

.ident:
        cmp al, '_'
        je .append_ident
        cmp al, '0'
        jb .flush_ident
        cmp al, '9'
        jbe .append_ident
        cmp al, 'A'
        jb .flush_ident
        cmp al, 'Z'
        jbe .append_ident
        cmp al, 'a'
        jb .flush_ident
        cmp al, 'z'
        ja .flush_ident
.append_ident:
        movzx bx, byte [TOKENLEN]
        cmp bl, 31
        jae .done
        mov [TOKEN+bx], al
        inc bx
        mov byte [TOKEN+bx], 0
        mov [TOKENLEN], bl
        jmp .done
.flush_ident:
        mov byte [STATE], ST_DEFAULT
        call flush_token
        mov al, [INCHAR]
        mov bl, COLOR_TEXT
        jmp .default

.number:
        cmp al, '0'
        jb .number_dot
        cmp al, '9'
        jbe .number_more
        cmp al, 'A'
        jb .number_dot
        cmp al, 'Z'
        jbe .number_more
        cmp al, 'a'
        jb .number_dot
        cmp al, 'z'
        jbe .number_more
.number_dot:
        cmp al, '.'
        je .number_more
        cmp al, '_'
        je .number_more
        mov byte [STATE], ST_DEFAULT
        mov bl, COLOR_TEXT
        jmp .default
.number_more:
        mov bl, COLOR_CONSTANT
        jmp .emit

.emit:
        call emit
.done:
        popa
        ret

; Classify and write the buffered identifier. The delimiter remains INCHAR.
; Dictionary records are attribute byte + NUL-terminated spelling. Attribute
; high bits remember #include headers and declarations requiring an entity.
flush_token:
        lea di, [bp+dictionary]
        mov si, TOKEN
.next_record:
        mov bl, [cs:di]
        inc di
        test bl, bl
        jz .plain
        push si
.compare:
        lodsb
        mov ah, [cs:di]
        inc di
        cmp al, ah
        jne .mismatch
        test al, al
        jz .matched
        jmp .compare
.mismatch:
        test ah, ah
        jz .skip_done
.skip_record:
        mov ah, [cs:di]
        inc di
        test ah, ah
        jnz .skip_record
.skip_done:
        pop si
        jmp .next_record
.matched:
        pop si
        test bl, F_INCLUDE
        jz .not_include
        or byte [FLAGS], F_INCLUDE
.not_include:
        test bl, F_DECL
        jz .found
        or byte [FLAGS], F_DECL
.found:
        and bl, 0x0f
        jmp .write
.plain:
        test byte [FLAGS], F_DECL
        jz .not_entity_decl
        and byte [FLAGS], ~F_DECL
        mov bl, COLOR_ENTITY
        jmp .write
.not_entity_decl:
        mov al, [TOKEN]
        cmp al, 'A'
        jb .constant
        cmp al, 'Z'
        jbe .entity
.constant:
        cmp al, 'k'
        jne .function
        mov al, [TOKEN+1]
        cmp al, 'A'
        jb .function
        cmp al, 'Z'
        jbe .numeric_constant
.function:
        cmp byte [INCHAR], '('
        jne .plain_text
        mov bl, COLOR_FUNCTION
        jmp .write
.numeric_constant:
        mov bl, COLOR_CONSTANT
        jmp .write
.entity:
        mov bl, COLOR_ENTITY
        jmp .write
.plain_text:
        mov bl, COLOR_TEXT
.write:
        mov si, TOKEN
.write_loop:
        lodsb
        test al, al
        jz .written
        call emit
        jmp .write_loop
.written:
        ret

; Write AL with attribute BL on page zero. AH=09 sets a text-mode attribute
; but does not advance; AH=0E then performs the normal wrap/scroll/advance.
emit:
        cmp al, 13
        je .advance
        cmp al, 10
        je .advance
        push ax
        mov ah, 0x09
        xor bh, bh
        mov cx, 1
        int 0x10
        pop ax
.advance:
        mov ah, 0x0e
        xor bh, bh
        int 0x10
        ret

; C++20 keywords (including preprocessor directives), then built-in types.
; F_DECL marks class/struct/enum/namespace; F_INCLUDE marks include.
dictionary:
%macro KEYWORD 1
        db COLOR_KEYWORD, %1, 0
%endmacro
%macro DECLKEYWORD 1
        db COLOR_KEYWORD | F_DECL, %1, 0
%endmacro
        KEYWORD "alignas"
        KEYWORD "alignof"
        KEYWORD "asm"
        KEYWORD "auto"
        KEYWORD "break"
        KEYWORD "case"
        KEYWORD "catch"
        DECLKEYWORD "class"
        KEYWORD "concept"
        KEYWORD "const"
        KEYWORD "consteval"
        KEYWORD "constexpr"
        KEYWORD "constinit"
        KEYWORD "const_cast"
        KEYWORD "continue"
        KEYWORD "co_await"
        KEYWORD "co_return"
        KEYWORD "co_yield"
        KEYWORD "decltype"
        KEYWORD "default"
        KEYWORD "delete"
        KEYWORD "do"
        KEYWORD "dynamic_cast"
        KEYWORD "else"
        DECLKEYWORD "enum"
        KEYWORD "explicit"
        KEYWORD "export"
        KEYWORD "extern"
        KEYWORD "false"
        KEYWORD "final"
        KEYWORD "for"
        KEYWORD "friend"
        KEYWORD "goto"
        KEYWORD "if"
        db COLOR_KEYWORD | F_INCLUDE, "include", 0
        KEYWORD "inline"
        KEYWORD "mutable"
        DECLKEYWORD "namespace"
        KEYWORD "new"
        KEYWORD "noexcept"
        KEYWORD "not"
        KEYWORD "nullptr"
        KEYWORD "operator"
        KEYWORD "override"
        KEYWORD "pragma"
        KEYWORD "private"
        KEYWORD "protected"
        KEYWORD "public"
        KEYWORD "reinterpret_cast"
        KEYWORD "requires"
        KEYWORD "return"
        KEYWORD "sizeof"
        KEYWORD "static"
        KEYWORD "static_assert"
        KEYWORD "static_cast"
        DECLKEYWORD "struct"
        KEYWORD "switch"
        KEYWORD "template"
        KEYWORD "this"
        KEYWORD "thread_local"
        KEYWORD "throw"
        KEYWORD "true"
        KEYWORD "try"
        KEYWORD "typedef"
        KEYWORD "typeid"
        KEYWORD "typename"
        DECLKEYWORD "union"
        KEYWORD "using"
        KEYWORD "virtual"
        KEYWORD "volatile"
        KEYWORD "while"
%macro BUILTIN 1
        db COLOR_BUILTIN, %1, 0
%endmacro
        BUILTIN "bool"
        BUILTIN "char"
        BUILTIN "char8_t"
        BUILTIN "char16_t"
        BUILTIN "char32_t"
        BUILTIN "double"
        BUILTIN "float"
        BUILTIN "int"
        BUILTIN "int8_t"
        BUILTIN "int16_t"
        BUILTIN "int32_t"
        BUILTIN "int64_t"
        BUILTIN "long"
        BUILTIN "short"
        BUILTIN "signed"
        BUILTIN "size_t"
        BUILTIN "uint8_t"
        BUILTIN "uint16_t"
        BUILTIN "uint32_t"
        BUILTIN "uint64_t"
        BUILTIN "unsigned"
        BUILTIN "void"
        BUILTIN "wchar_t"
        db 0
