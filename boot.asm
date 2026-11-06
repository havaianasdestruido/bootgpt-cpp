; bootgpt-cpp: a neural network that boots your PC and dreams C++20.
; Stage 1 (this, <=510 bytes): loads int8 weights from sectors 2+ and
; runs char-level MLP inference in 16-bit real mode on a 386+.
; Weights (FS=0x0800 segment, physical 0x08000..0x0F7FF):
;   +0       vocab[VOCAB]         ASCII, '\n'=10
;   +VOCAB   emb[VOCAB*EMBDIM]    int8, [v*EMBDIM+r]
;   +W1_OFF  W1[HID*(CTX*EMBDIM)] int8, [j*(CTX*EMBDIM)+i], i = k*EMBDIM+r
;   +W2_OFF  W2[VOCAB*HID]        int8, [v*HID+h]
;   +B1_OFF  b1[HID]              int16 Q12
;   +B2_OFF  b2[VOCAB]            int16 Q8
;   +...     seedctx[CTX], hdr, byemsg (addressed via DS=0 at 0x8000+off)
; Runtime (DS=0, SS=0):
;   ctx 0x7600[32], y 0x7700[100], logits 0x7800[176], xin 0x7A00[192],
;   rnd 0x7B80, bootdrv 0x7B82, dbgctr 0x7B84

[org 0x7C00]
[cpu 386]
[bits 16]
%include "model/params.inc"

EMB_OFF     equ VOCAB

CTXBUF      equ 0x7600
YBUF        equ 0x7700
LOGBUF      equ 0x7800
XINBUF      equ 0x7A00
RND         equ 0x7B80
BOOTDRV     equ 0x7B82
DBGCTR      equ 0x7B84

start:
        cli
        xor ax, ax
        mov ds, ax
        mov ss, ax
        mov sp, 0x7000
        sti
        mov [BOOTDRV], dl
        mov ax, 0x0800
        mov fs, ax
        mov es, ax

; ---- load SECTORS sectors (weights) from CHS 0/0/2 to 0x0800:0000 ----
.load:
        mov ax, 0x0200 | SECTORS
        mov cx, 0x0002          ; C=0, S=2
        xor dx, dx              ; H=0
        mov dl, [BOOTDRV]
        xor bx, bx
        int 0x13
        jc .load

%ifndef DBGPORT
        mov si, HDRADDR
        call puts
%endif

; ---- seed the LCG from PIT ticks (SEEDFIX for reproducible debug runs) ----
%ifdef SEEDFIX
        mov dx, SEEDFIX
%else
        xor ah, ah
        int 0x1A
%ifndef DBGPORT
        or dx, 1
%endif
%endif
        mov [RND], dx

; ---- ctx = seed indices (stored in weights blob at SEEDADDR) ----
        mov si, SEEDADDR
        xor bx, bx
.ctxin:
        mov al, [si+bx]
        mov [CTXBUF+bx], al
        inc bx
        cmp bx, CTX
        jb .ctxin

mainloop:
        call infer              ; -> bx = next char index (bh=0)
%ifdef DBGSTOP
        inc byte [DBGCTR]
        cmp byte [DBGCTR], DBGSTOP
        jb .go
        jmp $                   ; park: buffers hold last inference state
.go:
%endif
        push bx

        mov al, [fs:bx]         ; vocab[idx]
        cmp al, 10
        jne .noCR
        mov al, 13
        call putc
        mov al, 10
.noCR:
        call putc
        pop bx

        ; shift ctx, append prediction
        mov si, CTXBUF
        mov cx, CTX-1
.sh:
        mov al, [si+1]
        mov [si], al
        inc si
        loop .sh
        mov [si], bl

%ifndef DBGPORT
        mov ah, 1               ; key pressed?
        int 0x16
        jnz keyexit

        mov ah, 0x86            ; wait ~10 ms
        xor cx, cx
        mov dx, 10000
        int 0x15
%endif
        jmp mainloop

keyexit:
        xor ah, ah              ; consume the keypress so the buffer
        int 0x16                ; is empty after the reboot
%ifndef DBGPORT
        mov si, BYEADDR
        call puts
%endif
        int 0x19                ; reboot

; ---------------- inference (mirrors BootSim in train.py) ----------------
infer:
        ; xin[i] = emb[EMB_OFF + ctx[k]*EMBDIM + r], i = k*EMBDIM+r
        xor bx, bx
        xor bp, bp
.kl:
        movzx ax, byte [CTXBUF+bp]
        imul si, ax, EMBDIM
        add si, EMB_OFF
        mov cx, EMBDIM
.rl:
        mov al, [fs:si]
        mov [XINBUF+bx], al
        inc si
        inc bx
        loop .rl
        inc bp
        cmp bp, CTX
        jb .kl

        ; hidden: y[j] = trunc(p*64 / (4096+|p|))   (always < 64, no clip)
        ;          p = (Sigma xin*W1[j]) [<</>>]A1SHIFT + b1[j]   (Q12)
        mov di, W1_OFF
        xor bp, bp
.jl:
        xor esi, esi
        xor bx, bx
.mac1:
        movsx eax, byte [XINBUF+bx]
        movsx ecx, byte [fs:di+bx]
        imul eax, ecx
        add esi, eax
        inc bx
        cmp bx, CTX*EMBDIM
        jb .mac1
%if A1SHIFT >= 0
        sar esi, A1SHIFT
%else
        shl esi, -A1SHIFT
%endif
        mov bx, bp
        add bx, bx
        movsx eax, word [fs:bx+B1_OFF]
        add esi, eax            ; esi = p
        mov ebx, esi
        test ebx, ebx
        jns .p1
        neg ebx
.p1:
        add ebx, 4096           ; ebx = 4096+|p|
        mov eax, esi
        shl eax, 6
        cdq
        idiv ebx                ; eax = trunc(p*64 / (4096+|p|))
        mov [YBUF+bp], al
%ifdef DBGPORT
        mov eax, esi
        out 0x80, al
        shr eax, 8
        out 0x80, al
        shr eax, 8
        out 0x80, al
        shr eax, 8
        out 0x80, al
%endif
        add di, CTX*EMBDIM
        inc bp
        cmp bp, HID
        jb .jl

        ; logits: L[v] = (Sigma y*W2[v]) [<</>>]A2SHIFT + b2[v]  (Q8, int16)
        ; track dx = max_v L[v] for Min-P logit-margin gating
        mov dx, 0x8000
        mov di, W2_OFF
        xor bp, bp
.vl:
        xor esi, esi
        xor bx, bx
.mac2:
        movsx eax, byte [YBUF+bx]
        movsx ecx, byte [fs:di+bx]
        imul eax, ecx
        add esi, eax
        inc bx
        cmp bx, HID
        jb .mac2
%if A2SHIFT >= 0
        sar esi, A2SHIFT
%else
        shl esi, -A2SHIFT
%endif
        mov bx, bp
        add bx, bx
        movsx eax, word [fs:bx+B2_OFF]
        add esi, eax
        mov [LOGBUF+bx], si
        cmp si, dx
        jle .nm
        mov dx, si
.nm:
        add di, HID
        inc bp
        cmp bp, VOCAB
        jb .vl

        ; Min-P sample: among v with L[v] >= L_max - MARGIN,
        ; argmax( L[v] + int8(rnd>>8)*NOISE ), first index wins ties
        movsx esi, dx
        sub esi, MARGIN
        mov edi, 0x80000000
        xor bx, bx
        xor bp, bp
.sl:
        mov ax, [RND]
        imul ax, ax, 25173
        add ax, 13849
        mov [RND], ax
        movsx ecx, word [LOGBUF+bp]
        cmp ecx, esi
        jl .nx
        mov al, ah
        cbw
        imul ax, ax, NOISE
        movsx edx, ax
        add ecx, edx
        cmp ecx, edi
        jle .nx
        mov edi, ecx
        mov bx, bp
.nx:
        add bp, 2
        cmp bp, VOCAB*2
        jb .sl
        shr bx, 1
        ret                     ; bx = argmax index (bh=0)

; ---------------- bios helpers ----------------
putc:                           ; AL = char (clobbers AX/BX, FS untouched)
        mov ah, 0x0E
        xor bx, bx
        int 0x10
        ret

puts:                           ; DS:SI -> asciiz
        lodsb
        or al, al
        jz .d
        call putc
        jmp puts
.d:
        ret

        times 510-($-$$) db 0
        dw 0xAA55
