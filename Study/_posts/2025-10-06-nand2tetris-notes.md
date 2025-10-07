---
title: "Revisiting Nand2Tetris: Notes"
tags:
  - nand2tetris
  - computer system
  - notes
---
## Boolean Logic
### De Morgan’s Law

$$
\overline{AB} = \overline{A} + \overline{B}
$$

$$
\overline{A + B} = \overline{A}\,\overline{B}
$$

## Boolean Arithmetic (Combinational Logic)
#### Half Adder

![half-adder.png](https://images.zijianguo.com/half-adder.png)

#### Full Adder

![full-adder.png](https://images.zijianguo.com/full-adder.png)

#### Two’s Complement
##### Quick Flow

```
Decimal → (abs) → bin → invert → +1 → Two’s complement
Binary  → (MSB=1?) → invert → +1 → decimal → negative
```
##### Binary → Decimal
**Concept**

```
if MSB = 0 → positive
     normal binary value

if MSB = 1 → negative
     invert bits → add 1 → decimal → add minus sign


Example:

    11101100₂
    
    invert → 00010011
    +1     → 00010100 = 20
    → -20₁₀
```

**Formula**

$$
\text{Range} = [-2^{n-1},\, 2^{n-1} - 1]
$$

$$
\text{Decimal} = -b_{n-1} \times 2^{n-1} + \sum_{i=0}^{n-2} b_i \times 2^i
$$

**Example via Formula**

```
Directly convert a binary number to decimal using the formula:

    11101100₂
    = -1×128 + (1×64 + 1×32 + 0×16 + 1×8 + 1×4 + 0×2 + 0×1)
    = -128 + 108
    = -20₁₀
```

##### Decimal → Binary
**Concept**

```
choose bit width (ex, 8 bits)

if positive:
    normal binary → pad zeros

if negative:
    abs(decimal) → binary → invert → add 1

Example:

    -20

    20  → 00010100
    inv → 11101011
    +1  → 11101100
```

**Example: Positive via Long Division**

```
Example, 20₁₀, target width 8 bits

          ┌─────────── remainder
      2 ) 20           r0
          10           r0
           5           r1
           2           r0
           1           r1
           0  stop

remainders bottom to top, 1 0 1 0 0  →  00010100
```

**Example: Negative via Long Division**

```
Example, -20₁₀, target width 8 bits

Step A, abs value long divide

          ┌─────────── remainder
      2 ) 20           r0
          10           r0
           5           r1
           2           r0
           1           r1
           0  stop

unsigned, 10100  → pad to width → 00010100

Step B, invert bits
00010100 → 11101011

Step C, add 1
11101011 + 1 → 11101100

Result, -20₁₀ → 11101100₂
```

**4-bit Table**

```
Decimal | Two’s Complement
--------------------------
   7    | 0111
   6    | 0110
   5    | 0101
   4    | 0100
   3    | 0011
   2    | 0010
   1    | 0001
   0    | 0000
  -1    | 1111
  -2    | 1110
  -3    | 1101
  -4    | 1100
  -5    | 1011
  -6    | 1010
  -7    | 1001
  -8    | 1000
```

## Sequential Logic

### DFF

![dff.png](https://images.zijianguo.com/dff.png)

### Bit (1-bit register)

![1-bit-register.png](https://images.zijianguo.com/1-bit-register.png)

## Computer Architecture

![hack-computer.png](https://images.zijianguo.com/hack-computer.png)

### ALU
![alu.png](https://images.zijianguo.com/alu.png)

### ALU Notes
- In two's complement representation, the bitwise NOT operation $!y$ can be expressed as $!y = -y - 1$.


## Assembly Language
### Overview
The Hack assembly language contains two instruction types: **A-instruction** and **C-instruction**, plus labels for symbolic addresses. Each instruction is 16 bits long.

### A-instruction
**Form**
```
@value
@symbol
```
Loads a constant or address into the A register. The value also becomes the memory address for `M`.

**Example**
```
@10
D=A
@counter
M=0
```

### C-instruction
![c-instructions.png](https://images.zijianguo.com/c-instructions.png)

**Form**
```
dest=comp;jump
```
Performs computation and optionally stores or jumps.

- `dest`: target (M, D, A, MD, AM, AD, AMD)
- `comp`: computation (ALU operation)
- `jump`: condition (JGT, JEQ, JGE, JLT, JNE, JLE, JMP)

**Example**
```
D=M
D;JGT
0;JMP
```

### Common comp Values
```
0, 1, -1
D, A, M, !D, !A, !M, -D, -A, -M
D+1, A+1, M+1, D-1, A-1, M-1
D+A, D+M, D-A, D-M, A-D, M-D
D&A, D&M, D|A, D|M
```

### Labels and Symbols
**Label Declaration**
```
(LOOP)
```
Marks a location. The label’s value is the address of the next instruction.

**Predefined Symbols**

| Symbol                      | Address | Meaning                                                           |
| --------------------------- | ------- | ----------------------------------------------------------------- |
| **SP**                      | `0`     | Stack pointer (top of the stack)                                  |
| **LCL**                     | `1`     | Base address of the local segment                                 |
| **ARG**                     | `2`     | Base address of the argument segment                              |
| **THIS**                    | `3`     | Base address of the this segment                                  |
| **THAT**                    | `4`     | Base address of the that segment                                  |
| **R0–R15**                  | `0–15`  | General purpose registers, aliases for the first 16 RAM addresses |
| **temp (R5–R12)**           | `5–12`  | Fixed temporary segment, used for intermediate storage            |
| **pointer (THIS/THAT)**     | `3–4`   | Pointer segment that maps to `THIS` (0) and `THAT` (1)            |
| **static (FileName.index)** | `16+`   | Static variables unique to each `.vm` file, starting from RAM[16] |
| **SCREEN**                  | `16384` | Base address of the screen memory (for display pixels)            |
| **KBD**                     | `24576` | Address of the keyboard memory-mapped register                    |



Variables (custom symbols) start from address 16.

### Example Program: Sum 1+2+...+n
```
@i
M=0
@sum
M=0
(LOOP)
  @i
  D=M
  @n
  D=D-M
  @END
  D;JGT
  @i
  D=M
  @sum
  M=M+D
  @i
  M=M+1
  @LOOP
  0;JMP
(END)
```

### Quick Reference
- `@value`: load A
- `dest=comp;jump`: compute and control flow
- Symbols: variables and labels
- Memory map: R0–R15, SCREEN(16384), KBD(24576)

### ASM Notes
- First Pass: Scans the code to record label symbols and their ROM addresses in the symbol table.
- Second Pass: Translates all instructions into binary code and assigns memory addresses to variables.

---

> The following can be considered at the software level while the upper part is at the hardware level.

## Virtual Machine Language

The VM language is a stack-based intermediate language that abstracts away hardware details. It describes computation using stack operations, memory access, branching, and function calls.

### Stack and SP
The stack starts at address 256. The `SP` (Stack Pointer) always points to the next free slot.
- `push` writes to `*SP`, then `SP = SP + 1`
- `pop` decrements `SP`, then reads from `*SP`

```
// push D onto stack
@SP
A=M
M=D
@SP
M=M+1

// pop top of stack into D
@SP
AM=M-1
D=M
```

### Memory Segments

| VM Segment | Meaning | Assembly Base | Address Computation | Example |
|-------------|----------|----------------|---------------------|----------|
| **argument** | Function arguments | `ARG` | `A = M + index` | `push argument 2` → `*(ARG + 2)` |
| **local** | Local variables of the current function | `LCL` | `A = M + index` | `pop local 0` → `*(LCL + 0)` |
| **this** | "this" pointer area | `THIS` | `A = M + index` | `push this 1` → `*(THIS + 1)` |
| **that** | "that" pointer area | `THAT` | `A = M + index` | `pop that 2` → `*(THAT + 2)` |
| **temp** | Temporary storage (RAM[5–12]) | `5` | `A = 5 + index` | `push temp 3` → `@8` |
| **pointer** | Stores THIS and THAT pointers (RAM[3–4]) | `3` | `A = 3 + index` | `pop pointer 0` → `THIS = *(SP-1)` |
| **static** | File-specific static variables | `16` | `@FileName.index` | `push static 2` → `@Foo.2` |
| **constant** | Immediate values, not in RAM | — | `D = A` | `push constant 7` → `D = 7` |

### Basic Syntax of VM Language
```
push constant i
push segment i
pop segment i
add | sub | neg | eq | gt | lt | and | or | not
label X
goto X
if-goto X
function f k
call f n
return
```

### Common Translations

**add**
```
@SP
AM=M-1
D=M
A=A-1
M=M+D
```

**pop argument 0**
```
@SP
AM=M-1
D=M
@ARG
A=M
M=D
```

**push constant i**
```
@i
D=A
@SP
AM=M+1
A=A-1
M=D
```

**pop local i**
```
@i
D=A
@LCL
D=M+D
@5 // temp
M=D
@SP
AM=M-1
D=M
@5
A=M
M=D
```

**eq (boolean comparison)**
```
@SP
AM=M-1
D=M
@SP
AM=M-1
D=M-D
@labelTrue
D;JEQ
D=0
@labelFalse
0;JMP
(labelTrue)
D=-1
(labelFalse)
@SP
A=M
M=D
@SP
M=M+1
```

### Bootstrap
```
@256
D=A
@SP
M=D
@Sys.init
0;JMP
```

### Program Control
#### Subroutines & Functions
##### Stack Implementation
![vm-stack-implementation.png](https://images.zijianguo.com/vm-stack-implementation.png)

##### Call Implementation
![vm-call-command.png](https://images.zijianguo.com/vm-call-command.png)

##### Function Implementation
![vm-function-command.png](https://images.zijianguo.com/vm-function-command.png)

##### Return Implementation
![vm-return-command.png](https://images.zijianguo.com/vm-return-command.png)

### VM Notes
- The VM provides portability by hiding the Hack memory details.
- SP management ensures correct push/pop order.
- Always use unique labels for comparisons and calls.
- `call`: push return address and segment pointers, then jump to function.
- `return`: restore caller frame, reposition `SP`, and jump back to return address.

## High-Level Language (Jack)
Jack source compiles to VM commands, VM maps to Hack assembly.

### Segment Mapping

| Jack variable kind | VM segment | Notes |
|---|---|---|
| static | static | Per class file scope |
| field | this | Base in pointer 0 |
| var, local | local | Subroutine private |
| argument | argument | Call site provided |
| array base | this or local or argument | Depends on declaration |

### Subroutine Kinds

| Jack subroutine | VM header | Entry actions |
|---|---|---|
| function | function ClassName.func k | No implicit this |
| method | function ClassName.method k | push argument 0, pop pointer 0 |
| constructor | function ClassName.new k | push constant fieldCount, call Memory.alloc 1, pop pointer 0 |

### Statements, minimal templates
1. let x = expr
```
... code for expr
pop segment index        // x
```
2. let a[i] = expr
```
... code for a
... code for i
add
... code for expr
pop temp 0
pop pointer 1
push temp 0
pop that 0
```
3. y = a[i]
```
... code for a
... code for i
add
pop pointer 1
push that 0
... assign to y via pop segment index
```
4. do subCall(args)
```
... push object ref if method call
... push args
call QualName nArgs
pop temp 0               // discard return
```
5. if (cond) { S1 } else { S2 }
```
... code for cond
if-goto IF_TRUE$n
goto IF_FALSE$n
label IF_TRUE$n
... S1
goto IF_END$n
label IF_FALSE$n
... S2
label IF_END$n
```
6. while (cond) { S }
```
label WHILE_EXP$n
... code for cond
not
if-goto WHILE_END$n
... S
goto WHILE_EXP$n
label WHILE_END$n
```
7. return, return expr
```
push constant 0          // void
return
```
```
... code for expr
return
```

### Expressions, operators

| Jack | VM expansion |
|---|---|
| - x | neg |
| not x | not |
| x + y | add |
| x − y | sub |
| x & y | and |
| x \| y | or |
| x < y | lt |
| x > y | gt |
| x = y | eq |
| x * y | call Math.multiply 2 |
| x / y | call Math.divide 2 |

### Literals and keywords

| Jack | VM |
|---|---|
| integer n | push constant n |
| true | push constant 0, not |
| false, null | push constant 0 |
| this | push pointer 0 |

### String literal "abc"
```
push constant 3
call String.new 1
push constant 97
call String.appendChar 2
push constant 98
call String.appendChar 2
push constant 99
call String.appendChar 2
```
Length first, then append each code point.

### Calls, qualification

| Jack form | VM call name | Arg0 rule |
|---|---|---|
| obj.m(a,b) | ClassName.m | push obj reference then a,b |
| Class.f(a,b) | Class.f | no implicit object |
| m(a,b) inside a method | ClassName.m | push pointer 0 then a,b |

### Label policy
Use per subroutine counters, labels must be unique per function, for example `IF_TRUE$n`, `WHILE_END$n`.
