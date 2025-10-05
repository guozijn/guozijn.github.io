---
title: "Revisiting Nand2Tetris: Building a Computer from Scratch"
tags:
  - nand2tetris
  - computer system
---
## Introduction
In the age where computing systems are defined by abstraction layers and pre-built frameworks, the *Nand2Tetris* project, which was designed by Noam Nisan and Shimon Schocken, offers a rare opportunity to return to the foundations of computer science. This educational journey begins with the simplest possible logic gate, the NAND gate, and gradually guides learners toward constructing a fully functioning computer capable of running a high-level programming language and simple applications such as the game Tetris. By bridging hardware architecture, machine language, operating systems, and compiler design, Nand2Tetris provides an integrated understanding of how each layer of computing interacts to form a cohesive whole.

---

## Chapter 1: Boolean Logic

This chapter begins with **Boolean algebra** and the **NAND gate**, the universal logic gate from which all others can be constructed. Students implement basic gates such as NOT, AND, OR, and XOR, laying the foundation for digital computation.

```
A ----\                
       NAND ----> Output
B ----/
```

Through these constructions, learners understand how simple gates combine to form complex logical circuits.

---

## Chapter 2: Boolean Arithmetic

The second chapter focuses on **binary arithmetic**. Using logic gates, students build half-adders and full-adders, then chain them to construct multi-bit adders capable of handling binary numbers.

```
A ----\
       XOR ---- Sum
B ----/ \        \
        AND ---- Carry
```

This establishes how computers perform arithmetic at the hardware level.

---

## Chapter 3: Sequential Logic

Sequential logic introduces **state**—the ability to remember information. Using feedback loops and flip-flops, students design circuits that store data over time.

```
     +---------+
     |         |
Input ---> NAND ---> Output
       ^         |
       |_________|
```

These principles lead to the design of **registers** and **counters**, key elements of memory systems.

---

## Chapter 4: Machine Language

Here, students are introduced to the **Hack machine language**, the instruction set that the computer will eventually execute. They learn how the CPU interprets binary codes as instructions for computation and memory manipulation.

Example program:
```
@2
D=A
@3
D=D+A
@0
M=D
```

This simple example adds two numbers and stores the result in memory.

---

## Chapter 5: Computer Architecture

This chapter integrates earlier components into a complete **CPU**. Students combine the ALU (Arithmetic Logic Unit), registers, and program counter to build a central processing unit capable of running the Hack machine language.

```
Instruction --> Decoder --> Control Bits
Registers --> ALU --> Output + Flags
```

This marks the transition from circuit design to system-level computation.

---

## Chapter 6: Assembler

With the hardware ready, students build an **assembler** to translate Hack assembly language into binary machine code. The assembler resolves symbolic labels and memory variables into numeric addresses.

```
@LOOP  →  @16
@i     →  @17
```

This software bridge allows humans to program the hardware more effectively.

---

The **Virtual Machine (VM)** language introduces a stack-based computation model that abstracts hardware operations. It sits between the Jack high-level language and the Hack assembly language, providing a clean interface for arithmetic, logic, and memory commands.

All computations occur on a stack using `push` and `pop` instructions. Operands are pushed onto the stack, an operation (like `add` or `sub`) is performed, and the result is stored back on top.

Example:
```
push constant 7
push constant 8
add
```
Execution:
```
Stack: [7] → [7,8] → add → [15]
```

The VM defines memory segments that map to hardware:

| Segment | Purpose | Hack Mapping |
|----------|----------|--------------|
| constant | literal values | none |
| local | function locals | RAM[LCL] |
| argument | function args | RAM[ARG] |
| this/that | object refs | RAM[THIS]/RAM[THAT] |
| temp | temporary | RAM[5–12] |
| pointer | controls this/that | RAM[3–4] |
| static | global vars | RAM[16+] |

Arithmetic and logic commands include:
```
add, sub, neg, eq, gt, lt, and, or, not
```

Example translation:
```
// VM: add
@SP
AM=M-1
D=M
A=A-1
M=M+D
```

This translator layer introduces structured computation independent of physical memory layout and prepares for Chapter 8, which adds branching and function control.

---

## Chapter 8: Virtual Machine II — Program Control

Extending the VM, this chapter adds **program control** capabilities like branching, looping, and function calls. It demonstrates how higher-level logic is implemented atop a stack-based execution model.

```
function Main.fibonacci 0
push argument 0
push constant 2
lt
if-goto BASE_CASE
```

---

## Chapter 9: High-Level Language

Students are introduced to **Jack**, a simple, object-based language. Jack programs are compiled into VM code, showing the bridge from human-readable syntax to machine-executable logic.

Example:
```java
class Main {
  function void main() {
    do Output.printString("Hello, world!");
    return;
  }
}
```

---

## Chapter 10: Compiler I — Syntax Analysis

Here, the compiler is built. Students first construct a **syntax analyser** that parses Jack programs into structured representations (parse trees). This teaches the foundations of compiler design.

```
Jack Source → Tokeniser → Syntax Tree
```

---

## Chapter 11: Compiler II — Code Generation

In this chapter, students implement **code generation**, translating parsed Jack syntax into executable VM commands. This finalises the high-level language pipeline from source code to VM bytecode.

```
Jack → VM Code → Assembly → Binary → Execution
```

---

## Chapter 12: Operating System

Students write the **Jack operating system (OS)**, implementing essential libraries like `Math`, `Memory`, `String`, and `Array`. These provide higher-level abstractions that simplify application development.

```
+------------------+
| Application Code |
| OS Libraries     |
| VM + Compiler    |
| CPU + Memory     |
+------------------+
```

The OS marks the final layer of abstraction between hardware and user-level software.

---

## Chapter 13: Postscript — More Fun to Go

The book concludes with reflections on further exploration. Having built a full computer system—from hardware logic to operating system—students can now explore real-world architectures, programming languages, and computer science research topics.

---

## Conclusion

Nand2Tetris demystifies the complexity of computing by reconstructing it from first principles. It unifies hardware and software learning, empowering students to understand every layer of modern computation, from NAND gates to game programs.
