---
title: "Algorithm and Data Structure Analysis: Study Notes"
tags:
  - algorithm
  - data structure
---

## 1. Asymptotic Notations

### Definitions and Meanings

| Notation | Mathematical Definition | Description |
|-----------|--------------------------|--------------|
| **$$f(n) = O(g(n))$$** | $\exists c > 0, \exists n_0 > 0 \text{ such that } \forall n \ge n_0, \|f(n)\| \le c g(n)$ | Asymptotic upper bound |
| **$$f(n) = \Omega(g(n))$$** | $\exists c > 0, \exists n_0 > 0 \text{ such that } \forall n \ge n_0, \|f(n)\| \ge c g(n)$ | Asymptotic lower bound |
| **$$f(n) = \Theta(g(n))$$** | $\exists c_1, c_2 > 0, \exists n_0 > 0 \text{ such that } \forall n \ge n_0, c_1 g(n) \le \|f(n)\| \le c_2 g(n)$ | Asymptotic tight bound |
| **$$f(n) = o(g(n))$$** | $\forall c > 0, \exists n_0 > 0 \text{ such that } \forall n \ge n_0, \|f(n)\| \le c g(n)$ | Upper bound that is not asymptotically tight |
| **$$f(n) = \omega(g(n))$$** | $\forall c > 0, \exists n_0 > 0 \text{ such that } \forall n \ge n_0, \|f(n)\| \ge c g(n)$ | Lower bound that is not asymptotically tight |

Additionally, $f(n) = O(g(n))$ can also be written as $f(n) \in O(g(n))$.

### Limit-Based Characterisations

| Notation | Limit Definition | Interpretation |
|-----------|------------------|----------------|
| **$$f(n) = O(g(n))$$** | $\lim_{n \to \infty} \frac{f(n)}{g(n)} \ne \infty$ | $f(n)$ grows no faster than $g(n)$ |
| **$$f(n) = \Omega(g(n))$$** | $\lim_{n \to \infty} \frac{f(n)}{g(n)} \ne 0$ | $f(n)$ grows at least as fast as $g(n)$ |
| **$$f(n) = \Theta(g(n))$$** | $\lim_{n \to \infty} \frac{f(n)}{g(n)} \ne 0, \infty$ | $f(n)$ and $g(n)$ grow at the same rate |
| **$$f(n) = o(g(n))$$** | $\lim_{n \to \infty} \frac{f(n)}{g(n)} = 0$ | $f(n)$ grows strictly slower than $g(n)$ |
| **$$f(n) = \omega(g(n))$$** | $\lim_{n \to \infty} \frac{f(n)}{g(n)} = \infty$ | $f(n)$ grows strictly faster than $g(n)$ |
| **$$f(n) \sim g(n)$$** | $\lim_{n \to \infty} \frac{f(n)}{g(n)} = 1$ | $f(n)$ and $g(n)$ are asymptotically equivalent |

### Examples

* $\log(n)^{10} \in o(n)$
* $3n^2 + 1 \in \Theta(n^2)$
* $2.1n^{2.1} + 1500n^2 \in \Theta(n^{2.1})$

## 2. Integer Arithmetic

### School Method Multiplication

Given two $n$-digit integers $a$ and $b$ in base $B$, the classical school method computes $p = a \cdot b$ by iterating over the digits of $b$ from least to most significant and accumulating their contributions:

```text
p ← 0
for i = 0 to n − 1 do
    p ← p + a · b_i · B^i
end for
```

This direct method performs $3n^2 + 2n = \Theta(n^2)$ primitive operations.

### Recursive Multiplication

The straightforward divide-and-conquer algorithm splits each $n$-digit operand into high and low halves, solves four subproblems, then combines the partial products:

```text
function Multiply(a, b, n):
    if n == 1:
        return a * b

    k = ceil(n / 2)
    a1, a0 = split(a, k)
    b1, b0 = split(b, k)

    p0 = Multiply(a0, b0, k)
    p1 = Multiply(a1, b0, k)
    p2 = Multiply(a0, b1, k)
    p3 = Multiply(a1, b1, k)

    return p3 * B^(2k) + (p1 + p2) * B^k + p0
```

Here `split(x, k)` returns the high and low blocks of $x$ with respect to base $B$ and block size $k$. The effort satisfies $T(1) = 1$ and, for $n \ge 2$, $T(n) = 4T(\lceil n/2 \rceil) + 6n$, which resolves to $T(n) = \Theta(n^2)$.

### Karatsuba Multiplication

Karatsuba trims one of the four sub-multiplications and compensates with cheaper additions:

```text
function Karatsuba(a, b, n):
    if n < 4:
        return a * b

    k = ceil(n / 2)
    a1, a0 = split(a, k)
    b1, b0 = split(b, k)

    p0 = Karatsuba(a0, b0, k)
    p2 = Karatsuba(a1, b1, k)
    p1 = Karatsuba(a0 + a1, b0 + b1, k + 1)

    return p2 * B^(2k) + (p1 - p2 - p0) * B^k + p0
```

With $k = \lceil n/2 \rceil$, write $a = a_1 B^k + a_0$ and $b = b_1 B^k + b_0$. Then $ab = (a_1 b_1)B^{2k} + \big[(a_0 + a_1)(b_0 + b_1) - a_1 b_1 - a_0 b_0\big]B^k + a_0 b_0$,

so only three multiplications: $a_0 b_0$, $a_1 b_1$, and $(a_0 + a_1)(b_0 + b_1)$ are required, alongside six additions/subtractions of at most $2n$-digit numbers.

The sum operands can gain an extra digit, so the middle call uses size $k + 1$. The work therefore satisfies $T(n) = 3T(\lceil n/2 \rceil + 1) + 12n$ for $n \ge 4$ (with the base case bounded by $3n^2 + 2n$), and hence $T(n) = \Theta(n^{\log_2 3}) \approx \Theta(n^{1.585})$.

## 3. Recursion and Master Theorem

* **Master Theorem:** For $T(n) = aT(n/b) + f(n)$ with

  * $T(n)$ — total cost to solve the problem of size $n$
  * $a$ — number of subproblems produced at each level
  * $n/b$ — size of each recursive subproblem (assuming $b > 1$)
  * $f(n)$ — non-recursive work spent on splitting, combining, or other overhead per level

  * If $f(n) = O(n^{\log_b a - \epsilon})$, then $T(n) = \Theta(n^{\log_b a})$
  * If $f(n) = \Theta(n^{\log_b a})$, then $T(n) = \Theta(n^{\log_b a} \log n)$
  * If $f(n) = \Omega(n^{\log_b a + \epsilon})$, and regularity condition holds, then $T(n) = \Theta(f(n))$

**Example:**
$T(n) = 4T(n/2) + n^2$
$\Rightarrow a=4, b=2, f(n)=n^2$, $n^{\log_2 4} = n^2$ → Case 2 → $T(n) = \Theta(n^2 \log n)$

## 4. Linear Time Sorting

### Counting Sort

Counting sort assumes integer keys in the range $[0, k]$ and counts the frequency of each key before reconstructing the output in linear time:

```text
function CountingSort(A, B, k):
    for i = 0 to k:
        C[i] = 0
    for j = 1 to n:
        C[A[j]] = C[A[j]] + 1
    for i = 1 to k:
        C[i] = C[i] + C[i - 1]
    for j = n downto 1:
        B[C[A[j]]] = A[j]
        C[A[j]] = C[A[j]] - 1
```

Here $A[1..n]$ stores the input keys, $B[1..n]$ collects the stable output, and $C[0..k]$ maintains cumulative counts. The algorithm is stable and runs in $\Theta(n + k)$ time using $\Theta(n + k)$ extra space.

### Radix Sort

Radix sort applies a stable digit-based sort, such as counting sort, to each digit from least to most significant when keys have $d$ digits in base $B$:

```text
function RadixSort(A, d, B):
    for i = 0 to d - 1:
        A = CountingSortByDigit(A, i, B)
    return A
```

Here `CountingSortByDigit` sorts by the $i$-th digit only and preserves previous digit order. The overall cost is $\Theta(d (n + B))$, which becomes linear when $d$ and $B$ are bounded.

#### Comparison vs. Non-Comparison Sorts

| Sort | Category | Best | Average | Worst | Extra Space | Stable |
|------|-----------|------|---------|-------|-------------|--------|
| Bubble | Comparison | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(n^2)$ | $\Theta(1)$ | Yes |
| Selection | Comparison | $\Theta(n^2)$ | $\Theta(n^2)$ | $\Theta(n^2)$ | $\Theta(1)$ | No |
| Insertion | Comparison | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(n^2)$ | $\Theta(1)$ | Yes |
| Merge | Comparison | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(n)$ | Yes |
| Heap | Comparison | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(1)$ | No |
| Quick | Comparison | $\Theta(n \log n)$ | $\Theta(n \log n)$ | $\Theta(n^2)$ | $\Theta(\log n)$ | No |
| Counting | Non-comparison | $\Theta(n + k)$ | $\Theta(n + k)$ | $\Theta(n + k)$ | $\Theta(n + k)$ | Yes |
| Radix | Non-comparison | $\Theta(d (n + B))$ | $\Theta(d (n + B))$ | $\Theta(d (n + B))$ | $\Theta(n + B)$ | Yes |
| Bucket | Non-comparison | $\Theta(n)$ | $\Theta(n)$ | $\Theta(n^2)$ | $\Theta(n)$ | Yes |

## 5. Order Statistics

### Selection Problem

Given an unsorted array $A[1..n]$, the $i$-th order statistic is the element that would appear at index $i$ after sorting. A naive approach sorts $A$ in $\Theta(n \log n)$ time and returns the $i$-th entry.

### Randomized Selection

RandomizedSelect partitions around a random pivot and recurses into the relevant subarray, giving expected linear time:

```text
function RandomizedSelect(A, p, r, i):
    if p == r:
        return A[p]
    q = RandomizedPartition(A, p, r)
    k = q - p + 1
    if i == k:
        return A[q]
    if i < k:
        return RandomizedSelect(A, p, q - 1, i)
    return RandomizedSelect(A, q + 1, r, i - k)
```

Here $A[p..r]$ denotes the active subarray; $p$ and $r$ mark its inclusive bounds, and $i$ (1-indexed) is the desired order statistic within that subarray.

Expected complexity is $\Theta(n)$ with $\Theta(1)$ auxiliary space when the partition is performed in-place, though the worst case remains $\Theta(n^2)$ if pivot choices are consistently poor.

### Deterministic Linear-Time Selection

The median-of-medians algorithm chooses pivots deterministically to guarantee $\Theta(n)$ worst-case time by grouping elements, selecting medians recursively, and partitioning around that pivot.

## 6. Hashing

* **Chaining:** Insert in linked list — average $O(1)$, worst $O(n)$
* **Linear Probing:** $h'(k) = (h(k) + i) \bmod m$
* **Quadratic Probing:** $h'(k) = (h(k) + c_1 i + c_2 i^2) \bmod m$
* **Expected search cost:** $O(1 + \alpha)$ where $\alpha = n/m$
* **Deletion in open addressing:** worst case $O(n)$

```text
HashInsertChaining(T, key):
    index = h(key) mod m
    prepend key to linked list T[index]

HashSearchChaining(T, key):
    index = h(key) mod m
    for each node in T[index]:
        if node.key == key:
            return node
    return null

HashInsertLinearProbing(T, key):
    index = h(key) mod m
    while T[index] occupied:
        index = (index + 1) mod m
    T[index] = key

HashSearchLinearProbing(T, key):
    index = h(key) mod m
    while T[index] not empty:
        if T[index] == key:
            return T[index]
        index = (index + 1) mod m
    return null
```

Hash table with chaining:

| Operation | Worst Case | Average Case |
| --------- | ---------- | ------------ |
| Insert    | $O(1)$     | $O(1)$       |
| Remove    | $\Theta(n)$ | $O(1 + \alpha)$ |
| Find      | $\Theta(n)$ | $O(1 + \alpha)$ |

Hash table with linear probing:

| Operation | Worst Case | Average Case |
| --------- | ---------- | ------------ |
| Insert    | $\Theta(n)$ | $O(1)$       |
| Remove    | $\Theta(n)$ | $O(1 + \alpha)$ |
| Find      | $\Theta(n)$ | $O(1 + \alpha)$ |

Worst case analysis of data structures:

| Data Structure | Insert     | Remove     | Find      |
| -------------- | ---------- | ---------- | --------- |
| Array          | $O(n)$     | $O(n)$     | $O(1)$    |
| Linked List    | $O(1)$     | $O(1)$     | $\Theta(n)$ |
| AVL Tree       | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |

## 7. Skip Lists

* Expected height of an element: $H = 1/p$, typically $p = 1/2$
* Expected search/insert/delete: $O(\log n)$

## 8. Binary Search Trees

* **Properties:**

  * Left < Root < Right
  * In-order traversal gives sorted order

* **Average Search Cost Example:** $\frac{1}{n}\sum_{i=1}^{n} (\text{depth}(i)+1)$

* **Complexities:** Search, Insert, Delete = $O(h)$ where $h$ is tree height

  * Balanced BST: $h = O(\log n)$
  * Unbalanced BST: $h = O(n)$

### Core Operations

```text
Search(node, key):
    if node is null or node.key == key:
        return node
    if key < node.key:
        return Search(node.left, key)
    return Search(node.right, key)

Insert(node, key):
    if node is null:
        return new Node(key)
    if key < node.key:
        node.left = Insert(node.left, key)
    else:
        node.right = Insert(node.right, key)
    return node

Delete(node, key):
    if node is null:
        return null
    if key < node.key:
        node.left = Delete(node.left, key)
    else if key > node.key:
        node.right = Delete(node.right, key)
    else:
        if node.left is null:
            return node.right
        if node.right is null:
            return node.left
        successor = Minimum(node.right)
        node.key = successor.key
        node.right = Delete(node.right, successor.key)
    return node
```

### Example Tree

```text
        8
      /   \
     3    10
    / \     \
   1   6     14
      / \    /
     4   7  13
```

### Runtime Summary

For search, insert, and delete on a BST with height $h$:

* **Best case:** $\Theta(\log n)$ — balanced tree with $h = \Theta(\log n)$
* **Average case:** $\Theta(\log n)$ — random insertions keep the tree nearly balanced
* **Worst case:** $\Theta(n)$ — degenerate (e.g., sorted input) tree with $h = n - 1$

## 9. AVL Trees

* **Balance Factor:** $BF = height(left) - height(right)$, rebalancing keeps $BF \in \{-1,0,1\}$ for every node.
* **Rotations:**

  * LL → Single right rotation
  * RR → Single left rotation
  * LR → Left rotation + right rotation
  * RL → Right rotation + left rotation

### Insertion

```text
AVLInsert(node, key):
    if node is null:
        return new Node(key)
    if key < node.key:
        node.left = AVLInsert(node.left, key)
    else if key > node.key:
        node.right = AVLInsert(node.right, key)
    else:
        return node  // duplicate keys not inserted

    updateHeight(node)
    balance = height(node.left) - height(node.right)

    if balance > 1 and key < node.left.key:
        return rotateRight(node)          // LL case
    if balance < -1 and key > node.right.key:
        return rotateLeft(node)           // RR case
    if balance > 1 and key > node.left.key:
        node.left = rotateLeft(node.left) // LR case
        return rotateRight(node)
    if balance < -1 and key < node.right.key:
        node.right = rotateRight(node.right) // RL case
        return rotateLeft(node)

    return node
```

### Deletion

```text
AVLDelete(node, key):
    if node is null:
        return null
    if key < node.key:
        node.left = AVLDelete(node.left, key)
    else if key > node.key:
        node.right = AVLDelete(node.right, key)
    else:
        if node.left is null:
            return node.right
        if node.right is null:
            return node.left
        successor = Minimum(node.right)
        node.key = successor.key
        node.right = AVLDelete(node.right, successor.key)

    updateHeight(node)
    balance = height(node.left) - height(node.right)

    if balance > 1 and BF(node.left) >= 0:
        return rotateRight(node)
    if balance > 1 and BF(node.left) < 0:
        node.left = rotateLeft(node.left)
        return rotateRight(node)
    if balance < -1 and BF(node.right) <= 0:
        return rotateLeft(node)
    if balance < -1 and BF(node.right) > 0:
        node.right = rotateRight(node.right)
        return rotateLeft(node)

    return node
```

### Rotations

```text
rotateLeft(x):
    y = x.right
    x.right = y.left
    y.left = x
    updateHeight(x)
    updateHeight(y)
    return y

rotateRight(y):
    x = y.left
    y.left = x.right
    x.right = y
    updateHeight(y)
    updateHeight(x)
    return x
```

#### Rotation Examples During Updates

```text
Insertion causing LL case (insert 5 near root 30):

Before insertion (balanced):     After insertion (unbalanced):

      30                               30
     /  \                             /  \
   20    40       insert 5 →        20    40
  / \                              / \
10  25                           10  25
                                /
                               5

After right rotation on 30:

      20
     /  \
   10    30
  /     / \
 5    25  40

Insertion causing LR case (insert 22 near root 30):

Before insertion (balanced):     After insertion (unbalanced):

      30                               30
     /  \                             /  \
   20    40       insert 22 →       20    40
  / \                              /  \
18  25                           18   25
                                     /
                                   22

After left rotation on 20:          After right rotation on 30:

      30                                      25
     /  \                                    /  \
    25   40                                 20   30
   /                                       / \     \
  20                                     18  22     40
 / \
18 22

Deletion causing RR case (delete 10 from tree rooted at 20):

Before deletion (balanced):       After deletion (unbalanced):

    20                                 20
   /  \         delete 10 →             \
 10   30                                30
     /  \                              /  \
   25   40                           25   40

After left rotation on 20:

      30
     /  \
   20    40
     \
     25

Deletion causing RL case (delete 5 from tree rooted at 20):

Before deletion (balanced):       After deletion (unbalanced):

     20                                  20
    /  \          delete 5 →            /  \
   15   40                            15    40
  /     /                                  /
 5     30                                30
        \                                 \
        35                                35

After right rotation on 40:         After left rotation on 20:

     20                                      30
    /  \                                    /  \
   15   30                                20    40
         \                               /      /
         40                            15     35
        /
       35
```

### Example AVL Tree

```text
        30
      /    \
    20      40
   /  \       \
 10  25       50
```

Each node’s balance factor stays within $[-1,1]$, and rotations repair any violations after insertions or deletions.

## 10. Graph Representations

| Representation   | Space    | Description   |
| ---------------- | -------- | ------------- |
| Adjacency Matrix | $O(n^2)$ | Dense graphs  |
| Adjacency List   | $O(n+m)$ | Sparse graphs |

## 11. Graph Traversal Algorithms

### Depth-First Search (DFS)

```text
DFS(G):
    visited[v] = false for all v ∈ G.V
    for each s ∈ G.V:
        if visited[s] == false:
            visited[s] = true
            parent[s] = s                      // new DFS tree root
            DFS_Explore(s, s)

DFS_Explore(u, v):
    for each (v, w) ∈ G.E:
        if visited[w] == true:
            traverseNonTreeEdge(v, w)          // w was reached before
        else:
            traverseTreeEdge(v, w)             // w was not reached before
            visited[w] = true
            parent[w] = v
            DFS_Explore(v, w)
    backtrack(u, v)                            // return from v along the incoming edge
```

* Complexity: $O(V + E)$

### Breadth-First Search (BFS)

```text
BFS(G, s):
    dist[v] = ∞ for all v ∈ G.V            // distance from s
    parent[v] = NIL for all v ∈ G.V         // BFS tree parent
    dist[s] = 0
    parent[s] = s                           // root self-loop
    Q = {s}                                 // current frontier
    Q_next = ∅                              // next frontier
    for level = 0 while Q ≠ ∅:
        for each u in Q:
            for each (u, v) ∈ G.E:          // scan outgoing edges
                if parent[v] == NIL:        // discovered new vertex
                    Q_next = Q_next ∪ {v}
                    dist[v] = level + 1
                    parent[v] = u
        (Q, Q_next) = (Q_next, ∅)           // advance to next layer
    return (dist, parent)
```

* For unweighted graphs → shortest path

## 12. Shortest Path Algorithms

### Dijkstra’s Algorithm

```text
Dijkstra(G, s):
    dist[v] = ∞ for all v ∈ G.V                    // tentative distances
    parent[v] = NIL for all v ∈ G.V                 // shortest-path tree
    dist[s] = 0
    parent[s] = s                                   // root self-loop
    Q = empty priority queue
    Q.insert(s, key = 0)
    while Q ≠ ∅:
        u = Q.deleteMin()                           // settle closest vertex
        for each edge (u, v) ∈ G.E:                 // relax outgoing edges
            if dist[v] > dist[u] + w(u, v):
                dist[v] = dist[u] + w(u, v)
                parent[v] = u
                if v in Q:
                    Q.decreaseKey(v, dist[v])
                else:
                    Q.insert(v, key = dist[v])
    return (dist, parent)
```

Example relaxation step:

```text
Current shortest frontier:

    s --(2)--> u --(3)--> v

dist[s] = 0, dist[u] = 2, dist[v] = ∞

When u is extracted, relaxing (u, v) sets:
    dist[v] = dist[u] + 3 = 5
    parent[v] = u
The updated key 5 is pushed (or decreased) in Q.
```

* **Complexity:** $O((V + E)\log V)$ using priority queue
* **Cannot handle negative edges**

### Bellman-Ford Algorithm

* Handles negative weights but **not negative cycles**

```text
BellmanFord(G, s):
    dist[v] = ∞ for all v ∈ G.V
    parent[v] = NIL for all v ∈ G.V
    dist[s] = 0
    parent[s] = s
    for i = 1 to |G.V| - 1:
        for each edge (u, v) ∈ G.E:
            if dist[u] + w(u, v) < dist[v]:
                dist[v] = dist[u] + w(u, v)
                parent[v] = u
    for each edge (u, v) ∈ G.E:
        if dist[u] + w(u, v) < dist[v]:
            markNegativeCycle(v)
    return (dist, parent)

markNegativeCycle(v):
    if dist[v] > -∞:
        dist[v] = -∞
        for each edge (v, w) ∈ G.E:
            markNegativeCycle(w)
```

### Floyd-Warshall Algorithm

```text
FloydWarshall(G):
    dist[i][j] = ∞ for all i, j ∈ G.V
    for each edge (u, v) ∈ G.E:
        dist[u][v] = w(u, v)                    // initialize weights
    for each vertex v ∈ G.V:
        dist[v][v] = 0
    for k in G.V:
        for i in G.V:
            for j in G.V:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
```

Example adjacency matrix input:

```text
Vertices: {A, B, C}

    A   B   C
A [ 0,  4, ∞ ]
B [ ∞,  0,  1 ]
C [ 3, ∞,  0 ]

Output dist matrix gives all-pairs shortest paths.
```

* **Complexity:** $O(\|V\|^3)$
* Solves the all-pairs shortest-path problem on weighted directed graphs, handling negative edge weights as long as there is no negative cycle.
* Works with negative edges but not negative cycles

| Algorithm | Paradigm | Core idea |
|-----------|----------|-----------|
| Dijkstra | Greedy | Extract the closest unsettled vertex and relax outgoing edges |
| Bellman-Ford | Dynamic programming | Relax edges in rounds, increasing the maximum path length |
| Floyd-Warshall | Dynamic programming | Incrementally allow more intermediate vertices to update all-pairs distances |
| Prim | Greedy | Grow the tree from a seed by picking the lightest crossing edge |
| Kruskal | Greedy | Sort edges globally, add the lightest edge that joins two components |
| BFS | Search | Explore layer by layer; breadth gives unweighted shortest paths |
| DFS | Search | Dive deeply along a branch, backtrack to explore unseen edges |

## 13. Minimum Spanning Trees (MST)

### Key Properties

An MST of a connected, weighted, undirected graph can be constructed with greedy algorithms (such as Prim and Kruskal) thanks to two structural guarantees:

* **Cut property:** For any cut of the graph, let $e$ be the minimum-weight edge that crosses the cut. There always exists an MST that includes $e$, so choosing such safe edges never blocks an optimal tree.
* **Cycle property:** For any cycle, let $e$ be the maximum-weight edge on that cycle. Some MST excludes $e$, which means heaviest cycle edges can be discarded without losing optimality.

### Prim’s Algorithm

```text
Prim(G):
    choose any start vertex s
    MST = {}
    while not all vertices in MST:
        choose edge (u,v) with minimum weight where u in MST, v not in MST
        add (u,v) to MST
```

* **Complexity:** $O(E\log V)$ with heap

```text
function PrimMST(G):
    for each vertex v in G.V:
        dist[v] = ∞
        parent[v] = null
    choose arbitrary start s
    dist[s] = 0
    Q = priority queue of all vertices keyed by dist

    while Q not empty:
        u = Q.deleteMin()
        for each edge (u, v) in G.E:
            if v in Q and w(u, v) < dist[v]:
                dist[v] = w(u, v)
                parent[v] = u
                Q.decreaseKey(v, dist[v])

    return {(v, parent[v]) : parent[v] != null}
```

### Kruskal’s Algorithm

```text
Kruskal(G):
    sort edges by weight
    for each edge (u,v):
        if Find(u) != Find(v):
            Union(u,v)
            add edge (u,v) to MST
```

* **Complexity:** $O(E\log E)$
* **Implementation note:** Kruskal relies on a disjoint-set structure to detect cycles efficiently. Each edge check performs `Find` on its endpoints and `Union` on accepted edges; the pseudocode in the Union-Find section below shows a typical implementation.

### Prim vs. Kruskal

- Both algorithms are greedy MST builders, but Prim grows a single tree from an arbitrary seed, always picking the lightest edge that spans the cut between the tree and the remaining vertices; with a heap this yields $O(E \log V)$ time.
- Kruskal instead sorts all edges globally and keeps adding the lightest edge that connects two different components, relying on Union-Find to avoid cycles, which leads to $O(E \log E)$ complexity.
- Prim fits dense graphs or adjacency-matrix setups (fewer edge weight comparisons), whereas Kruskal shines with sparse graphs where the edge sort dominates the work.

## 12. Disjoint Set Union (Union-Find)

* **Operations:**

  * Find: returns representative
  * Union: merges two sets
  * Path compression + union by rank → $O(\alpha(n))$

```text
class UnionFind(n):
    parent[i] = i for i in 1..n
    rank[i] = 0 for i in 1..n

    function Find(i):
        if parent[i] != i:
            parent[i] = Find(parent[i])      // path compression
        return parent[i]

    procedure Link(i, j):
        if rank[i] < rank[j]:
            parent[i] = j
        else if rank[i] > rank[j]:
            parent[j] = i
        else:
            parent[j] = i
            rank[i] = rank[i] + 1

    procedure Union(i, j):
        ri = Find(i)
        rj = Find(j)
        if ri != rj:
            Link(ri, rj)
```

## 13. P, NP, and NP-Completeness

* **P:** Problems solvable in polynomial time
* **NP:** Problems verifiable in polynomial time
* **NP-Complete:** In NP and as hard as any in NP

| Problem | Category | Notes |
|---------|----------|-------|
| Single-source shortest path (Dijkstra on non-negative weights) | P | Polynomial-time solvable via greedy relaxation. |
| Minimum spanning tree | P | Solvable in polynomial time by Prim or Kruskal. |
| Hamiltonian Cycle Problem (HCP) | NP-Complete | Does a graph contain a cycle visiting every vertex exactly once? |
| Traveling Salesman Problem (decision) | NP-Complete | Is there a tour ≤ given cost? Optimization version is NP-hard. |
| Graph Coloring (k ≥ 3) | NP-Complete | Can you color vertices with k colors so adjacent ones differ? |
| Clique Problem | NP-Complete | Does the graph contain a complete subgraph of size k? |
| Boolean Satisfiability (SAT) | NP-Complete | First NP-complete problem (Cook-Levin). |
| Subset Sum | NP-Complete | Select numbers whose sum matches a target. |
| Multi-objective Minimum Spanning Tree | NP-Hard | Optimizes multiple criteria simultaneously; conflicts make it NP-hard. |

**MST is in P:** solvable by Kruskal or Prim in polynomial time.

## 14. Typical Exam Problem Types

1. **True/False** — complexity and correctness
2. **Induction proofs** — summations, correctness of algorithms
3. **Recurrence solving** — apply Master theorem
4. **Hashing exercises** — show table after insertions
5. **AVL rotations** — show before/after balancing
6. **DFS/BFS pseudo-code** — write and trace
7. **Dijkstra/Bellman-Ford** — show step-by-step distances
8. **MST construction** — find all MSTs or use Prim/Kruskal
9. **P vs NP conceptual** — definitions, examples, classification

## 15. Key Formulae Summary

* **Geometric Sum:** $1 + r + r^2 + ... + r^n = \frac{r^{n+1} - 1}{r - 1}$
* **Induction Base and Step Example:**

  * Prove $2^0 + 2^1 + ... + 2^n = 2^{n+1} - 1$
  * Base: $n=0$, holds
  * Step: Assume true for $k$, show for $k+1$:
    $2^{k+1}-1 + 2^{k+1} = 2^{k+2}-1$
