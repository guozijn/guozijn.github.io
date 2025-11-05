---
title: "Statistical Machine Learning: Study Notes"
tags:
  - machine learning
---

## High-Yield Comparisons
### Supervised vs. Unsupervised vs. Semi-Supervised
- Supervised learning maps inputs to targets using labelled pairs $(x, y)$; relies on empirical risk $R_{\text{emp}}(f)$ as an estimator of population risk and evaluates with metrics such as accuracy, F1, RMSE.
- Unsupervised learning discovers latent structure without labels: clustering, density estimation, dimensionality reduction, anomaly detection; objectives include maximising likelihood $p(x \mid \theta)$ or minimising reconstruction error.
- Semi-supervised learning blends both: propagate limited labels across unlabelled samples via graph regularisation or confident pseudo-labels; exploits cluster/low-density separation assumptions.

### Regression vs. Classification vs. Clustering
- Regression outputs continuous responses; losses include MSE/MAE; metrics such as $R^2$, RMSE; sensitive to outliers and assumes a noise model.
- Classification outputs discrete classes; uses cross-entropy, hinge losses; metrics include accuracy, precision/recall, AUROC; decision boundaries capture separability.
- Clustering partitions data without labels; optimises intra-cluster similarity vs. inter-cluster separation; evaluation uses internal metrics (silhouette, inertia) or external (ARI) when ground truth exists.

### Overfitting vs. Underfitting vs. Well-Specified
- Overfitting: low training error, high test error; variance-dominated; symptoms include complex hypotheses, noisy gradients, validation degradation; mitigated with regularisation, early stopping, dropout, data augmentation.
- Underfitting: high bias, high training and test error; arises from overly simple models or poor feature design; resolved by richer architectures, longer training, reduced regularisation.
- Well-specified: balanced bias/variance; training and validation curves converge; residual error approximates irreducible noise.

### Generative vs. Discriminative Models
- Generative models estimate $p(x, y)$ or $p(x \mid y)$ (Naive Bayes, GMMs, VAEs); enable sampling, handle missing data, provide class priors explicitly.
- Discriminative models learn $p(y \mid x)$ or direct decision boundaries (logistic regression, SVMs, neural nets); often achieve lower asymptotic error with enough labelled data but lack generative capabilities.
- Exam hook: contrast assumptions (conditional independence vs. margin maximisation) and training objectives (likelihood maximisation vs. loss minimisation).

### Parametric vs. Non-Parametric Methods
- Parametric: fixed number of parameters (linear/logistic regression, neural nets with fixed width); fast inference, easier regularisation, higher bias if model mis-specified.
- Non-parametric: complexity grows with data (KNN, kernel density, Gaussian processes); flexible decision surfaces, slower inference, require careful distance/kernel design.
- Bias–variance intuition: parametric $\rightarrow$ higher bias/lower variance; non-parametric $\rightarrow$ lower bias/higher variance.

### Optimization Playbook
- Gradient-based methods (GD, SGD, Adam) require differentiable losses; sensitive to learning rates and conditioning; use momentum, adaptive step sizes, normalisation.
- Expectation–Maximisation alternates latent-variable inference (E-step) with parameter maximisation (M-step); guarantees non-decreasing likelihood but may converge to local optima.
- Coordinate descent/proximal methods tackle non-smooth penalties (L1, elastic net) via closed-form soft-thresholding updates; useful for sparse models.

## Module 1 – The Landscape of AI & Mathematical Bases for ML
### Core Concepts
- Evolution chain: Symbolic AI → Machine Learning → Deep Learning → Agentic AI; each shift driven by richer data and compute enabling stronger optimization.
- Progress triangle: data (quality, volume), compute (hardware, parallelism), optimization (algorithms, tuning).

### Linear Algebra Quick Reference
- Vectors and matrices: $Ax = b$, $x = A^{-1} b$ when invertible; use SVD $A = U \Sigma V^{\top}$ for stable solutions.
- Eigenvalues: solve $\det(A - \lambda I) = 0$; diagonalization $A = Q \Lambda Q^{-1}$ when eigenvectors span $\mathbb{R}^n$.
- Covariance matrix $\Sigma = \tfrac{1}{n} \sum_i (x_i - \mu)(x_i - \mu)^{\top}$; symmetric positive semi-definite.

### Probability & Statistics Essentials
- Expectation $\mathbb{E}[X] = \sum_x x \, p(x)$ or $\int x f(x) \, dx$; linear in X.
- Variance $\operatorname{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$.
- Covariance $\operatorname{Cov}(X,Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$.
- Bayes’ rule $p(\theta \mid x) = \frac{p(x \mid \theta) p(\theta)}{p(x)}$.

### Optimization Basics
- Gradient of squared loss $L = \tfrac{1}{n} \sum_i (y_i - w^{\top} x_i)^2$: $\nabla_w L = -\tfrac{2}{n} \sum_i x_i (y_i - w^{\top} x_i)$.
- Directional derivative along $d$: $\nabla L \cdot d$.
- Convexity check: Hessian $\succeq 0$; Jensen’s inequality for expectations.
- Gradient descent update: $w \leftarrow w - \eta \nabla L$; ensure step size respects Lipschitz constant.

## Formula Table (based on SML Slides)

| Topic | Formula | Notes |
| - | - | - |
| Summation–Product Identity | $\sum_{i=1}^M \sum_{j=1}^N a_i b_j = \Big(\sum_{i=1}^M a_i\Big)\Big(\sum_{j=1}^N b_j\Big)$ | From "Summation and Product" slide; shows double sum factorisation. |
| Matrix–Vector Product | $Ab = [a_1^{\top} b, \dots, a_n^{\top} b]^{\top}$ | Treat rows $a_i^{\top}$ of $A$ as vectors whose dot-products with $b$ form the result. |
| Diagonal Scaling | $A\Lambda = [a_1,\dots,a_n] \operatorname{diag}(\lambda_1,\dots,\lambda_n) = [\lambda_1 a_1,\dots,\lambda_n a_n]$ | Scaling each column vector $a_i$ by its associated eigenvalue. |
| Inner Product | $\langle x, y \rangle = x^{\top} y = \sum_i x_i y_i$ | Definition of vector inner product. |
| Vector Norms | $\|x\|_2 = \sqrt{\sum_i x_i^2}$<br>$\|x\|_1 = \sum_i \|x_i\|$<br>$\|x\|_p = \big(\sum_i \|x_i\|^p\big)^{1/p}$ | Common norms listed on "Inner product and norms" slide. |
| Trace Basics | $\operatorname{Tr}(A) = \sum_i a_{ii}$, $\operatorname{Tr}(a) = a$ | Definition for square matrices and scalars. |
| Trace Identities | $\operatorname{Tr}(X^{\top} Y) = \operatorname{Tr}(XY^{\top}) = \operatorname{Tr}(Y^{\top} X) = \operatorname{Tr}(YX^{\top})$ | Cyclic invariance of the trace used in slides. |
| Frobenius Norm | $\lVert A\rVert_F=\sqrt{\sum_{i,j}a_{ij}^{2}}=\sqrt{\operatorname{Tr}!\big(AA^{\top}\big)}=\sqrt{\operatorname{Tr}!\big(A^{\top}A\big)}$ | Connects Frobenius norm to the trace operator. |
| Linear Subspace | $\mathcal{S} = \{ x \mid x = t_1 v_1 + t_2 v_2 + \cdots + t_k v_k \}$ | All linear combinations of basis vectors span the subspace. |
| Orthogonal Basis | $\langle v_i, v_j \rangle = 0,~ \forall i \neq j$ | Condition for orthogonality in the basis set. |
| Eigenvalue Equation | $Au = \lambda u$ | Defines eigenpairs $(\lambda, u)$ of matrix $A$. |
| Eigen Decomposition | $A = Q\Lambda Q^{-1}$<br>Symmetric case: $A = Q \Lambda Q^{\top}$, $Q^{\top} Q = QQ^{\top} = I$ | Factorisations highlighted in "Matrix decomposition" slide. |
| Polynomial Kernel | $K(x,x') = (c + x^{\top} x')^d$<br>$d=2:~K(x,x') = \sum_i (x_i^2)(x_i'^2) + \sum_{i>j} (\sqrt{2} x_i x_j)(\sqrt{2} x_i' x_j') + \sum_i (\sqrt{2c} x_i)(\sqrt{2c} x_i') + c^2$ | Expansion shows explicit feature map for quadratic kernel. |
| Euclidean & Mahalanobis Distance | $d_2(a,b) = \|a-b\|_2$<br>$d_M(a,b) = \sqrt{(a-b)^{\top} M (a-b)}$ | Distances used on "Distance measurement" slide. |
| Expectation & Variance | $\mathbb{E}[X] = \sum_x x\,p(x)$ or $\int x f(x)\,dx$<br>$\operatorname{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | Basics from expectation/variance slides. |
| Covariance Matrix | $\Sigma = \frac{1}{n}\sum_{i=1}^n (x_i - \mu)(x_i - \mu)^{\top}$ | Appears in covariance discussion for latent variable / EM sections. |
| Bayes Rule | $p(y\mid x) = \frac{p(x\mid y) p(y)}{p(x)}$ | Derived in Bayesian classification slide. |
| Linear Regression (Normal Equation) | $\hat{w} = \arg\min_w \sum_{i=1}^N (w^{\top} x_i - y_i)^2 = (X^{\top} X)^{-1} X^{\top} y$ | Closed-form solution from regression section. |
| Ridge Regression | $\mathcal{L}(w) = \sum_{i=1}^N (w^{\top} x_i - y_i)^2 + \lambda \|w\|_2^2$<br>$\hat{w} = (X^{\top} X + \lambda I)^{-1} X^{\top} y$ | Regularised regression objective and solution. |
| P-Norm Illustration | $\lVert x\rVert_{0}=\lvert{i:x_{i}\neq0}\rvert,\quad \lVert x\rVert_{1}=\sum_{i}\lvert x_{i}\rvert,\quad \lVert x\rVert_{2}=\big(\sum_{i}x_{i}^{2}\big)^{1/2},\quad \lVert x\rVert_{\infty}=\max_{i}\lvert x_{i}\rvert$ | Highlights how different $p$ values shape the norm. |
| Gaussian Mixture Model | $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ | Likelihood decomposition used by EM. |
| EM Responsibilities | $\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$ | E-step posterior of latent component. |
| Log-Likelihood for EM | $\mathcal{L}(\theta) = \sum_{i=1}^N \log \sum_{k=1}^K \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$ | Objective maximised in EM algorithm. |
| K-Means Objective | $\min_{\{c_i\},\{\mu_k\}} \sum_{i=1}^N \|x_i - \mu_{c_i}\|_2^2$ | Shown on K-means slides. |
| MDP Return & Value | $R_{t}=\sum_{k=0}^{\infty}\gamma^{k}r_{t+k},\quad V^{\pi}(s)=\mathbb{E}{\pi}[R{t}\mid s_{t}=s],\quad Q^{\pi}(s,a)=\mathbb{E}{\pi}[R{t}\mid s_{t}=s,a_{t}=a]$ | Definitions from reinforcement learning module. |
| Bellman Expectation | $V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a) [r(s,a) + \gamma V^\pi(s')]$ | Policy evaluation equation. |
| Bellman Optimality | $V^{}(s)=\max_{a}\sum_{s'}P(s'\mid s,a),[r(s,a)+\gamma V^{}(s')],\quad Q^{}(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^{}(s',a')$ | Optimal value function recursions. |
| Policy Improvement | $\pi'(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)[r(s,a) + \gamma V^\pi(s')]$ | Step in policy iteration slides. |
| Q-Learning Update | $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \big[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t,a_t)\big]$ | Incremental update rule from RL section. |
{: .formula-table }

### Key Properties & Exam Hooks
- Symmetric matrices admit orthonormal eigenbases; positive-definite $\Rightarrow$ convex quadratic forms—useful for proving optimisation objectives are convex.
- SVD guarantees rank-revealing factorisation even for rectangular matrices; singular values non-negative and sorted.
- Expectation is linear even when random variables are dependent; variance adds only for independent variables.
- Bayes’ rule combines likelihood and prior; normalising constant $p(x)$ can be expanded via law of total probability.
- Gradient questions (e.g. practice exam Q1) test chain rule with vector calculus; remember $\nabla_w (w^{\top} x) = x$ and matrix calculus identities.
- Typical trick exams: compare growth limits of data vs. model capacity, reason about scaling laws (data, compute, algorithms).

## Module 2 – Supervised Learning
### Core Concepts
- Regression predicts continuous $y$; classification predicts discrete labels.
- Empirical Risk Minimization (ERM): choose $f$ minimizing $R_{\text{emp}}(f) = \tfrac{1}{n} \sum_i \ell(f(x_i), y_i)$ as proxy for true risk.
- Bias–variance trade-off: $\mathbb{E}[(\hat{y} - y)^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$.
- Overfitting ↔ high variance; underfitting ↔ high bias (per practice exam Q2).

### Model Selection & Regularisation
- Validation set: assess generalization; $k$-fold CV averages validation over folds for stability.
- Ridge (L2): add $\lambda \lVert w \rVert_2^2$; shrink weights uniformly. Lasso (L1): $\lambda \lVert w \rVert_1$; induces sparsity.
- Early stopping: monitor validation loss; dropouts regularize by stochastic subnet sampling.

### Algorithms & Gradients
- Linear classifier: decision boundary from $w^{\top} x + b = 0$; logistic regression loss gradient $\nabla_w = \sum_i (\sigma(z_i) - y_i) x_i$.
- Gradient update practice: compute per-sample or minibatch; check signs carefully.
- KNN: lazily stores data; prediction via majority vote or averaged value among $k$ nearest by chosen distance (practice exam Q3).
- Weight decay equivalent to L2 regularization in gradient-based optimizers.

### Key Properties & Exam Hooks
- Supervised setup assumes i.i.d. labelled pairs; stationarity violations (concept drift) require re-weighting or online learning.
- Classification vs. regression: different loss curvature (logistic convex, hinge piecewise-linear, squared loss sensitive to outliers).
- Bias–variance diagnostics: plot train/test curves; high variance when curves diverge, high bias when both plateau high.
- Regularisation effects: L2 shrinks weights smoothly, L1 induces sparsity via soft-thresholding, dropout approximates model averaging; early stopping behaves like $L2$ in linear models.
- KNN properties: $k=1$ zero training error but high variance; higher $k$ smooths decision boundary; distance scaling requires feature normalisation.
- Model selection dialogue: validation set prevents optimistic bias; $k$-fold CV reduces variance of estimate; nested CV for unbiased model comparisons.
- Potential exam traps: relate ERM to structural risk minimisation (add capacity penalty) and discuss when empirical minimiser overfits.

## Module 3 – SVM / Kernel Methods
### Margin Geometry
- Hard-margin objective: minimize $\tfrac{1}{2} \lVert w \rVert^2$ subject to $y_i (w^{\top} x_i + b) \ge 1$; margin $\gamma = 1/\lVert w \rVert$.
- Soft-margin introduces slack $\xi_i$ with penalty $C \sum_i \xi_i$.

### Dual Formulation & KKT
- Lagrangian $\mathcal{L} = \tfrac{1}{2} \lVert w \rVert^2 - \sum_i \alpha_i [y_i (w^{\top} x_i + b) - 1]$; stationarity gives $w = \sum_i \alpha_i y_i x_i$.
- Dual problem: maximize $\sum_i \alpha_i - \tfrac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^{\top} x_j)$ subject to $0 \le \alpha_i \le C$ and $\sum_i \alpha_i y_i = 0$.
- KKT conditions: complementary slackness $\alpha_i [y_i (w^{\top} x_i + b) - 1] = 0$ guides support vector identification.

#### Hard-Margin Derivation Walkthrough
1. Start from the separable primal:
   $$
   \min_{w, b}\; \tfrac{1}{2}\lVert w \rVert^2 \quad \text{s.t.}\quad y_i (w^{\top} x_i + b) \ge 1.
   $$
   Introduce Lagrange multipliers $\alpha_i \ge 0$ for each constraint.
2. Form the Lagrangian
   $$
   \mathcal{L}(w, b, \alpha) = \tfrac{1}{2}\lVert w \rVert^2 - \sum_i \alpha_i \big[y_i (w^{\top} x_i + b) - 1\big].
   $$
   Rearranging highlights the linear dependence on $w$ and $b$.
3. Enforce stationarity (set partial derivatives to zero):
   $$
   \frac{\partial \mathcal{L}}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i,
   $$
   $$
   \frac{\partial \mathcal{L}}{\partial b} = - \sum_i \alpha_i y_i = 0 \Rightarrow \sum_i \alpha_i y_i = 0.
   $$
4. Substitute $w$ back into $\mathcal{L}$ to eliminate primal variables:
   $$
   \mathcal{L}(\alpha) = \sum_i \alpha_i - \tfrac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^{\top} x_j),
   $$
   which depends only on the multipliers; constraints collapse to $\alpha_i \ge 0$ and $\sum_i \alpha_i y_i = 0$.
5. Read off the dual: maximise $\mathcal{L}(\alpha)$ under those constraints. Complementary slackness
   $$
   \alpha_i \big[y_i (w^{\top} x_i + b) - 1\big] = 0
   $$
   identifies support vectors: only points exactly on the margin ($y_i (w^{\top} x_i + b) = 1$) can have $\alpha_i > 0$.

#### Soft-Margin Extension
1. Introduce slack variables to absorb violations:
   $$
   \min_{w, b, \xi}\; \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i \quad \text{s.t.}\quad y_i (w^{\top} x_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0.
   $$
   Two constraint families require multipliers: $\alpha_i \ge 0$ for the margin and $\mu_i \ge 0$ for $\xi_i \ge 0$.
2. Lagrangian now includes both sets:
   $$
   \mathcal{L}(w, b, \xi, \alpha, \mu) = \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i - \sum_i \alpha_i\big[y_i(w^{\top}x_i + b) - 1 + \xi_i\big] - \sum_i \mu_i \xi_i.
   $$
3. Stationarity gives familiar structure plus a new upper bound:
   $$
   \frac{\partial \mathcal{L}}{\partial w} = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i,\qquad \frac{\partial \mathcal{L}}{\partial b} = 0 \Rightarrow \sum_i \alpha_i y_i = 0,
   $$
   $$
   \frac{\partial \mathcal{L}}{\partial \xi_i} = 0 \Rightarrow C - \alpha_i - \mu_i = 0.
   $$
   The last line implies $0 \le \alpha_i \le C$ because $\mu_i \ge 0$.
4. Eliminating $w$, $b$, $\xi$ leaves the dual objective identical in form to hard margin but with the new box constraints:
   $$
   \max_{\alpha}\; \sum_i \alpha_i - \tfrac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j (x_i^{\top} x_j) \quad \text{s.t.}\quad 0 \le \alpha_i \le C,\; \sum_i \alpha_i y_i = 0.
   $$
5. Complementary slackness splits cases:
   $$
   \alpha_i\big[y_i(w^{\top}x_i + b) - 1 + \xi_i\big] = 0, \qquad \mu_i \xi_i = 0,
   $$
   so support vectors with $0 < \alpha_i < C$ lie exactly on the margin ($\xi_i = 0$), points with $\alpha_i = C$ incur slack ($\xi_i > 0$), and non-support points have $\alpha_i = 0$.

### Kernel Trick
- Replace dot products with kernel $K(x_i, x_j) = \phi(x_i)^{\top} \phi(x_j)$ to work in feature space without explicit mapping.
- Common kernels: linear $x_i^{\top} x_j$; polynomial $(x_i^{\top} x_j + c)^d$; RBF $\exp\bigl(-\tfrac{\lVert x_i - x_j \rVert^2}{2 \sigma^2}\bigr)$.

### Preparation Reminders
- Practice deriving the dual starting from primal constraints.
- Understand geometric intuition: maximizing margin reduces VC dimension, improves generalization.

### Key Properties & Exam Hooks
- Support vectors lie on or violate margin: only points with $\alpha_i > 0$ influence $w$; removal of non-support points leaves decision unchanged.
- Hard vs. soft margin: hard assumes separability; soft introduces slack to tolerate noise; $C$ controls trade-off between margin width and misclassification penalty.
- Kernel choice implies feature space geometry; RBF yields infinite-dimensional mapping with localization; polynomial introduces global interactions; ensure Mercer’s condition (PSD).
- Compare SVM vs. logistic regression: hinge loss focuses on margin violations, logistic provides probabilistic outputs; both convex.
- Dual variables satisfy KKT complementarity; exam often asks to identify conditions for $\alpha_i = C$ vs. interior points.
- For kernelised SVM, prediction uses $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$—emphasise sparsity from support vectors.
- Understand scaling: features should be standardised; otherwise margin dominated by high-variance dimensions.

## Module 4 – Unsupervised and Semi-Supervised Learning
### Clustering
- K-means loop: (1) assign points to nearest centroid, (2) update centroids mean of assigned points; converges to local optimum of within-cluster SSE.
- Hierarchical clustering: agglomerative (start from singletons) vs. divisive; linkage choices (single, complete, average).

### Gaussian Mixture Models & EM
- GMM density $p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$.
- E-step (per exam Q4): compute responsibilities $\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$.
- M-step: update $N_k = \sum_i \gamma_{ik}$, $\pi_k = N_k / n$, $\mu_k = \tfrac{1}{N_k} \sum_i \gamma_{ik} x_i$, $\Sigma_k = \tfrac{1}{N_k} \sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^{\top}$.
- EM increases (but may not maximize globally) the expected complete-data log-likelihood.

### Semi-Supervised Learning
- Label propagation: diffuse labels across graph via normalized Laplacian; iterate until convergence.
- Pseudo-labeling: train on labelled set, assign confident predictions to unlabelled examples for augmentation.

### Evaluation
- Metrics: inertia (K-means SSE), silhouette score $\frac{b - a}{\max(a, b)}$, ARI for ground-truth comparisons.

### Key Properties & Exam Hooks
- K-means assumptions: spherical clusters, equal variance, similar size; sensitive to initialization (use k-means++); objective non-convex, converge to local minima.
- Hierarchical clustering trade-offs: agglomerative $O(n^2 \log n)$ but builds dendrogram for arbitrary cluster counts; linkage controls cluster shape (single-chain vs. complete-compact).
- GMMs generalise K-means: when covariances $\Sigma_k = \sigma^2 I$ and equal priors $\pi_k$, EM reduces to soft K-means.
- Responsibility matrix $\gamma_{ik}$ rows sum to 1; M-step updates maintain $\sum_k \pi_k = 1$.
- EM increases log-likelihood monotonically; exam may ask to show why complete-data log-likelihood provides lower bound.
- Semi-supervised assumptions: low-density separation (decision boundaries should lie in low-density regions) and cluster consistency—justify pseudo-labelling use cases.
- Typical pitfalls: scaling features impacts distance-based methods; need to compare clustering algorithms by both inertia and qualitative inspection.

## Module 5 – Dimension Reduction and Disentanglement
### PCA Essentials
- Goal: find orthogonal directions maximizing variance (exam MC Q5 answer A).
- Compute covariance $\Sigma$, eigen-decompose to obtain principal components; project data $Z = X W_k$.
- SVD approach: $X = U \Sigma V^{\top}$; PCs = columns of $V$.
- Explained variance ratio guides component count: $\lambda_k / \sum_j \lambda_j$.

### ICA vs. PCA
- ICA seeks statistically independent components (non-Gaussian sources) vs PCA focuses on decorrelated variance.
- Uses contrast functions (kurtosis, neg-entropy) or FastICA iterations.

### Autoencoders & Manifolds
- Undercomplete autoencoder approximates PCA if linear; nonlinear AE captures curved manifolds.
- Regularized variants (denoising, sparse, contractive) encourage disentanglement.

### Disentangled Representations
- Aim: separate latent factors influencing observations; often requires inductive biases (β-VAE, group theory).
- Identifiability challenges: without supervision, latent factors not unique up to affine transformations.

### Key Properties & Exam Hooks
- PCA facts: principal components orthonormal; first $k$ components minimise reconstruction error $\lVert X - X_k \rVert_F^2$ among rank-$k$ approximations.
- Eigenvalues quantify variance captured; scree plot helps detect elbows—likely exam prompt.
- Whitening scales PCs by $\lambda_k^{-1/2}$ to produce unit variance features; useful before ICA.
- ICA requires non-Gaussian independent sources; order not identifiable (permutation, scaling ambiguity).
- Autoencoder vs. PCA: linear AE with MSE and no bottleneck replicates PCA; nonlinear AE capture manifolds but may not ensure disentanglement.
- Manifold learning (Isomap, t-SNE, UMAP) preserves geodesic or neighbourhood distances; emphasise local vs. global structure differences.
- Disentanglement metrics: mutual information gap, $\beta$-VAE uses $\beta > 1$ to trade reconstruction for factorisation—highlight evaluation challenges.

## Module 6 – MLP, CNNs, and Implementations
### MLP Recap
- Forward pass: $a^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$, activation $h^{(l)} = \sigma(a^{(l)})$.
- Backprop: propagate $\delta^{(l)} = (W^{(l+1)})^{\top} \delta^{(l+1)} \odot \sigma'(a^{(l)})$ starting from output gradient.
- Loss functions: MSE, cross-entropy; ensure gradient checks via finite differences.

### Activation Function Cheatsheet

| Activation | Formula $\sigma(z)$ | Derivative $\sigma'(z)$ | Range | Notes / When to Use |
| --- | --- | --- | --- | --- |
| Sigmoid | $\frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | $(0, 1)$ | Probabilistic outputs; saturates $\Rightarrow$ vanishing gradients; use for binary logits or gates. |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ | $(-1, 1)$ | Zero-centred variant of sigmoid; still saturates; common in RNN hidden states. |
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ | $[0, \infty)$ | Sparse activations, efficient; risk of “dead ReLU” when $z < 0$ persistently. |
| Leaky ReLU / PReLU | $\max(\alpha z, z)$ | $\alpha$ (for $z \le 0$), $1$ otherwise | $(-\infty, \infty)$ | Mitigates dead neurons; $\alpha$ fixed (Leaky) or learned (PReLU). |
| ELU | $$\displaystyle\left\{\begin{array}{ll} z & \text{if } z > 0 \\ \alpha\big(e^{z} - 1\big) & \text{if } z \le 0 \end{array}\right.$$ | $$\displaystyle\left\{\begin{array}{ll} 1 & \text{if } z > 0 \\ \alpha e^{z} & \text{if } z \le 0 \end{array}\right.$$ | $(-\alpha, \infty)$ | Smooth negative region; improves mean activation close to zero. |
| GELU | $\tfrac{1}{2} z \left(1 + \operatorname{erf}\left(\tfrac{z}{\sqrt{2}}\right)\right)$ | $\tfrac{1}{2}\left(1 + \operatorname{erf}\left(\tfrac{z}{\sqrt{2}}\right)\right) + \tfrac{z}{\sqrt{2\pi}} e^{-z^2/2}$ | $(-\infty, \infty)$ | Used in Transformers; stochastic interpretation; smooth bump near zero. |
| Softplus | $\log(1 + e^{z})$ | $\frac{1}{1 + e^{-z}}$ | $(0, \infty)$ | Smooth ReLU; useful when differentiability needed. |
| Softmax (vector) | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | $\sigma_i(\delta_{ij} - \sigma_j)$ | $(0, 1)$ per class | Converts logits to class probabilities; gradient couples outputs. |

### Regularisation & Dropout
- Dropout randomly zeroes activations with keep probability $p$; inference scales weights by $p$ (practice exam Q6).
- Batch norm normalizes activations per minibatch; complements but does not replace dropout.
- L2 regularisation (weight decay) penalises $\lambda \lVert w \rVert_2^2$; shrinks weights, discourages large values, stabilises learning, equivalent to Gaussian prior in Bayesian view.
- L1 regularisation penalises $\lambda \lVert w \rVert_1$; drives sparsity, acts like Laplace prior, useful for feature selection.
- Early stopping halts training once validation loss rises; acts as implicit L2 by limiting number of gradient steps.
- Dropout approximates model averaging by sampling subnetworks; reduces co-adaptation, adds multiplicative noise to activations, increases robustness to overfitting.

### CNN Mechanics
- Convolution: $y_{i,j} = \sum_{m,n,c} K_{m,n,c} x_{i+m, j+n, c}$; parameter sharing arises by reusing same kernel across spatial positions (exam Q7).
- Padding/stride control spatial resolution; pooling (max/avg) adds translation invariance.
- Feature hierarchy: early filters detect edges, deeper layers capture semantic parts.
- Output spatial size (per dimension) with input $n$, kernel $k$, padding $p$, stride $s$, dilation $d$: 
  $$
  n_{\text{out}} = \left\lfloor \frac{n + 2p - d \cdot (k - 1) - 1}{s} \right\rfloor + 1.
  $$
  For standard convolutions $d=1$. Height and width computed independently; channels $C_{\text{out}} = $ number of filters.
- Receptive field grows as $r_{\ell} = r_{\ell-1} + (k_{\ell} - 1) \prod_{j < \ell} s_j$; exam questions may chain multiple conv/pool layers.
- Example sanity check: input $28 \times 28$, $k=3$, $p=1$, $s=1$ $\Rightarrow$ $28 \times 28$ output; with $s=2$ $\Rightarrow$ $\lfloor \tfrac{28 + 2 - 3 - 1}{2} \rfloor + 1 = 14$.

### Key Properties & Exam Hooks
- Universal Approximation: single hidden layer with non-linear activation can approximate continuous functions on compact sets; depth improves parameter efficiency.
- Activation choice matters: ReLU avoids vanishing gradients but unbounded; sigmoid/tanh saturate—remember derivative shapes.
- Dropout expectation: at train time multiply activations by Bernoulli mask; at inference scale weights by keep probability to preserve expected activation.
- Batch norm normalises per mini-batch then rescales by learned $\gamma$, $\beta$; stabilises gradients and allows higher learning rates.
- Convolution advantages: sparse connectivity, weight sharing; receptive field grows with depth/pooling; compute output shape via $(n + 2p - k)/s + 1$.
- Parameter counts: conv layer parameters $k_h \times k_w \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$; exam may ask to compare vs. dense layers.
- Implementation hygiene: set random seeds for reproducibility; watch mode (train vs. eval) for dropout/batchnorm; gradient clipping for exploding gradients.

## Module 7 – RNNs and Transformers
### RNN Foundations
- Standard RNN update: $h_t = \phi(W_h h_{t-1} + W_x x_t + b)$; outputs $y_t = W_y h_t$.
- Vanishing/exploding gradients: repeated multiplication by Jacobians with eigenvalues $\|\lambda\| < 1$ leads to shrinkage, while $\|\lambda\| > 1$ causes blow-up (exam Q8).

### Gated Architectures
- LSTM gates: input $i_t$, forget $f_t$, output $o_t$, candidate $g_t$; cell $c_t = f_t \odot c_{t-1} + i_t \odot g_t$.
- GRU combines gates: update $z_t$, reset $r_t$, hidden $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$.

### Transformers & Attention
- Self-attention: $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\bigl(\tfrac{QK^{\top}}{\sqrt{d_k}}\bigr) V$.
- Multi-head: split embedding into $h$ heads, concatenate outputs, project.
- Positional encodings: sinusoidal $PE_{(\text{pos},2i)} = \sin\bigl(\tfrac{\text{pos}}{10000^{2i/d}}\bigr)$.
- Strength vs RNNs: direct attention handles long-range dependencies (exam Q9); parallelizable sequence processing.

### Key Properties & Exam Hooks
- Sequential gradients: product of Jacobians causes vanishing/exploding; LSTM cell state has additive path $c_t = f_t \odot c_{t-1} + \dots$ preserving gradients when $f_t \approx 1$.
- Gating meaning: forget gate resets memory, input gate writes new info, output gate controls exposure; GRU merges update/reset, fewer parameters.
- Teacher forcing vs. free-running: mismatch leads to exposure bias; scheduled sampling can mitigate.
- Attention complexity: self-attention $O(T^2 d)$ vs. RNN $O(T d^2)$; exam may ask to analyse runtime or memory.
- Multi-head attention allows different subspaces to attend; residual connections + layer norm stabilise deep transformers.
- Compare sequence models: RNNs good for streaming/online; Transformers require full context but excel at long-range dependencies and parallel training.
- Typical calculations: compute attention weights for 2- or 3-token toy example; verify softmax sums to 1 and outputs weighted average of values.

### Vanishing & Exploding Gradients
- Root cause: backprop multiplies Jacobians across layers/time. Eigenvalues $\|\lambda\| < 1$ $\Rightarrow$ gradients shrink exponentially (vanish); $\|\lambda\| > 1$ $\Rightarrow$ gradients grow (explode).
- Saturating activations (sigmoid/tanh near $\pm$ large inputs), deep unnormalised networks, poorly initialised weights exacerbate vanishing; recurrent weight matrices with large spectral radius or long sequences exacerbate explosion.
- Mitigations: use non-saturating activations (ReLU, GELU), careful initialisation (He/Glorot), normalisation layers (batch/layer norm), residual/skip connections, gradient clipping (global norm or per-parameter) for explosions.
- Optimiser tricks: adaptive methods (Adam, RMSProp) rescale updates; learning-rate warmup and schedules stabilise training.
- Architecture solutions: LSTM/GRU introduce gating to maintain constant error flow; Transformers avoid recurrence, leveraging attention and residuals to sidestep vanishing.

## Module 8 – Large Language Models (LLMs)
### Training Pipeline
- Pretraining: next-token or masked LM objective on massive corpora; scales with tokens, parameters, compute (scaling laws: loss $\approx$ power-law in model/resources).
- Fine-tuning: supervised fine-tuning (SFT) on task data; instruction tuning for aligned outputs.
- RLHF stages: (1) collect preference data, (2) train reward model, (3) optimize policy via PPO or similar to maximize reward under KL constraint.

### Architecture & Data Handling
- Tokenization: BPE/WordPiece merges; vocabulary influences context length.
- Embeddings: token + positional + optional segment embeddings.
- Optimization: mixed-precision (fp16/bfloat16), gradient checkpointing for memory.

### Evaluation & Safety
- Use perplexity, exact match, BLEU; monitor for hallucination, bias.
- Alignment knobs: system prompts, supervised rejection sampling, policy gradient with KL penalty.

### Key Properties & Exam Hooks
- Scaling laws: loss $\approx A N^{-\alpha}$ where $N$ is compute/parameters/data; diminishing returns guide resource allocation questions.
- Pretraining vs. fine-tuning: pretraining learns general representations; SFT adapts to instructions; RLHF aligns outputs with human preference—contrast objectives.
- Tokenisation quirks: subword units allow open vocabulary but can split rare words; maximum context length set by positional embeddings.
- Parameter vs. activation memory: inference cost $\propto$ parameters; training adds optimizer states; gradient checkpointing trades compute for memory.
- Safety dimensions: hallucination, bias, toxicity; mitigations via reinforcement learning, constitutional prompts, filtering training data.
- Evaluation nuance: perplexity correlates with likelihood; instruction-following measured via exact match or human eval; know limitations.
- Potential exam prompt: describe RLHF pipeline steps and how KL penalty maintains closeness to SFT policy.

## Module 9 – Agentic AI and Reinforcement Learning
### MDP Building Blocks
- States $S$, actions $A$, transition $P(s' \mid s,a)$, reward $r$, discount $\gamma$.
- Return from time $t$: $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ (exam Q10).

### Bellman Equations
- State-value:

  $$V^{\pi}(s) = \mathbb{E}_{\pi}[ r_t + \gamma V^{\pi}(s_{t+1}) \mid s_t = s ]$$

- Action-value:

  $$Q^{\pi}(s, a) = \mathbb{E}_{\pi}[ r_t + \gamma Q^{\pi}(s_{t+1}, a_{t+1}) \mid s_t = s, a_t = a ]$$

- Optimality:

  $$Q^{*}(s, a) = \mathbb{E}[ r + \gamma \max_{a'} Q^{*}(s', a') ]$$

### Algorithms
- Value-based example (practice exam solution): Q-learning update $Q \leftarrow Q + \alpha [ r + \gamma \max_{a'} Q(s', a') - Q ]$.
- Policy-based example: REINFORCE update $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$.
- Actor–critic merges value baseline with policy gradient for lower variance.

### Planning vs Learning
- Planning (value iteration, policy iteration) uses known model $P$; learning (model-free) samples experience.
- Agent loop: observe state, choose action, receive reward/new state, update policy/value.

### Key Properties & Exam Hooks
- Discount $\gamma$ close to 1 emphasises long-term reward but increases variance; $\gamma = 0$ reduces to myopic reward.
- Value iteration contraction: Bellman optimality operator $T$ is a $\gamma$-contraction in sup norm; ensures convergence.
- On-policy (SARSA, REINFORCE) evaluates behaviour policy; off-policy (Q-learning) learns about greedy policy while following exploratory behaviour.
- Policy gradients unbiased but high variance; baselines (state value) reduce variance without bias—know derivation $\nabla_\theta J = \mathbb{E}[ (G_t - b(s_t)) \nabla_\theta \log \pi_\theta ]$.
- Exploration strategies: $\epsilon$-greedy, Boltzmann (softmax) exploration, entropy regularisation in policy gradient.
- Temporal Difference vs Monte Carlo: TD bootstrap for lower variance but potential bias; MC unbiased but high variance.
- Potential exam tasks: derive Bellman equation (practice exam), simulate one value iteration update in small gridworld, compare Q-learning vs. REINFORCE updates.

## Module 10 – Vision-Language Models
### Cross-Modal Alignment
- CLIP objective: contrastive InfoNCE loss $L = - \tfrac{1}{N} \sum_i [ \log \exp(\operatorname{sim}(v_i, t_i)/\tau) - \log \sum_j \exp(\operatorname{sim}(v_i, t_j)/\tau) ]$.
- Cosine similarity $\operatorname{sim}(u, v) = \frac{u \cdot v}{\lVert u \rVert \lVert v \rVert}$ ensures scale invariance.
- Paired image-text embeddings encourage shared latent space for retrieval.

### Architectures
- Dual-encoder (CLIP): independent vision & text encoders + contrastive training.
- Encoder–decoder (BLIP, Flamingo): cross-attention from text to vision tokens for generation.

### Applications & Tips
- Zero-shot classification via text prompts.
- Evaluate with recall@K, mean reciprocal rank; check image-text alignment qualitatively.

### Diffusion-Based Generators (Stable Diffusion)
- Latent diffusion splits work: CLIP text encoder supplies conditioning tokens; U-Net denoiser iteratively refines noisy latent codes; VAE decoder maps final latent back to pixel space.
- Training mimics denoising score matching: forward process adds Gaussian noise; model learns to predict noise $\epsilon_\theta(z_t, t, c)$; classifier-free guidance blends conditional/unconditional predictions to steer samples.
- Exam-ready pipeline summary: prompt $\rightarrow$ tokenizer/encoder $\rightarrow$ scheduler of timesteps (DDIM, PNDM) $\rightarrow$ guided denoising loop $\rightarrow$ VAE decode; CFG scale tunes fidelity vs diversity, larger scales risk washed-out compositions.

### Key Properties & Exam Hooks
- Contrastive losses enforce symmetric alignment: temperature $\tau$ controls softness of softmax; lower $\tau$ sharpens distribution.
- Dual-encoder inference efficient (cosine similarity lookup); encoder–decoder better for generation but heavier.
- Vision backbones (ViT vs. CNN) change receptive fields; text encoders often transformers—know architectural pairings.
- Multi-modal fusion options: cross-attention (query text attend to visual tokens), gated fusion, FiLM; exam may ask to compare.
- Common pitfalls: modality imbalance (text dominating), need to normalise embeddings before cosine similarity, negative sampling strategies.
- Evaluation nuance: retrieval vs. captioning metrics; ensure you mention zero-shot classification pipeline (prompt engineering).
- Relationship to self-supervised learning: InfoNCE similar to SimCLR, extends across modalities.

## Module 11 – Causal AI
### Graphical Basics
- Directed acyclic graph (DAG) encodes causal relations; edges represent direct effects.
- Collider (per exam Q11): two arrows pointing into same node; conditioning on collider can introduce dependence.
- Confounder: common cause of treatment and outcome; adjust via back-door criterion.

### Pearl’s Ladder
- Association: observe correlations.
- Intervention: apply do-operator $\mathrm{do}(X = x)$; use truncated factorization.
- Counterfactual: reason about alternate outcomes for specific units via structural causal models.

### Tools & Examples
- Back-door adjustment: $\mathbb{E}[Y \mid \mathrm{do}(X = x)] = \sum_z \mathbb{E}[Y \mid X = x, Z = z] \; p(z)$ if Z blocks back-door paths.
- Front-door criterion for mediator-based adjustment.
- Counterfactual query: compute $Y_{X \leftarrow x'}$ by (1) abduction (infer latent vars), (2) action (intervene), (3) prediction.
- Example: Ice cream sales ↔ drowning; temperature as confounder illustrates correlation vs causation vs confounding.

### Key Properties & Exam Hooks
- d-separation tests conditional independence: block directed chains and forks by conditioning on middle node; avoid conditioning on colliders unless also conditioning on descendants.
- Confounding vs. mediation vs. collider: know graphical patterns and their adjustment strategies; exam may ask to identify from DAG sketch.
- Do-calculus rules allow moving between observational and interventional distributions; memorise Rule 1 (insertion/deletion of observations), Rule 2 (action/observation exchange), Rule 3 (insertion/deletion of actions).
- Average Treatment Effect (ATE): $\mathbb{E}[Y \mid \mathrm{do}(T=1)] - \mathbb{E}[Y \mid \mathrm{do}(T=0)]$; front-door/back-door provide estimators when experiments unavailable.
- Counterfactual reasoning requires structural equations and noise terms; emphasise three-step procedure (abduction, action, prediction).
- Typical pitfalls: conditioning on post-treatment variables introduces bias; avoid controlling for mediators when estimating total effect.

## Rapid Practice Prompts
- Derive gradient for ridge regression and compare with OLS.
- Sketch K-fold CV splits for $k=5$ and explain variance reduction.
- Derive dual of hard-margin SVM and identify support vectors.
- Write EM updates for a 2-component GMM with diagonal covariance.
- Perform PCA on small matrix, compute explained variance ratio.
- Manually execute one LSTM step with provided gates.
- Calculate self-attention scores for 2-token example; show how scaling affects softmax.
- Compute one Q-learning update and REINFORCE gradient estimate.
- Draw a simple DAG with collider and show effect of conditioning.

## Exam-Style Example Problems
### Assignment 1
#### Problem 1 – Inner Product Identities
**Question:** Let $u \in \mathbb{R}^d$ and $z_i \in \mathbb{R}^d$ for $i = 1, \ldots, n$. Which expressions are equivalent to $u^\top \sum_i z_i$?
- $\sum_i \operatorname{Tr}(z_i u^\top)$
- $\sum_i \langle u, z_i \rangle$
- $\sum_i z_i^\top u$
- $(\sum_i z_i) u$

**Answer:** $\sum_i \operatorname{Tr}(z_i u^\top)$, $\sum_i \langle u, z_i \rangle$, and $\sum_i z_i^\top u$ all equal $u^\top \sum_i z_i$; $(\sum_i z_i)u$ is not equivalent in $\mathbb{R}^d$.
**Derivation:** Using trace-cyclic invariance, $\operatorname{Tr}(z_i u^\top) = \operatorname{Tr}(u^\top z_i) = u^\top z_i$; inner product and transpose forms are identical scalars, so summing preserves equality. The expression $(\sum_i z_i)u$ attempts to multiply two $d$-vectors without a transpose, yielding an outer product; it therefore does not reduce to the desired scalar.

#### Problem 2 – Mean Scatter vs Pairwise Scatter
**Question:** Let $\{x_i\}$ be $d$-dimensional vectors with average $\mu = \tfrac{1}{N} \sum_i x_i$. Define $m_1 = \tfrac{1}{N} \sum_i \|\mu - x_i\|_2^2$ and $m_2 = \tfrac{1}{N^2} \sum_i \sum_j \|x_j - x_i\|_2^2$. What is the relationship between $m_1$ and $m_2$?
- $m_1 = m_2$
- $2 m_1 = m_2$
- $4 m_1 = m_2$
- $m_1 = 2 m_2$

**Answer:** $2 m_1 = m_2$.
**Derivation:** Let $\mu = \tfrac{1}{N} \sum_i x_i$ and expand
$$\sum_{i,j} \|x_j - x_i\|_2^2 = \sum_{i,j} (\|x_j\|_2^2 + \|x_i\|_2^2 - 2 x_i^\top x_j) = 2N \sum_i \|x_i - \mu\|_2^2.$$
Divide by $N^2$ to obtain $m_2 = \tfrac{1}{N^2} \sum_{i,j} \|x_j - x_i\|_2^2 = \tfrac{2}{N} \sum_i \|x_i - \mu\|_2^2 = 2 m_1$.

#### Problem 3 – Diagonalising $XX^\top$
**Question:** Let $X \in \mathbb{R}^{d \times d}$ and $P \in \mathbb{R}^{d \times d}$. Assume each row of $P$ is an independent eigenvector of $XX^\top$. What kind of matrix is $P XX^\top P^\top$?
- Identity Matrix
- Diagonal Matrix
- Zero Matrix
- Scalar

**Answer:** $P XX^\top P^\top$ is diagonal.
**Derivation:** Let $p_k^\top$ denote the $k$-th row of $P$. By assumption $p_k^\top XX^\top = \lambda_k p_k^\top$ and the eigenvectors are orthonormal, so $p_k p_\ell^\top = 0$ when $k \ne \ell$. The $(k,\ell)$ entry of $P XX^\top P^\top$ is $p_k^\top XX^\top p_\ell = \lambda_\ell p_k^\top p_\ell$, which is zero for $k \ne \ell$ and $\lambda_k$ on the diagonal. Hence the product yields the diagonal matrix of eigenvalues.

#### Problem 4 – Epigraph Reformulation of $\max$
**Question:** Which optimisation problem is equivalent to
$$\min_w \|w\|_2^2 + \sum_i \max(5,\, 2 - w^\top x_i)\, ?$$
- $\min_w \|w\|_2^2 + \sum_i \xi_i$ subject to $w^\top x_i \ge 1 - \xi_i$, $\xi_i \ge 0$
- $\min_w \|w\|_2^2 + \sum_i \xi_i$ subject to $w^\top x_i \ge 1 - \xi_i$, $\xi_i \le 5$
- $\min_w \|w\|_2^2 + \sum_i \xi_i$ subject to $w^\top x_i \ge 1 - \xi_i$, $\xi_i \ge 5$
- $\min_w \|w\|_2^2 + \sum_i \xi_i$ subject to $w^\top x_i \ge 2 - \xi_i$, $\xi_i \ge 5$

**Answer:** Introduce slack variables via the epigraph to obtain
$$\min_{w, \{\xi_i\}} \|w\|_2^2 + \sum_i \xi_i \quad \text{s.t.} \quad \xi_i \ge 5,\; \xi_i \ge 2 - w^\top x_i, \; \forall i.$$
**Derivation:** For any scalar functions $f_i(w)$ we can rewrite $\max(a, f_i(w))$ as the optimal value of $\min_{\xi_i} \xi_i$ subject to $\xi_i \ge a$ and $\xi_i \ge f_i(w)$. Applying this epigraph construction with $a=5$ and $f_i(w)=2 - w^\top x_i$ reproduces each hinge term inside the sum.

#### Problem 5 – Ridge Regression vs Constrained Form
**Question:** Which optimisation is equivalent to minimising the regularised loss $\min_v \|A v - b\|_2^2 + \alpha \|v\|_2^2$ for $A \in \mathbb{R}^{m \times n}$, $v \in \mathbb{R}^n$, $b \in \mathbb{R}^m$?
- $\min_v \|A v - b\|_2^2$ subject to $\|v\|_2 \ge \sqrt{\alpha}$
- $\min_v \|A v - b\|_2^2 + \tfrac{\alpha}{2} \|v\|_2^2$
- $\min_v \|A v - b\|_2^2$ subject to $\|v\|_2 \le \sqrt{\alpha}$
- $\min_v \|A v - b\|_2^2 + \alpha \|v\|_1$

**Answer:** Equivalent to $\min_v \|A v - b\|_2^2$ subject to $\|v\|_2 \le \sqrt{\alpha}$.
**Derivation:** The Lagrangian of the constrained problem $\mathcal{L}(v, \lambda) = \|A v - b\|_2^2 + \lambda (\|v\|_2^2 - \alpha)$ has stationary condition $\nabla_v \mathcal{L} = 2 A^\top (A v - b) + 2 \lambda v = 0$. Identifying $\lambda$ with the regularisation weight shows the solution satisfies $\nabla_v (\|A v - b\|_2^2 + \lambda \|v\|_2^2) = 0$. Choosing $\lambda = \alpha$ yields the penalised ridge objective, and complementary slackness enforces $\|v\|_2 \le \sqrt{\alpha}$ at optimum.

### Assignment 2
#### Question 1 – Two-Layer Network Backprop with Cross-Entropy
**Question:** Provide the forward and backward pass derivations for a 2-layer neural network using cross-entropy loss.

**Answer:** With input $x \in \mathbb{R}^d$, one-hot target $y \in \mathbb{R}^K$, hidden width $H$, weights $W_1 \in \mathbb{R}^{H \times d}$, $W_2 \in \mathbb{R}^{K \times H}$, biases $b_1 \in \mathbb{R}^H$, $b_2 \in \mathbb{R}^K$, and activation $\phi$ on the hidden layer:
- Forward pass: $z_1 = W_1 x + b_1$, $h = \phi(z_1)$, $z_2 = W_2 h + b_2$, $\hat y = \operatorname{softmax}(z_2)$, $L = -\sum_{k=1}^K y_k \log \hat y_k$.
- Backward pass: $\delta_2 = \hat y - y$, $\frac{\partial L}{\partial W_2} = \delta_2 h^\top$, $\frac{\partial L}{\partial b_2} = \delta_2$, $\delta_1 = (W_2^\top \delta_2) \odot \phi'(z_1)$, $\frac{\partial L}{\partial W_1} = \delta_1 x^\top$, $\frac{\partial L}{\partial b_1} = \delta_1$.

**Derivation:**  
1. **Forward definitions:** $z_1 = W_1 x + b_1$ is the pre-activation hidden vector; $h = \phi(z_1)$ applies an elementwise nonlinearity (e.g. ReLU with $\phi'(z)=\mathbb{1}[z>0]$). The second layer produces $z_2 = W_2 h + b_2$, and the softmax gives $\hat y_k = \exp(z_{2k}) / \sum_{j=1}^K \exp(z_{2j})$. Cross-entropy loss is $L = -\sum_k y_k \log \hat y_k$.
2. **Output layer gradient:** The Jacobian identity for softmax-cross-entropy yields $\partial L / \partial z_{2k} = \hat y_k - y_k$; stacking components gives $\delta_2 \in \mathbb{R}^K$. Matrix calculus provides $\partial L / \partial W_2 = \delta_2 h^\top$ and $\partial L / \partial b_2 = \delta_2$.
3. **Hidden layer gradient:** Propagate through the linear map and activation: $\delta_1 = (W_2^\top \delta_2) \odot \phi'(z_1)$, where $\odot$ denotes the Hadamard product. Gradients follow as $\partial L / \partial W_1 = \delta_1 x^\top$ and $\partial L / \partial b_1 = \delta_1$.
4. **Batch updates:** For a mini-batch average each gradient, then update parameters with learning rate $\eta$: $W_\ell \leftarrow W_\ell - \eta \,\partial L/\partial W_\ell$, $b_\ell \leftarrow b_\ell - \eta \,\partial L/\partial b_\ell$ for layers $\ell \in \{1,2\}$.

#### Question 2 – Hard-Margin SVM Dual from Primal
**Question:** Derive the dual form of the hard-margin SVM starting from its primal optimisation.

**Answer:** The primal $\min_{w,b} \tfrac{1}{2}\|w\|_2^2$ subject to $y_i(w^\top x_i + b) \ge 1$ yields the dual
$$
\max_{\alpha \in \mathbb{R}^N} \sum_{i=1}^N \alpha_i - \tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^\top x_j
$$
with constraints $\alpha_i \ge 0$ and $\sum_i \alpha_i y_i = 0$. The optimal separator satisfies $w^\star = \sum_i \alpha_i^\star y_i x_i$; recover $b^\star$ from any support vector via $b^\star = y_k - (w^\star)^\top x_k$ for $0 < \alpha_k^\star$.

**Derivation:**  
1. **Lagrangian:** Introduce multipliers $\alpha_i \ge 0$ for each margin constraint, forming
$$\mathcal{L}(w,b,\alpha) = \tfrac{1}{2}\|w\|_2^2 - \sum_{i=1}^N \alpha_i \big[y_i(w^\top x_i + b) - 1\big].$$
2. **Stationarity:** Setting derivatives to zero gives $w = \sum_i \alpha_i y_i x_i$ and $\sum_i \alpha_i y_i = 0$.
3. **Dual objective:** Substitute these expressions back into $\mathcal{L}$ to obtain
$$g(\alpha) = \sum_{i=1}^N \alpha_i - \tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^\top x_j.$$
4. **KKT conditions:** Complementary slackness $\alpha_i [y_i(w^\top x_i + b) - 1] = 0$ identifies support vectors and ensures only points on the margin retain non-zero $\alpha_i$.
5. **Solution recovery:** Maximising $g(\alpha)$ under $\alpha_i \ge 0$ and $\sum_i \alpha_i y_i = 0$ delivers $\alpha^\star$ and, consequently, $w^\star$ and $b^\star$.

#### Question 3 – Soft-Margin SVM Dual with Slack Penalties
**Question:** Derive the dual of the $\ell_2$-regularised soft-margin SVM, whose primal is
$$
\min_{w,b,\xi} \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^N \xi_i
\quad \text{s.t.} \quad y_i (w^\top x_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0.
$$

**Answer:** The dual maximisation problem is

$$
\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^\top x_j
$$

subject to $0 \le \alpha_i \le C$ and $\sum_{i=1}^N \alpha_i y_i = 0$. At optimum $w^\star = \sum_i \alpha_i^\star y_i x_i$ and any index with $0<\alpha_i^\star<C$ satisfies $b^\star = y_i - (w^\star)^\top x_i$.

**Derivation:**  
1. **Primal Lagrangian:** Introduce multipliers $\alpha_i \ge 0$ for the margin constraints and $\mu_i \ge 0$ for the non-negativity of $\xi_i$:

   $$
   L(w,b,\xi,\alpha,\mu) = \tfrac{1}{2}\|w\|_2^2 + C\sum_i \xi_i
   - \sum_i \alpha_i \big[y_i(w^\top x_i + b) - 1 + \xi_i\big]
   - \sum_i \mu_i \xi_i.
   $$

1. **Stationarity:**  
   - $\partial L/\partial w = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i$.
   - $\partial L/\partial b = 0 \Rightarrow \sum_i \alpha_i y_i = 0$.
   - $\partial L/\partial \xi_i = 0 \Rightarrow C - \alpha_i - \mu_i = 0$.
1. **Bounding multipliers:** Since $\mu_i \ge 0$, the last relation enforces $0 \le \alpha_i \le C$.
1. **Dual objective:** Substitute the stationary expressions into $L$, eliminate $\mu_i$ using $\mu_i = C - \alpha_i$, and simplify to obtain
   $$
   g(\alpha) = \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j.
   $$
1. **KKT conditions:** Complementary slackness provides the usual support-vector structure: $\alpha_i [y_i(w^\top x_i + b) - 1 + \xi_i]=0$ and $(C - \alpha_i)\xi_i = 0$. Indices with $0 < \alpha_i < C$ lie exactly on the soft margin and supply $b^\star$.

#### Question 4 – Ridge Regression Closed-Form Solution
**Question:** Given training inputs $X \in \mathbb{R}^{N \times d}$, targets $y \in \mathbb{R}^N$, and ridge penalty $\lambda > 0$, derive the closed-form minimiser of
$$
J(w) = \|Xw - y\|_2^2 + \lambda \|w\|_2^2.
$$

**Answer:** Setting the gradient to zero yields $(X^\top X + \lambda I_d) w^\star = X^\top y$, so the ridge estimator is
$$
w^\star = (X^\top X + \lambda I_d)^{-1} X^\top y.
$$
If the design matrix includes a bias column of ones, omit that column from regularisation by augmenting the penalty with a selector matrix that leaves the bias coefficient unpenalised.

**Derivation:**  
1. Expand $J(w)$ in matrix form: $J(w) = (Xw - y)^\top (Xw - y) + \lambda w^\top w$.
2. Differentiate: $\nabla_w J = 2 X^\top (Xw - y) + 2 \lambda w$.
3. Set $\nabla_w J = 0$ to obtain the normal equation $(X^\top X + \lambda I_d) w = X^\top y$.
4. Because $\lambda > 0$ ensures $X^\top X + \lambda I_d$ is positive definite (hence invertible), solving gives $w^\star$ above.
5. When excluding the intercept from shrinkage, replace $\lambda I_d$ with $\lambda D$ for diagonal $D$ whose entries are $1$ for slope coefficients and $0$ for the intercept; the derivation proceeds identically.

#### Question 6 – Implementing L2 Weight Decay Without Penalising Bias
**Question:** Modify the training loop so each update applies L2 regularisation (weight decay) to model parameters. Explain how to implement it and why biases are exempt.

**Answer:** Introduce a hyperparameter $\lambda_{\text{wd}}$ and apply for every weight tensor $W$ the update
$$W \leftarrow W - \eta \left(\nabla_W L + \lambda_{\text{wd}} W\right),$$
while biases keep the plain gradient step $b \leftarrow b - \eta \nabla_b L$. Use the same $\lambda_{\text{wd}}$ across steps so the shrinkage is consistent throughout training.

**Derivation:** Adding $$\frac{\lambda_{\text{wd}}}{2}\sum_\ell \|W_\ell\|_2^2$$ to the loss differentiates to $$\nabla_{W_\ell} L + \lambda_{\text{wd}} W_\ell$$, giving the weight-decay term in the update. Bias vectors are omitted because they merely translate activations; penalising them would shift optimal intercepts and can introduce systematic error without controlling model capacity. Moreover, weight decay combats exploding weights that amplify inputs, an effect that biases do not share. This is why standard implementations regularise weights but not biases.

#### Question 7 – Hyperparameter Tuning and Test-Set Discipline
**Question:** Why is hyperparameter tuning crucial, and why must the test set remain untouched during tuning?

**Answer:** Hyperparameters govern optimisation and model capacity. Oversized learning rates destabilise training; undersized rates slow convergence. Small batches increase gradient noise and exploration, large batches smooth updates but can settle in sharp minima. Hidden size, depth, and regularisation strengths dictate the bias–variance balance: too little capacity underfits; too much capacity overfits unless regularised. Tuning on a validation set searches this space for the configuration that maximises generalisation. The test set must stay unused during tuning to avoid leakage: incorporating it would bias decisions toward that data, yielding over-optimistic estimates. Reserve the test set for a single, final evaluation after hyperparameters are fixed.
