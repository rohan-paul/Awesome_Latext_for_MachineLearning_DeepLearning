- ### [Find me in Twitter](https://twitter.com/rohanpaul_ai)

## [📚 I write daily for my 112K+ readers on actionable AI developments. Get a 1300+ page Python book as soon as you subscribing (its FREE) ↓↓)](https://www.rohan-paul.com/s/daily-ai-newsletter/archive?sort=new)

[logo]: https://github.com/rohan-paul/rohan-paul/blob/master/assets/newsletter_rohan.png

[![Rohan's Newsletter][logo]](https://www.rohan-paul.com/) &nbsp;

=====================================================================
### 📣 Classical ML Algorithms
=====================================================================

## **Linear Regression** –

# $` \hat{y} = w^T x + b `$

### –`$w,b`$

weights, bias – predicts continuous output via linear combination. It is sensitive to outliers, so robust variants are often preferred in noisy datasets.

----------------------

## **Normal Equation** –

# $` w^* = (X^T X)^{-1} X^T y`$

–`$X,y`$: data matrix, targets – closed-form solution for least squares. It can be computationally expensive for large feature sets because matrix inversion is typically \(O(n^3)\).

----------------------

## **Logistic Regression** –

# $` \log\frac{p}{1-p} = w^T x + b`$

# $p`$:`$P(y=1|x)$

Linear logits yield sigmoid-class probability. It can be extended with regularization or nonlinear features for more complex decision boundaries.

----------------------

## **Bayes’ Theorem** –

# $` P(Y|X) = \frac{P(X|Y)\,P(Y)}{P(X)} `$

–`$P(X|Y)$: likelihood – updates class probability given evidence. It underpins many ML algorithms by formally combining prior knowledge with new evidence for better predictive accuracy.

----------------------

## **Naïve Bayes** –

# $` P(y|x_{1:n}) \propto P(y)\prod_{i=1}^n P(x_i|y) `$

 – assumes features`$x_i$` independent – simple probabilistic classifier. Despite the "naïve" assumption, it often performs surprisingly well in text classification and spam filtering tasks.

----------------------

## **Entropy (Shannon)** –

# $` H(p) = -\sum_i p_i \log p_i`$

–`$p_i`$: class prob – measures impurity/uncertainty of a distribution. Low entropy implies high confidence in predictions, making it crucial for measuring model uncertainty in classification tasks.

----------------------

## **Gini Impurity** –

# $` G(p) = \sum_i p_i(1-p_i) `$

–`$p_i`$: class prob – alternate impurity metric for decision tree splits. It's computationally simpler than entropy, which can make decision tree training faster in practice.

----------------------

## **Information Gain** –

# $` \Delta H = H(\text{parent}) - \sum_k \frac{N_k}{N}H(\text{child}_k) `$

–`$N_k`$: child size – entropy reduction by a split. Sometimes replaced by Gini gain for efficiency, but conceptually highlights how a split reduces uncertainty.

----------------------

## **Linear Decision Boundary** –

# $` w^T x + b = 0`$

–`$w`$: normal vector – defines a hyperplane separating classes in feature space. In high-dimensional spaces, hyperplanes can separate data more easily, but risk overfitting if dimensions vastly exceed samples.

----------------------

## **Perceptron Update** –

# $` w \leftarrow w + \eta\,y\,x $` (if`$y\,w^T x < 0$)

 –`$\eta`$: learning rate – adjusts linear classifier until data separable. Converges only if the data is linearly separable; otherwise, it may never find a perfect boundary.

----------------------

## **SVM Decision (Kernel)** –

# $` f(x) = \text{sign}\!\Big(\sum_{i\in SV}\alpha_i y_i K(x_i,x) + b\Big) `$

–`$K`$: kernel – classification using support vectors in feature space. Choosing an appropriate kernel is crucial, as it implicitly defines the feature space for separation.

----------------------

## **RBF Kernel** –

# $` K(x,z) = \exp\!\Big(-\frac{\|x-z\|^2}{2\sigma^2}\Big) `$

–`$\sigma`$: bandwidth – popular kernel mapping into infinite-dimensional space. A small sigma leads to complex boundaries, risking overfit; a large sigma oversmooths the boundary.

----------------------

## **Fisher’s LDA** –

# $` J(w) = \frac{w^T S_B\,w}{w^T S_W\,w} `$

–`$S_B,S_W`$: between/within-class scatter – finds projection maximizing class separability. It’s closely related to PCA but specifically optimizes class separability rather than just variance in data.

=================================


## **Gaussian Mixture** –

# $` p(x) = \sum_{k=1}^K \pi_k\,\mathcal{N}(x|\mu_k,\Sigma_k) `$

–`$\pi_k`$: mixture weights – models data as weighted sum of Gaussians. Can capture multimodal distributions, but might need careful initialization to avoid poor local optima in parameter estimation.

----------------------

## **Expectation-Maximization**

# $Q(\theta|\theta^{old}) = E_{z|x,\theta^{old}}[\log p(x,z|\theta)]$

–`$\theta`$: params – iterative E-step/M-step to find MLE with latent variables. Can be extended to various mixture models and often converges quickly, but only guarantees finding local maxima.

----------------------

## **Ridge Regression** –

# $` w^* = (X^T X + \lambda I)^{-1} X^T y`$

–`$\lambda`$: regularizer – closed-form solution adding`$L2$` penalty (prevents overfit). Balances fitting error and coefficient shrinkage, often outperforming ordinary least squares in the presence of multicollinearity.

----------------------

## **Bayes Optimal Classifier** –

# $` y^* = \arg\max_y P(Y=y\,|\,x) `$

 – posterior`$P(Y|x)$

 – theoretical minimal-error classifier given true distribution. Although theoretically unbeatable, it's rarely achievable in practice because the true posterior is usually unknown.

=========================


## **Bias–Variance Split** –

# $` \mathbb{E}[(\hat{f}-f)^2] = \text{Bias}^2 + \text{Var} + \sigma^2`$

 – decomposes generalization error – highlights trade-off in model complexity. Ensemble methods often reduce variance but can raise bias; controlling both is key for robust performance.

----------------------

## **Maximum Likelihood** –

# $` \hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^N p(x_i|\theta) `$

–`$\theta`$: model parameters – chooses parameters that maximize data probability. It might overfit without regularization, so in high-dimensional data, more robust methods are often preferred.

==============


## **Maximum a Posteriori** –

# $` \hat{\theta}_{MAP} = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]`$

 – includes prior`$p(\theta)$

 – regularized estimator (Bayesian inference). Acts as a middle ground between MLE and fully Bayesian approaches, including prior beliefs in parameter estimation.

----------------------

## **0–1 Loss** –

# $` L(y,\hat{y}) = \mathbf{1}\{y \neq \hat{y}\} `$

 – indicator misclassification – basic error count (non-differentiable, basis of accuracy). Despite its simplicity, many practical algorithms minimize differentiable surrogates like cross-entropy for computational tractability.

----------------------

## **Mutual Information** –

# $` I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)} `$

 – measures dependency between`$X,Y`$

 – maximized in feature selection for predictive power. High mutual information indicates strongly dependent variables, which can guide feature selection to boost classification performance.

-------------------

## **K-Means** –

```math
\Huge
\min_{C_1,\dots,C_K}\sum_{k=1}^K\sum_{x\in C_k}\|x-\mu_k\|^2
```

– \(\mu_k\): cluster centroids –
A popular clustering method that partitions data into \(K\) groups by minimizing within-cluster variance. Proper initialization (e.g., k-means++) can greatly improve results and avoid poor local minima in complex datasets.

-------------------

## **Elastic Net** –

```math
\Huge
w^* = \arg\min_{w}\Bigl(\|y - Xw\|^2 + \alpha\lambda\|w\|_1 + \frac{1-\alpha}{2}\lambda\|w\|_2^2\Bigr)
```

## **K-Nearest Neighbors** –

```math
\Huge
\hat{y} = \text{majority}\bigl(\{y_i \mid x_i \in N_k(x)\}\bigr)\quad\text{or}\quad \hat{y}=\frac{1}{k}\sum_{x_i\in N_k(x)}y_i
```
– classification/regression –
Makes predictions based on \(k\) closest training samples in feature space. Although simple and effective, its high memory usage and sensitivity to irrelevant features necessitate careful distance metric and feature selection.


– combines L1/L2 –
Balances L1’s feature selection with L2’s stability, often outperforming pure Lasso or Ridge in high-dimensional settings. It mitigates Lasso’s tendency to arbitrarily pick among correlated predictors by distributing shrinkage more evenly.

-------------------

## **Decision Tree** –

```math
\Huge
\text{Split}(D)=\arg\max_{\text{feature},\,\text{threshold}} \bigl[\text{ImpurityReduction}(D)\bigr]
```
– recursively partitions –
Uses a greedy approach to split data until stopping criteria. It’s easy to interpret but prone to overfitting; pruning or ensemble methods like random forests often enhance its generalization performance.

----------------------

## **Random Forest** –

```math
\Huge
\hat{f}(x) = \frac{1}{M}\sum_{m=1}^M f_m(x)
```
– bagged decision trees –
Trains many decision trees on bootstrap samples with feature subsampling. Its ensemble averaging stabilizes predictions and reduces overfitting, typically delivering strong performance with relatively little hyperparameter tuning.

----------------------

## **DBSCAN** –

```math
\Huge
N_{\varepsilon}(x)=\{\,x' \mid \|x'-x\|\leq \varepsilon\}
```
– density-based clustering –
Clusters points in dense regions and labels outliers as noise, automatically discovering clusters of arbitrary shape. Requires choosing \(\varepsilon\) (neighborhood size) and \(minPts\) carefully to capture true data structure.



## **Hierarchical Clustering** –

```math
\Huge
\text{Dendrogram construction via} \quad d(C_i,C_j)=\min(\|x_p - x_q\|)
```
– e.g., single linkage –
Builds a dendrogram by iteratively merging or splitting clusters based on a linkage criterion. It avoids specifying the number of clusters upfront, but can be computationally expensive for very large datasets.



## **Independent Component Analysis (ICA)** –

```math
\Huge
s = Wx,\quad \text{maximize non-Gaussianity of } s
```
– blind source separation –
Separates a multivariate signal into additive subcomponents assumed non-Gaussian and statistically independent. It’s commonly used in signal processing (e.g., EEG data) to isolate meaningful components from mixed observations.



## **Principal Component Analysis (PCA)** –

```math
\Huge
\max_{W}\,\text{trace}\bigl(W^T S_X W\bigr)\quad\text{subject to}\;W^T W=I
```
– finds directions of max variance –
Reduces dimensionality by projecting data onto uncorrelated axes of maximal variance. Often used as a pre-processing step to mitigate overfitting and simplify models, though it discards potentially important low-variance components.



## **Kernel PCA** –

```math
\Huge
\max_{\alpha}\,\alpha^T K \alpha\quad\text{subject to}\;\alpha^T \mathbf{1} = 0,\;\|\alpha\|=1
```
– nonlinear PCA –
Applies PCA in a high-dimensional feature space induced by a kernel function. Captures complex, nonlinear structures in data, but choosing an appropriate kernel and tuning parameters like bandwidth can be challenging.




## **Mean Shift** –

```math
\Huge
m(x) = \frac{\sum_{x_i \in N(x)} x_i K(\|x_i - x\|)}{\sum_{x_i \in N(x)} K(\|x_i - x\|)} - x
```
– mode-seeking clustering –
Iteratively shifts each point to the average of neighbors within a kernel window, revealing cluster “modes.” It doesn’t require specifying cluster counts, though bandwidth selection critically affects results.



## **Markov Chain** –

```math
\Huge
P(X_{t+1}=x\mid X_t,\dots,X_0) = P(X_{t+1}=x\mid X_t)
```
– memoryless state transitions –
Models systems where the probability of moving to the next state depends only on the current one. Widely used for stochastic processes like weather prediction and language modeling (via Markov assumptions).



## **Hidden Markov Model (HMM)** –

```math
\Huge
P(X,Z\mid\theta) = \prod_{t=1}^T P(x_t\mid z_t) P(z_t\mid z_{t-1})
```
– latent state sequence –
Captures sequences where observations depend on underlying hidden states that evolve with Markov dynamics. Commonly applied to speech recognition, part-of-speech tagging, and other sequential classification tasks.



## **Kalman Filter** –

```math
\Huge
\hat{x}_{t\mid t}=\hat{x}_{t\mid t-1}+K_t\Bigl(y_t - H\,\hat{x}_{t\mid t-1}\Bigr)
```
– linear-Gaussian state estimation –
Recursively estimates the hidden state of a dynamic system with Gaussian noise. Exploits prediction–update steps to smooth noisy measurements and track variables like position or velocity in real time.



## **Q-Learning** –

```math
\Huge
Q(s,a)\!\leftarrow Q(s,a) + \alpha\!\bigl(r + \gamma\max_{a'}Q(s',a') - Q(s,a)\bigr)
```
– off-policy RL –
Learns an optimal action-value function via temporal-difference updates, even while following an exploratory policy. Convergence is guaranteed under appropriate conditions, making it a foundational algorithm in reinforcement learning.



## **SARSA** –

```math
\Huge
Q(s,a)\!\leftarrow Q(s,a) + \alpha\!\bigl(r + \gamma\,Q(s',a') - Q(s,a)\bigr)
```
– on-policy RL –
Updates Q-values based on the actual action chosen (\(a'\)), maintaining consistency with the agent’s behavior policy. This “on-policy” nature makes SARSA sensitive to exploration strategies, differing subtly from off-policy Q-learning.



## **Policy Gradient** –

```math
\Huge
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau\sim\pi_\theta}\Bigl[\sum_{t}\nabla_\theta \log\pi_\theta(a_t\mid s_t)\,R(\tau)\Bigr]
```
– direct optimization of policy –
Optimizes parameterized policies by following the gradient of expected returns. Unlike value-based methods, it can model stochastic policies smoothly, enabling advanced strategies like continuous control in robotics.



## **Actor-Critic** –

```math
\Huge
\nabla_\theta J(\theta) \approx \nabla_\theta \log\pi_\theta(a_t\mid s_t)\,\delta_t,\quad \delta_t=r_t+\gamma V_{\phi}(s_{t+1})-V_{\phi}(s_t)
```
– hybrid RL approach –
Combines value-based and policy-based methods, reducing variance via a learned critic. The critic estimates state value to guide the policy gradient, yielding faster, more stable convergence than pure policy gradient methods.



## **ARIMA** –

```math
\Huge
(1 - \phi_1 B - \dots - \phi_p B^p) X_t = (1 + \theta_1 B + \dots + \theta_q B^q)\,\varepsilon_t
```
– time series forecasting –
A classical statistical approach modeling autoregressive and moving-average components with differencing for stationarity. Properly tuned ARIMA can be highly effective, but it may struggle with complex nonlinear patterns.



## **Neural Network (Fully-Connected)** –

```math
\Huge
h^{(l)} = \sigma\bigl(W^{(l)} h^{(l-1)} + b^{(l)}\bigr)
```
– stacked layers –
Learns hierarchical feature representations by composing linear transformations and nonlinear activations. Depth and hidden units are hyperparameters influencing capacity; large networks can overfit without regularization or sufficient data.



## **Convolutional Neural Network (CNN)** –

```math
\Huge
y^{(l)} = \sigma\bigl(W^{(l)} * x^{(l-1)} + b^{(l)}\bigr)
```
– shared filters –
Specialized for grid-like data (e.g., images), using convolution to extract local patterns and reduce parameters. Pooling layers further aggregate spatial information, enabling CNNs to excel in vision tasks and beyond.



## **GAN (Generative Adversarial Network)** –

```math
\Huge
\min_G \max_D \; V(D,G) = \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log (1 - D(G(z)))]
```
– generator vs. discriminator –
Trains two competing networks: a generator synthesizes data while a discriminator distinguishes real from fake. The adversarial setup pushes the generator to produce highly realistic samples, but training can be unstable.



## **VAE (Variational Autoencoder)** –

```math
\Huge
\mathcal{L}(\theta,\phi) = \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr)
```
– probabilistic generative model –
Learns latent representations by maximizing a variational lower bound. Balances reconstruction accuracy with a KL regularizer, enabling smooth latent spaces for generative tasks like image synthesis or anomaly detection.



=====================================================================
### 📣 Distance & Similarity Measures
=====================================================================

## **Euclidean Distance** –

# $` d_2(x,z) = \sqrt{\sum_{i}(x_i - z_i)^2} `$

 – standard`$L2$` norm – straight-line distance in`$\mathbb{R}^n`$, used in KNN and clustering. Also arises in PCA for measuring reconstruction error.

-------------------

## **Manhattan Distance** –

# $` d_1(x,z) = \sum_{i}|x_i - z_i|`$

 – sum of absolute differences

 –`$L1$` metric, robust to outliers in some cases. Often influences feature selection by promoting sparsity in L1 regularization.

-------------------

## **Minkowski Distance** –

# $` d_p(x,z) = \Big(\sum_{i}|x_i - z_i|^p\Big)^{1/p} `$

 – general`$L_p$` norm – unifies`$L1$,`$L2$` and others by parameter`$p`$. Fractional p values can emphasize large or small differences differently than L1 or L2.

-------------------

## **Mahalanobis Distance** –

# $` d_M(x,z) = \sqrt{(x-z)^T S^{-1}(x-z)} `$

–`$S`$: covariance matrix – distance accounting for feature correlations (used in anomaly detection). Crucial in linear discriminant analysis for class-based dimensionality reduction.

-------------------

## **Hamming Distance** –

# $` d_H(u,v) = \sum_{i}\mathbf{1}\{u_i \neq v_i\} `$

 – counts differing components – distance for binary strings/categorical vectors (used in hashing, error-correcting). Essential in cryptography for distinguishing codewords or hashing seeds.

-------------------

## **Cosine Similarity** –

# $` \cos(x,z) = \frac{x \cdot z}{\|x\|\,\|z\|} `$

 – dot-product normalized – measures angle closeness between vectors (used in recommender systems, embeddings). Invariant to vector magnitude, making it vital for orientation-based similarity.

----------------------

## **Chebyshev Distance**

# $d_\infty(x, z) = \max_i |x_i - z_i|$

*   Also known as L∞ norm distance; measures the greatest difference along any single dimension. Useful in logistics for chessboard-like movements.

-------------------

## **Squared Euclidean Distance**

# $d^2(x, z) = \sum_i (x_i - z_i)^2$

*   Computationally cheaper than Euclidean (avoids square root). Emphasizes larger differences more strongly; frequently used in optimization objectives (e.g., k-means).

-------------------

## **Canberra Distance**

# $d_C(x, z) = \sum_i \frac{|x_i - z_i|}{|x_i| + |z_i|}$

*   A weighted Manhattan distance, sensitive to changes near zero but less affected by large outliers. Useful for count data or non-negative sparse data.

-------------------

## **Bray-Curtis Dissimilarity**

# $d_{BC}(x, z) = \frac{\sum_i |x_i - z_i|}{\sum_i (x_i + z_i)}$

*   Bounded between 0 and 1 (for non-negative data). Widely used in ecology; ignores joint absences and emphasizes composition over total abundance.

-------------------

## **Correlation Distance**

# $d_{corr}(x, z) = 1 - \frac{(x - \bar{x}) \cdot (z - \bar{z})}{\|x - \bar{x}\|_2 \|z - \bar{z}\|_2}$

*   Measures dissimilarity based on Pearson correlation; sensitive to the linear relationship shape, invariant to scaling and translation of data.

-------------------

## **Angular Distance**

# $d_A(x, z) = \frac{1}{\pi} \arccos(\cos(x, z))$

*   Directly measures the angle between vectors, normalized to [0, 1]. Closely related to Cosine Similarity (Cosine Distance = 1 - Cosine Similarity).

-------------------

## **Jaccard Distance**

# $d_J(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|} = \frac{|A \triangle B|}{|A \cup B|}$

*   Measures dissimilarity between finite sets (or binary vectors); ignores shared absences (0-0 matches). Crucial for sparse data, like document similarity.

-------------------

## **Sørensen–Dice Distance**

# $d_{Dice}(A, B) = 1 - \frac{2|A \cap B|}{|A| + |B|}$

*   Similar to Jaccard but gives more weight to intersections. Widely used in image segmentation (Dice score) and bioinformatics.

-------------------

## **Tversky Index (Similarity)**

# $S(A, B) = \frac{|A \cap B|}{|A \cap B| + \alpha |A \setminus B| + \beta |B \setminus A|}$

*   Generalizes Dice ($\alpha=\beta=0.5$) and Jaccard ($\alpha=\beta=1$). Allows asymmetric weighting of differences, useful in modeling cognitive similarity judgments.

-------------------

## **Levenshtein Distance** – `$d_L(u, v) = \text{minimum number of single-character edits (insertions, deletions, substitutions) required to change } u \text{ into } v`$

*   Classic edit distance for strings. Fundamental in spell checking, DNA sequencing alignment, and natural language processing tasks.

-------------------

## **Damerau-Levenshtein Distance**

# $d_{DL}(u, v) = \text{minimum edits like Levenshtein, plus transpositions of adjacent characters}$

*   An extension of Levenshtein, counts transposition (e.g., "ca" -> "ac") as a single edit. Often more suitable for human typing errors.

-------------------

## **Jaro Similarity**

# $sim_J(s_1, s_2) = \begin{cases} 0 & \text{if } m = 0 \\ \frac{1}{3} \left( \frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m-t}{m} \right) & \text{otherwise} \end{cases}$` (where m = matching chars, t = transpositions)

*   String similarity measure considering matching characters and transpositions within a proximity window. Good for record linkage (matching names/addresses).

-------------------

## **Jaro-Winkler Similarity**

# $sim_W(s_1, s_2) = sim_J + l \cdot p (1 - sim_J)$` (where l = length of common prefix, p = scaling factor)

*   Refines Jaro similarity by giving extra weight to matching prefixes. Improves accuracy significantly for names or identifiers where prefixes often match.

-------------------

## **Kullback-Leibler (KL) Divergence**

# $D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$` or`$\int p(x) \log \frac{p(x)}{q(x)} dx`$

*   Measures information loss when approximating distribution P with Q. Asymmetric ($D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$). Central to variational inference.

-------------------

## **Jensen-Shannon (JS) Divergence**

# $JSD(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$, where`$M = \frac{1}{2}(P+Q)$

*   A symmetric, smoothed version of KL divergence, bounded by`$\log 2$. Often preferred over KL as a finite distance metric between distributions.

-------------------

## **Wasserstein Distance (Earth Mover's Distance)**

# $W_p(P, Q) = \left( \inf_{\gamma \in \Gamma(P, Q)} \int_{\mathcal{X} \times \mathcal{X}} d(x, y)^p \, d\gamma(x, y) \right)^{1/p}$

*   Measures the minimum "cost" to transform one distribution into another. Provides better gradients than KL/JS for training GANs, especially with non-overlapping supports.

-------------------

## **Hellinger Distance**

# $H(P, Q) = \frac{1}{\sqrt{2}} \sqrt{\sum_{i} (\sqrt{p_i} - \sqrt{q_i})^2}$` or`$H(P,Q) = \frac{1}{\sqrt{2}} \|\sqrt{p} - \sqrt{q}\|_2$

*   A symmetric metric measuring distance between probability distributions, related to Bhattacharyya distance. Bounded between 0 and 1.

-------------------

## **Bhattacharyya Distance**

# $D_B(P, Q) = -\ln \left( \sum_{i} \sqrt{p_i q_i} \right)$

*   Measures the similarity of two probability distributions. Related to Hellinger; the term inside ln is the Bhattacharyya coefficient (similarity).

-------------------

## **Haversine Distance**

# $d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + \cos(\phi_1) \cos(\phi_2) \sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)$` (r=radius,`$\phi`$=lat,`$\lambda`$=lon)

*   Calculates great-circle distance between two points on a sphere given latitudes/longitudes. Essential for geospatial data analysis and location-based services.

-------------------

## **Dynamic Time Warping (DTW)**

# $DTW(X, Y) = \min_{\pi} \sqrt{\sum_{(i, j) \in \pi} d(x_i, y_j)^2}$` (where`$\pi$` is a warping path)

*   Finds optimal alignment between two time series that may vary in speed or timing. Widely used in speech recognition and time series analysis.

-------------------

## **Variation of Information (VI)**

# $VI(X; Y) = H(X) + H(Y) - 2I(X; Y)$` (H=entropy, I=mutual information)

*   An information-theoretic metric measuring distance between clusterings. Obeys the triangle inequality, making it a true metric.

-------------------

## **Normalized Mutual Information (NMI) (Similarity)**

# $NMI(X, Y) = \frac{I(X; Y)}{\sqrt{H(X) H(Y)}}$` or`$\frac{2 I(X;Y)}{H(X)+H(Y)}$

*   Measures mutual dependence between two clusterings, normalized to [0, 1]. Widely used for evaluating clustering algorithm performance against ground truth.

-------------------

## **Russell-Rao Coefficient (Distance)**

# $d_{RR}(x, y) = \frac{n_{01} + n_{10}}{n}$` (for binary vectors:`$n_{ab}$` = count of`$a$` in`$x`$,`$b$` in`$y`$)

*   Proportion of mismatches (0-1 or 1-0) in binary vectors. Simple but often less informative than Jaccard for sparse data.

-------------------

## **Sokal-Michener Distance**

# $d_{SM}(x, y) = \frac{2(n_{01} + n_{10})}{n_{00} + n_{11} + 2(n_{01} + n_{10})}$

*   Gives equal weight to matches (0-0, 1-1) and mismatches (0-1, 1-0) in binary data comparisons.

-------------------

## **Rogers-Tanimoto Distance**

# $d_{RT}(x, y) = \frac{2(n_{01} + n_{10})}{n_{11} + n_{00} + 2(n_{01} + n_{10})}$

*   Similar structure to Sokal-Michener but gives double weight to mismatches relative to matches in binary data.

=====================================================================
### 📣 Activation Functions
=====================================================================

## **Binary Step** –

# $` f(x) = \begin{cases}1 & x>0\\0 & x\le 0\end{cases} `$

 – threshold at 0 – outputs binary state; non-differentiable (used in perceptrons, not in gradient-based training). Found in early neural networks like the McCulloch-Pitts model.

-------------------

## **Identity (Linear)** –

# $` f(x) = x`$

–`$x`$: input – returns input unchanged – used for regression outputs and skip connections. Purely linear usage nullifies deep network expressiveness.

-------------------

## **Sigmoid** –

# $` \sigma(x) = \frac{1}{1 + e^{-x}} `$

–`$x`$: input – squashes to`$(0,1)$` range – smooth activation for probabilities (prone to saturation for large`$|x|$). Still favored in output layers for binary classification.

-------------------

## **Tanh** –

# $` \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} `$

–`$x`$: input – outputs in`$(-1,1)$

 – zero-centered sigmoid variant, mitigates bias in activations. Commonly appears in LSTM gates to balance negative and positive signals.

-------------------

## **ReLU** –

# $` f(x) = \max(0,\,x) `$

–`$x`$: input – passes positive values, zeros out negative – simple, sparse activation that avoids vanishing gradient (popular in deep nets). However, it can produce skewed outputs if not initialized carefully.

-------------------

## **Leaky ReLU** –

# $` f(x) = \begin{cases}x & x\ge 0\\ \alpha x & x<0\end{cases} `$

–`$\alpha`$: small slope – allows small negative gradient – prevents “dying ReLU” by letting negativity through. It helps maintain non-zero gradients for negative inputs.

-------------------

## **PReLU** –

# $` f(x) = \begin{cases}x & x\ge 0\\ a\,x & x<0\end{cases} `$

–`$a`$: learnable slope – adaptive leaky ReLU – learns optimum negative slope per neuron. Empirically shown to speed up training versus standard ReLU variants.

-------------------

## **ELU** –

# $` f(x) = \begin{cases}x & x\ge 0\\ \alpha(e^x - 1) & x<0\end{cases} `$

–`$\alpha`$: scale (>0) – smooth exponential for`$x<0$

 – can produce negative outputs, improving mean activations. It also lessens the vanishing gradient issue for negative inputs.

-------------------

## **SELU** –

# $` f(x) = \lambda \begin{cases}x & x\ge 0\\ \alpha(e^x - 1) & x<0\end{cases} `$

 – fixed`$\alpha,\lambda`$

 – self-normalizing ELU – keeps activations mean/var stable through layers. Often paired with AlphaDropout to preserve self-normalizing behavior.

-------------------

## **Softplus** –

# $` f(x) = \log(1 + e^x) `$

 – smooth approximation of ReLU – always differentiable – never truly zero, so avoids dead neurons. Also useful to enforce positivity in model parameters.

-------------------

## **Softsign** –

# $` f(x) = \frac{x}{1 + |x|} `$

 – sigmoid-like curve – maps to`$(-1,1)$

 – gentler slope for large`$|x|$, used occasionally for stable training. It has smaller gradients for large values, potentially slowing training if poorly tuned.

----------------------

## **Swish (SiLU)** –

# $` f(x) = x\,\sigma(x) `$

–`$\sigma`$: sigmoid – input scaled by its sigmoid – non-monotonic, outperforms ReLU in some deep models (used in EfficientNet). Emerging as a default in advanced architectures for enhanced gradient flow.

-------------------

## **Mish** –

# $` f(x) = x \tanh(\ln(1+e^x)) `$

 – smooth, non-monotonic – retains small negative values – empirically boosts vision model accuracy (used in some YOLO versions). Introduced by Diganta Misra, it can sometimes outperform Swish.

-------------------

## **Hard Sigmoid** –

# $` f(x) = \max(0,\,\min(1,\;\frac{x+1}{2})) `$

 – piecewise-linear sigmoid approximation – cheaper to compute – used in mobileNets for efficiency. Despite being approximate, it delivers good results with lower overhead.

-------------------

## **Hard Tanh** –

# $` f(x) = \max(-1,\,\min(1,\,x)) `$

 – clips`$x$` to`$[-1,1]$

 – fast approximation of`$\tanh`$

 – used in some recurrent nets or as activation clamp. Saturates quickly outside [-1,1], yet aids hardware efficiency.

-------------------

## **Hard Swish** –

# $` f(x) = x \frac{\max(0,\min(6,x+3))}{6} `$

 – linear approximation of swish – implemented as`$x * \text{ReLU6}(x+3)/6$

 – efficient for mobile CPU (MobileNetV3). Delivers near-Swish performance with simpler piecewise-linear ops.

-------------------

## **Softmax** –

# $` \sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}} `$

–`$z`$: logits – converts vector to probability distribution – used in output layer for multi-class classification. Temperature scaling can adjust smoothness, affecting model confidence.

-------------------

## **Maxout** –

# $` f(x) = \max_{j\in\{1,\dots,k\}} (w_j^T x + b_j) `$

–`$k`$: linear pieces – learns piecewise linear activation by selecting maximal affine output – increases model capacity per neuron. It can capture convex hulls of multiple linear functions, albeit at higher parameter costs.

-------------------

## **ReLU6** –

# $` f(x) = \min(\max(0,x), 6) `$

 – ReLU capped at 6; crucial for fixed-point inference efficiency on mobile devices (like MobileNets), preventing activations from growing unbounded while maintaining compatibility with hardware constraints.

-------------------

## **CELU (Continuously Differentiable ELU)** –

# $` f(x) = \max(0,x) + \min(0, \alpha(\exp(x/\alpha) - 1)) `$

–`$\alpha`$: scale (>0) – A variant of ELU designed to be continuously differentiable everywhere, including at`$x=0$, which can potentially aid optimization stability in some contexts.

-------------------

## **ISRU (Inverse Square Root Unit)** –

# $` f(x) = \frac{x}{\sqrt{1 + \alpha x^2}} `$

–`$\alpha`$: scale parameter – A non-monotonic function that implicitly performs a form of normalization; can sometimes achieve faster convergence than ReLU variants in specific architectures.

-------------------

## **ISRLU (Inverse Square Root Linear Unit)** –

# $` f(x) = \begin{cases} x & x \ge 0 \\ \frac{x}{\sqrt{1 + \alpha x^2}} & x < 0 \end{cases} `$

–`$\alpha`$: scale parameter – Combines linear behavior for positive inputs (like ReLU) with the ISRU function for negative inputs, aiming to prevent dying neurons while offering ISRU's properties.

-------------------

## **APL (Adaptive Piecewise Linear)** –

# $` f(x) = \max(0, x) + \sum_{s=1}^S a_i^s \max(0, -x + b_i^s) `$

–`$a_i^s, b_i^s`$: learnable parameters – Learns a hinge-shaped piecewise linear activation function during training, allowing the network to approximate complex functions more flexibly than fixed activations.

-------------------

## **Gaussian** –

# $` f(x) = \exp(-x^2) `$

 – Bell-shaped curve centered at 0; primarily used in Radial Basis Function (RBF) networks or as a component in attention mechanisms, sensitive to inputs near zero.

-------------------

## **Bipolar Sigmoid** –

# $` f(x) = \frac{1 - e^{-x}}{1 + e^{-x}} `$

 – Equivalent to Tanh, mapping inputs to`$(-1, 1)$; historically used alongside binary (-1, 1) targets, providing zero-centered outputs beneficial for gradient flow compared to standard sigmoid.

-------------------

## **LogSigmoid** –

# $` f(x) = \log(\sigma(x)) = \log\left(\frac{1}{1 + e^{-x}}\right) `$

–`$x`$: input – Computes the logarithm of the sigmoid function; often used directly in loss functions (like binary cross-entropy) for numerical stability, especially with large negative inputs.

-------------------

## **HardShrink** –

# $` f(x) = \begin{cases} x & \text{if } x > \lambda \\ x & \text{if } x < -\lambda \\ 0 & \text{otherwise} \end{cases} `$

–`$\lambda`$: threshold – Sets values within`$[-\lambda, \lambda]$` to zero, keeping others unchanged; used in sparse coding and related algorithms to promote sparsity by thresholding.

-------------------

## **SoftShrinkage** –

# $` f(x) = \begin{cases} x - \lambda & \text{if } x > \lambda \\ x + \lambda & \text{if } x < -\lambda \\ 0 & \text{otherwise} \end{cases} `$

–`$\lambda`$: threshold – Shrinks values towards zero by`$\lambda$` and thresholds values within`$[-\lambda, \lambda]$; central to iterative shrinkage-thresholding algorithms (ISTA) used in sparse signal recovery.

-------------------

## **TanhShrink** –

# $` f(x) = x - \tanh(x) `$

 – Subtracts the Tanh activation from the input; this function emphasizes values far from zero, finding use in specific signal processing or decomposition tasks within networks.

-------------------

## **Thresholded ReLU** –

# $` f(x) = \begin{cases} x & \text{if } x > \theta \\ 0 & \text{otherwise} \end{cases} `$

–`$\theta`$: threshold (>0) – A variant of ReLU where activation only occurs if the input exceeds a specific threshold`$\theta`$; useful when only significantly strong signals should propagate.

-------------------

## **CReLU (Concatenated ReLU)** –

# $` f(x) = [\max(0, x), \max(0, -x)]`$

 – Concatenates the output of ReLU applied to the input and its negation; doubles the output dimension but preserves complete information, potentially improving representational capacity early in networks.

-------------------

## **RReLU (Randomized Leaky ReLU)** –

# $` f(x) = \begin{cases} x & x \ge 0 \\ \alpha x & x < 0 \end{cases} `$` where $` \alpha \sim U(l, u) `$

 – Negative slope`$\alpha$` is randomly sampled from a uniform distribution during training, fixed during inference; acts as a regularizer reducing overfitting.

-------------------

## **GLU (Gated Linear Unit)** –

# $` f(x, W, V, b, c) = \sigma(xW + b) \odot (xV + c) `$

–`$\sigma`$: sigmoid,`$\odot`$: element-wise product – Uses a sigmoid gate to control information flow through a linear transformation; powerful in sequence modeling (e.g., Gated CNNs, Transformer variants).

-------------------

## **ReGLU (ReLU Gated Linear Unit)** –

# $` f(x, W, V, b, c) = \max(0, xW + b) \odot (xV + c) `$

–`$\odot`$: element-wise product – A GLU variant using ReLU for the gating mechanism instead of sigmoid; often performs competitively or better than standard GLU in Transformers.

-------------------

## **GeGLU (GELU Gated Linear Unit)** –

# $` f(x, W, V, b, c) = \text{GELU}(xW + b) \odot (xV + c) `$

–`$\odot`$: element-wise product – A GLU variant employing the smoother GELU activation for gating; used in some large language models for improved performance over ReLU/Sigmoid gates.

-------------------

## **SwiGLU (Swish Gated Linear Unit)** –

# $` f(x, W, V, b, c) = \text{Swish}(xW + b) \odot (xV + c) `$

–`$\odot`$: element-wise product – A GLU variant using the Swish (SiLU) activation for gating; demonstrated strong performance in recent large language models like PaLM, often outperforming other GLU variants.

-------------------

## **Softmin** –

# $` \text{softmin}(z)_i = \frac{e^{-z_i}}{\sum_j e^{-z_j}} `$

–`$z`$: logits vector – Similar to Softmax but emphasizes the *minimum* value by using negative exponents; useful in attention mechanisms or scenarios where selecting the least likely option is desired.

-------------------

## **Bent Identity** –

# $` f(x) = \frac{\sqrt{x^2 + 1} - 1}{2} + x`$

 – A simple non-linear function that behaves like identity for large positive`$x$` and approaches`$x/2$` for large negative`$x`$; provides non-linearity without saturation.

-------------------

## **SQNL (Squared Non-Linearity)** –

# $` f(x) = \begin{cases} 1 & x > 2 \\ x - x^2/4 & 0 \le x \le 2 \\ x + x^2/4 & -2 \le x < 0 \\ -1 & x < -2 \end{cases} `$

 – A piecewise quadratic function providing smooth transitions and bounded output`$[-1, 1]$; designed as an alternative to Tanh with potentially simpler computation.

-------------------

## **FReLU (Funnel ReLU)** –

# $` f(x, T(x)) = \max(x, T(x)) `$

–`$T(x)$: learnable spatial condition (Parametric Pooling) – Extends ReLU by adding a learnable, spatially-aware condition, allowing pixel-level modeling capacity; effective in visual recognition tasks by capturing spatial context.

-------------------

## **ACON (Activate Or Not)** –

# $` f(x) = (p_1 - p_2)x \cdot \sigma(\beta (p_1 - p_2)x) + p_2 x`$

–`$p_1, p_2, \beta`$: learnable parameters per channel – Generalizes activations from ReLU to Swish by learning whether to activate ($p_1$) or not ($p_2$) via a smooth maximum function controlled by`$\beta`$; adapts activation dynamically.

-------------------

## **DyReLU (Dynamic ReLU)** –

# $` f(x) = \max_{k=1..K} (\alpha^k(x) x + \beta^k(x)) `$

–`$\alpha^k(x), \beta^k(x)$: parameters generated by an attention mechanism based on input`$x`$

 – Creates input-dependent piecewise linear activations where slopes and intercepts are dynamically computed, making the activation highly context-aware.

-------------------

## **Snake Activation** –

# $` f(x) = x + \frac{1}{\alpha} \sin^2(\alpha x) `$

–`$\alpha`$: frequency parameter (can be learnable) – Adds a periodic component to the identity function, enabling the network to better learn periodic patterns present in data, particularly useful in physics-informed NNs or time-series.

-------------------

## **Gaussian Error Function (erf)** –

# $` \text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt`$

 – The integral of the Gaussian distribution; provides a smooth, sigmoid-like shape saturating at`$\pm 1$, related to GELU but simpler in form.

-------------------

## **Elliott Function (Approximation Sigmoid)** –

# $` f(x) = \frac{x}{1 + |x|} `$

 – (Note: This is identical to Softsign provided earlier, but sometimes referred to as Elliott). A computationally cheaper approximation to Tanh/Sigmoid; useful when hardware division is faster than exponentiation.

-------------------

## **LiSHT (Linearly Scaled Hyperbolic Tangent)** –

# $` f(x) = x \cdot \tanh(x) `$

 – Scales the input by its Tanh value; similar motivation to Swish (gating) but using Tanh instead of Sigmoid, resulting in a non-monotonic curve centered around zero.

-------------------

## **Serf (Sigmoid-Weighted Linear Unit with Error Function)** –

# $` f(x) = x \cdot \text{erf}(\ln(1+e^x)) `$

 – A smooth, non-monotonic activation inspired by Mish, but using the Gaussian Error Function (erf) instead of Tanh; aims for similar benefits with a different mathematical formulation.

-------------------

## **Max Sigmoid Unit (MSU)** - $` f(x, W, b) = \max_{i} (\sigma(W_i x + b_i)) `$` - Takes the maximum across several sigmoid units applied to the input; allows the neuron to select the most active sigmoidal response, increasing representational power.

-------------------

## **Shifted Softplus** - $` f(x) = \log(1 + e^{x - \text{shift}}) `$` - A Softplus function shifted horizontally; useful in contexts like variational autoencoders (VAEs) to control the mean or location of distributions represented by the activation.

-------------------

## **Rectified Exponential Unit (REU)** - $` f(x) = \begin{cases} x & x \ge 0 \\ \alpha (e^x - 1) & x < 0 \end{cases} `$` - (Note: This is identical to ELU provided earlier, sometimes just named REU). An alternative name for ELU, emphasizing its rectified nature for positive inputs and exponential for negative ones.

-------------------

## **Symmetric Elliott Activation** - $` f(x) = \frac{x}{1+|x|} `$` - (Note: This is identical to Softsign/Elliott provided earlier). A symmetric variant of the faster sigmoid approximation, mapping outputs to`$(-1, 1)$` like Tanh.

-------------------

## **Learnable Sigmoid-like Unit (LSU)** - $` f(x) = \frac{1}{(1 + |\beta x|^\gamma)^{1/\delta}} `$` -`$\beta, \gamma, \delta`$: learnable parameters - A flexible sigmoid/tanh-like activation whose shape (steepness, saturation points) can be learned during training, adapting to the specific task.

=====================================================================
### 📣 Loss Functions
=====================================================================

## **Mean Squared Error** – \( L = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2 \) – \(y_i\): true, \(\hat{y}_i\): pred – measures average squared regression error (penalizes large errors more). Often used for regression; its gradient is simpler than absolute errors, making it a common choice for gradient-based methods.

----------------------

## **Mean Absolute Error** – \( L = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i| \) – absolute difference – linear penalty on error magnitude – robust to outliers (L1 loss). Less sensitive to outliers than MSE but less smooth, potentially complicating gradient-based optimization.

----------------------

## **Binary Cross-Entropy** – \( L = -[\,y\log p + (1-y)\log(1-p)\,] \) – \(y\): label, \(p\): predicted prob – logistic loss for binary classification (convex, smooth surrogate to 0–1 loss). Directly tied to maximum likelihood under Bernoulli distribution, providing probabilistically sound metrics for binary predictions.

----------------------

## **Categorical Cross-Entropy** – \( L = -\sum_{c=1}^C y_c \log p_c \) – \(y_c\): one-hot label, \(p_c\): predicted prob – multi-class classification loss (maximizes likelihood of correct class). Equivalent to maximizing likelihood under a multinomial model, ensuring well-calibrated probabilities for multi-class tasks.

----------------------

## **Hinge Loss** – \( L = \max(0,\,1 - y\,f(x)) \) – \(y \in \{\pm1\}\), \(f(x)\): score – SVM margin loss – zero if correctly classified with margin, else linear penalty. Encourages a hard margin, but is not differentiable at the boundary, requiring subgradient or hinge-specific optimization techniques.

----------------------

## **Squared Hinge Loss** – \( L = \max(0,\,1 - y f(x))^2 \) – variation of hinge – stronger penalty on violations – used in some SVM implementations. Amplifies penalty near the decision boundary, providing stronger gradient feedback but possibly slowing convergence.

----------------------

## **Huber Loss** –

```math
\Huge
L =
\begin{cases}
\frac{1}{2}(y-\hat{y})^2 & |y-\hat{y}|\le \delta \\
\delta\,|y-\hat{y}| - \frac{1}{2}\delta^2 & |y-\hat{y}|>\delta
\end{cases}
```

 – quadratic for small errors, linear for large – robust regression loss (less sensitive to outliers). Commonly used in robust regression to handle outliers while retaining quadratic smoothing for small errors.

----------------------

## **KL Divergence** – \( D_{KL}(P\|Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)} \) – \(P\): true distrib, \(Q\): pred distrib – non-symmetric difference of distributions – used in VAEs, teacher-student distillation. Widely employed for distribution matching in tasks like language modeling and knowledge distillation, but grows infinite if \(Q=0\) where \(P>0\).

----------------------

## **Focal Loss** – \( L = -(1-p)^\gamma \log p \) – \(p\): predicted prob of true class, \(\gamma\): focusing param – down-weights easy examples, focuses on hard ones – useful for class imbalance (e.g. object detection). Often used in object detection to handle extreme class imbalance by down-weighting easy negatives through adjustable focusing parameter.

----------------------

## **Dice Loss** – \( L = 1 - \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i} \) – \(p_i\): pred, \(g_i\): ground truth (binary mask) – loss based on overlap (Dice coefficient) – common in segmentation tasks. Especially beneficial for medical image segmentation, as it directly maximizes overlap between predicted and ground-truth structures.

----------------------

## **Triplet Loss** – \( L = \max(0,\,d(a,p) - d(a,n) + m) \) – \(d\): distance, \(m\): margin – pulls anchor \(a\) closer to positive \(p\) than to negative \(n\) – trains embedding for ranking (face recognition, metric learning). Enhances metric learning by separating positive and negative pairs, crucial for tasks like face or image similarity search.

----------------------

## **Contrastive (InfoNCE) Loss** –

```math
\Huge
L = -\log \frac{\exp(\text{sim}(x,x^+)/\tau)}{\exp(\text{sim}(x,x^+)/\tau) + \sum_{x^-}\exp(\text{sim}(x,x^-)/\tau)}
```
 – \(\tau\): temperature – self-supervised loss that brings similar pairs \((x,x^+)\) together and separates negatives \(x^-\). Key in self-supervised learning setups, pushing similar samples together in embedding space and forming robust representations.

----------------------

## **Exponential Loss** – \( L = \exp(-y f(x)) \) – \(y\in\{\pm1\}\) – used in AdaBoost – heavily penalizes wrong predictions – leads to boosting weights update (upper bound on 0–1 loss). Exponential scaling intensifies the penalty for misclassified points, which can hasten overfitting if not carefully regularized.

----------------------

## **GAN Generator Loss** – \( L_G = -\mathbb{E}_{z}[\log D(G(z))] \) – \(D\): discriminator, \(G\): generator – encourages generator to fool discriminator (maximize \(\log D(G(z))\)) – drives fake data realism. GAN training can collapse if the generator only produces limited outputs that consistently fool the discriminator, hindering diversity.

----------------------

## **GAN Discriminator Loss** – \( L_D = -\mathbb{E}_{x}[\log D(x)] - \mathbb{E}_{z}[\log(1 - D(G(z)))] \) – outputs real vs fake – trains discriminator to distinguish real data from generator’s fakes. If discriminator overtakes generator too quickly, gradients vanish; balancing the training dynamics is crucial for stable GAN progress.

----------------------

## **Wasserstein Loss** – \( L = \mathbb{E}_{x}[D(x)] - \mathbb{E}_{z}[D(G(z))] \) – critic \(D\) tries to maximize this, \(G\) to minimize – yields Earth-Mover distance between real and generated distributions (stabilizes GAN training). Needs Lipschitz continuity for the critic, enforced via weight clipping or gradient penalty, to ensure meaningful distance computation.

----------------------

## **VAE Loss (ELBO)** – \( L = \mathbb{E}_{q(z|x)}[-\log p(x|z)] + D_{KL}(q(z|x)\|p(z)) \) – \(q(z|x)\): encoder, \(p(x|z)\): decoder – reconstruction error + KL regularizer – balances data fit and latent simplicity. Enables unsupervised data generation while learning smooth latent spaces, supporting interpolation and semi-supervised tasks.

----------------------

## **CTC Loss** – \( L = -\ln \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^T P(\pi_t|x) \) – sums probabilities of all label sequences \(\pi\) mapping to target \(y\) (collapsing repeats/blanks) – used for sequence alignment in speech/OCR without pre-segmentation. Allows end-to-end training in speech recognition systems by handling variable input lengths without explicit time alignment.

----------------------

## **ArcFace Loss** –

```math
\Huge
L = -\log \frac{e^{s(\cos(\theta_y + m))}}{e^{s(\cos(\theta_y + m))} + \sum_{j \neq y} e^{s \cos \theta_j}}
```

 – \(\theta_y\): angle between feature and true-class weight, \(m\): margin, \(s\): scale – adds an angular margin to softmax – greatly improves face recognition discriminability. Margin-based angular optimization significantly improves intra-class compactness and inter-class separability, boosting verification performance.

----------------------

## **Poisson NLL Loss** – \( L = \hat{\lambda} - y \log \hat{\lambda} \) – \(\hat{\lambda}\): predicted rate, \(y\): observed count – negative log-likelihood for Poisson regression – used for modeling count data. Useful when observed responses are counts, but excessive zero counts may require zero-inflated Poisson or hurdle models.

----------------------

## **Quantile Loss** – \( L_{\tau}(e) = \max(\tau e,\;(\tau-1)e) \) where \(e=y-\hat{y}\) – \(\tau \in (0,1)\): quantile – asymmetric linear penalty – trains models to estimate specified quantile (used in forecasting). Aids in estimating uncertainty by learning quantiles, enabling more nuanced predictions like upper or lower bounds.

### **Weighted Cross-Entropy**

```math
\Huge
L = -\sum_{i=1}^N w_i \,\Big[y_i \log p_i + (1 - y_i)\log (1 - p_i)\Big]
```
Balances classes by applying weight \(w_i\) to each example. Especially valuable for imbalanced datasets, it prevents majority classes from dominating training, improving minority-class recall and stabilizing convergence in skewed classification scenarios.



### **Tversky Loss**

```math
\Huge
L = 1 - \frac{\sum_i p_i g_i}{\sum_i p_i g_i + \alpha \sum_i p_i (1-g_i) + \beta \sum_i g_i (1-p_i)}
```
Generalizes Dice loss by weighting false positives (\(\alpha\)) and false negatives (\(\beta\)) differently. Particularly popular in medical image segmentation to handle pronounced class imbalance and control sensitivity vs specificity trade-off.



### **IoU (Jaccard) Loss**

```math
\Huge
L = 1 - \frac{\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i - \sum_i p_i g_i}
```
Directly maximizes intersection-over-union between prediction \(p_i\) and ground truth \(g_i\). Helpful for segmentation tasks by emphasizing overlap quality, leading to robust boundary alignment and reducing over-segmentation or under-segmentation issues.



### **Label Smoothing Cross-Entropy**

```math
\Huge
L = -\sum_{c=1}^C \bigl((1-\epsilon) y_c + \tfrac{\epsilon}{C}\bigr) \log p_c
```
Replaces the one-hot target with a slightly “smoother” distribution. This prevents overconfidence, improves generalization, and mitigates overfitting by penalizing models that assign all probability mass to a single class.



### **SoftMargin Loss**

```math
\Huge
L = \sum_{i=1}^N \log\bigl(1 + \exp(-y_i f(x_i))\bigr)
```
A smooth variant of hinge loss that uses the logistic function. It ensures continuous gradient updates at the margin boundary, simplifying optimization while preserving margin-based classification principles from SVM-like approaches.



### **Log-Cosh Loss**

```math
\Huge
L = \sum_{i=1}^N \log\bigl(\cosh(\hat{y}_i - y_i)\bigr)
```
Applies the hyperbolic cosine of prediction errors, combining smoothness and robustness. Large errors are penalized less severely than MSE, helping in regression tasks with occasional spikes or outliers while preserving a differentiable form.



### **Cosine Embedding Loss**

```math
\Huge
L =
\begin{cases}
1 - \cos(\mathbf{x}_1, \mathbf{x}_2) & \text{if } y = 1,\\
\max\bigl(0,\cos(\mathbf{x}_1, \mathbf{x}_2) - m\bigr) & \text{if } y = -1
\end{cases}
```
Encourages similar samples (label \(y=1\)) to have high cosine similarity, while dissimilar pairs (label \(y=-1\)) remain below a margin \(m\). Widely used in text, image, and audio similarity tasks.



### **Margin Ranking Loss**

```math
\Huge
L = \sum_{i=1}^N \max\bigl(0, -y_i (f(x_i^+) - f(x_i^-)) + m\bigr)
```
Compares positive and negative pairs with margin \(m\). Used in ranking systems (e.g., search engines, recommendation) to order relevant items above irrelevant ones, guiding the model to learn meaningful relative comparisons.



### **Gradient Harmonized Mechanism (GHM) Loss**

```math
\Huge
L = \sum_i \phi(\nabla_i) \cdot \ell_i
```
Adapts training by weighting samples based on gradient density (\(\phi\)). Addresses imbalance in easy vs. hard examples, stabilizing the training process in object detection and classification where gradient contributions are highly skewed.



### **Perceptual Loss**

```math
\Huge
L = \sum_{l=1}^L \|\Phi_l(\hat{x}) - \Phi_l(x)\|^2
```
Compares features \(\Phi_l(\cdot)\) extracted by a deep network at layer \(l\). Encourages generated outputs to match high-level perceptual cues. Used in image-to-image translation and style transfer to improve visual realism.



### **Style Loss**

```math
\Huge
L = \sum_{l=1}^L \bigl\|\;G(\Phi_l(\hat{x})) - G(\Phi_l(x))\bigr\|^2
```
Minimizes the difference in Gram matrices \(G(\cdot)\) of feature maps. Central to neural style transfer, preserving style statistics (textures, colors) independent of content structure, yielding visually coherent style transformations.



### **Wing Loss**

```math
\Huge
L =
\begin{cases}
w\ln(1 + \frac{|x|}{\epsilon}) & |x| < w, \\
|x| - C & \text{otherwise}
\end{cases}
```
Developed for facial landmark detection, reducing sensitivity around small errors and linearly penalizing larger discrepancies. Improves accuracy for fine-grained alignment tasks with fewer outlier distortions than standard L1 or L2 losses.



### **SphereFace Loss**

```math
\Huge
L = -\log \frac{e^{\|\mathbf{W}_y\|\|\mathbf{x}\|\cos(m\theta_y)}}{e^{\|\mathbf{W}_y\|\|\mathbf{x}\|\cos(m\theta_y)} + \sum_{j \neq y} e^{\|\mathbf{W}_j\|\|\mathbf{x}\|\cos(\theta_j)}}
```
Introduces a multiplicative angular margin \(m\) in feature space for face recognition. Encourages highly discriminative embeddings by pushing intra-class angles closer and separating inter-class angles, improving verification performance.



### **Tweedie Loss**

```math
\Huge
L = -\sum_{i=1}^N \Big[y_i \log(\hat{\mu}_i) - \frac{\hat{\mu}_i^{1-p}}{1-p}\Big]
```
Models compound Poisson-Gamma distributions (parameter \(p\)). Suited for insurance and count-based forecasting where outcomes can be sparse yet exhibit heavy tails. Flexibly interpolates between Poisson, Gamma, and Gaussian behaviors.



### **Correntropy Loss**

```math
\Huge
L = \sum_{i=1}^N \bigl[1 - \kappa_\sigma(\hat{y}_i - y_i)\bigr], \quad \kappa_\sigma(z)=\exp\Bigl(-\frac{z^2}{2\sigma^2}\Bigr)
```
Assesses similarity through a Gaussian kernel, robustly filtering out heavy outliers. Combines elements of information theory and robust statistics, making it useful in applications needing strong tolerance to noise.



### **Student’s t NLL**

```math
\Huge
L = -\sum_{i=1}^N \log \left[ \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1+\frac{(\hat{y}_i - y_i)^2}{\nu}\right)^{-\frac{\nu+1}{2}} \right]
```

Assumes residuals follow a heavy-tailed Student’s t distribution with \(\nu\) degrees of freedom. Captures outliers more gracefully than Gaussian-based losses, providing stability when data contain large deviations from the mean.



### **Ordinal Cross-Entropy**

```math
\Huge
L = -\sum_{k=1}^{K-1} \Bigl[y_{\le k}\log p_{\le k} + (1 - y_{\le k})\log (1 - p_{\le k})\Bigr]
```
Handles ordinal data by decomposing class probabilities into cumulative splits. Useful when label categories have a natural order, enabling models to penalize misclassifications based on ordinal distance rather than treating classes as independent.



### **CosFace Loss**

```math
\Huge
L = -\log \frac{e^{s(\cos(\theta_y) - m)}}{e^{s(\cos(\theta_y) - m)} + \sum_{j \neq y} e^{s \cos(\theta_j)}}
```
Applies an additive cosine margin \(m\) with a scale \(s\). Strengthens the decision boundary in face-recognition embeddings, enhancing class separation and improving verification accuracy under constrained intra-class variations.



### **Multi-Label Soft Margin Loss**

```math
\Huge
L = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C \Big[y_{ic} \log \sigma(\hat{y}_{ic}) + (1 - y_{ic}) \log\bigl(1 - \sigma(\hat{y}_{ic})\bigr)\Big]
```
Optimizes multi-label classification by applying a sigmoid \(\sigma\) per class, allowing each instance to have multiple positive labels. Crucial in tasks like tagging and image annotation, where categories are not mutually exclusive.



### **Class Balanced Loss**

```math
\Huge
L = - \sum_{c=1}^C \beta_c \Big[y_c \log p_c + (1 - y_c) \log(1 - p_c)\Big]
```
Reweights cross-entropy using inverse effective sample sizes (\(\beta_c\)) for each class. Mitigates overfitting in imbalanced scenarios by correcting the bias introduced by classes with too few or too many samples.





=====================================================================
### 📣 Optimization Algorithms
=====================================================================

## **Gradient Descent Step** –

# $` \theta_{t+1} = \theta_t - \eta\,\nabla_\theta L`$

–`$\eta`$: learning rate – moves parameters opposite to loss gradient – foundational iterative optimizer. Despite high dimensionality, gradient descent can often escape poor local minima thanks to the geometry of neural loss surfaces.

----------------------

## **Momentum Update** –

# $` v_t = \beta v_{t-1} + \nabla_\theta L_t,\;\; \theta_{t+1} = \theta_t - \eta\,v_t`$

–`$\beta`$: momentum factor – accumulates gradient velocity – accelerates descent in consistent direction, damps oscillations. It reduces variance in parameter updates, leading to smoother paths in complex error landscapes.

----------------------

## **Nesterov Accelerated Grad.** –

# $` v_t = \beta v_{t-1} + \nabla_\theta L(\theta_t - \beta v_{t-1}),\;\; \theta_{t+1} = \theta_t - \eta\,v_t`$

 – looks ahead before gradient – often improves momentum convergence. It corrects velocity before computing the gradient, typically yielding sharper speed gains than classical momentum in many tasks.

----------------------

## **AdaGrad** –

# $` r_t = r_{t-1} + g_t^2,\;\; \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_t}+\epsilon}\,g_t`$

–`$g_t`$: gradient,`$r_t`$: accumulative square – per-parameter learning rate adapts (decreases) with accumulated gradient magnitude – good for sparse features. Its adaptive steps can cause early learning rate decay, so high-dimensional tasks sometimes require advanced variants or manual damping.

----------------------

## **RMSProp** –

# $` r_t = \rho r_{t-1} + (1-\rho) g_t^2,\;\; \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_t}+\epsilon}\,g_t`$

–`$\rho`$: decay – like AdaGrad but with exponential moving average of squared gradients – prevents learning rate from decaying too drastically. Its exponential moving average stabilizes updates but can still be sensitive to tuning the decay parameter across different problems.

----------------------

## **Adam Optimizer** –

# $` m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\;\; v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 $` (with bias correction), $` \theta_{t+1} = \theta_t - \frac{\eta\,m_t}{\sqrt{v_t}+\epsilon} `$

 – combines momentum (first moment`$m_t`$) and AdaGrad (second moment`$v_t`$) – widely used adaptive optimizer for faster convergence. Bias correction is crucial; ignoring it can slow early training phases and degrade performance on large-scale tasks.

----------------------

## **AdamW (Decoupled Weight Decay)** –

# $` \theta_{t+1} = \theta_t - \eta \Big( \frac{m_t}{\sqrt{v_t}+\epsilon} + \lambda \theta_t \Big)$

–`$\lambda`$: weight decay – Adam with L2 regularization applied separately (prevents Adam from diminishing regularization effect). Separating weight decay from gradient-based updates preserves the intended regularization strength, especially beneficial in large language models.

----------------------

## **AdaMax** –

# $` u_t = \max(\beta_2 u_{t-1},\,|g_t|),\;\; \theta_{t+1} = \theta_t - \frac{\eta}{1-\beta_1^t} \frac{m_t}{u_t} `$

 – Adam variant using`$\ell_\infty$` norm of gradients ($u_t`$) instead of`$v_t`$

 – more stable when second moment diverges. Using the infinity norm can mitigate outliers, helping networks remain more robust under extreme gradient bursts.

----------------------

## **Nadam** –

# $` m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t $` (similar to Adam), $` \theta_{t+1} = \theta_t - \eta \Big( \frac{\beta_1 m_t + (1-\beta_1)g_t}{\sqrt{v_t}+\epsilon} \Big) `$

 – incorporates Nesterov momentum into Adam – slightly faster convergence in some cases. Its gradient lookahead synergy can reduce the need for meticulous manual tuning of momentum-related hyperparameters.

----------------------

## **AMSGrad** – (Adam with max-correction) `$` \hat{v}_t = \max(\hat{v}_{t-1},\,v_t),\;\; \theta_{t+1} = \theta_t - \frac{\eta\,m_t}{\sqrt{\hat{v}_t}+\epsilon} `$

 – ensures non-increasing second moment`$\hat{v}_t`$

 – provides convergence guarantee by preventing`$v_t$` from decreasing. Ensuring a non-increasing second moment helps stabilize updates, especially in adversarial or highly non-stationary learning environments.

----------------------

## **Lion Optimizer** –

# $` m_t = \beta_1 m_{t-1} + (1-\beta_1)\mathrm{sign}(g_t),\;\; \theta_{t+1} = \theta_t - \eta\, m_t`$

 – uses sign of gradients with momentum – memory-efficient like SGD, but benefits from sign-based updates (Evolved Sign Momentum). Its sign-based approach can accelerate training but may lose some sensitivity to gradient magnitude details.

----------------------

## **Sophia Optimizer** –

# $` H_t = \beta_2 H_{t-1} + (1-\beta_2)\nabla^2 L_t,\;\; \theta_{t+1} = \theta_t - \eta\,\frac{g_t}{\text{diag}(H_t)+\epsilon} `$

 – approximates diagonal Hessian`$H_t$` to precondition gradients – second-order method for faster convergence in large language model training. Its Hessian approximation needs careful maintenance; inaccurate diagonal estimates can hamper training stability.

----------------------

## **Newton’s Method** –

# $` \theta_{t+1} = \theta_t - H^{-1} \nabla_\theta L`$

–`$H`$: Hessian matrix – uses second-order curvature info to jump to optimum in one step for quadratic loss – fast convergence near minima (costly in high dimensions). Hessian inversion scales poorly, so practical usage often relies on approximations or special structures in the loss.

----------------------

## **L-BFGS** – uses past gradients to approximate Hessian – no single formula (iterative) – quasi-Newton optimization storing limited history – efficient second-order method for medium-scale problems (used in logistic reg, etc.). Though memory-efficient, it can still be expensive for very large models, requiring iterative updates to approximate curvature.

----------------------

## **Cosine Decay** –

# $` \eta(t) = \eta_{\min} + \frac{\eta_{\max}-\eta_{\min}}{2}\Big(1 + \cos\frac{\pi t}{T}\Big) `$

–`$t`$: current step,`$T`$: total steps – gradually anneals learning rate to minimum following a cosine curve – encourages better final convergence. By gradually reducing the learning rate, it can help avoid overshooting minima, often stabilizing later training epochs.

----------------------

## **Gradient Clipping** –

# $` g \leftarrow \min\!\Big(1,\;\frac{\tau}{\|g\|_2}\Big)\,g`$

–`$\tau`$: threshold – rescales gradient if norm exceeds`$\tau`$

 – prevents exploding gradients in RNNs and deep networks. This technique is crucial for training stability, especially in sequences, but can slow convergence if threshold is too small.

----------------------

## **Stochastic Weight Averaging** –

# $` \bar{\theta} = \frac{1}{M}\sum_{t=T_0}^{T_0+M-1} \theta_t`$

 – average of weights over last`$M$` epochs – produces flat minimum solution – often improves generalization (less overfitting). Averaging multiple local minima flattens the loss surface, boosting robustness and reducing variance among parameter configurations.

----------------------

## **Linear Warmup** –

# $` \eta(t) = \eta_0 \frac{t}{T_w} `$` for`$t<T_w$` (then use base schedule) – gradually increase learning rate from 0 to`$\eta_0$` over`$T_w$` steps – avoids instability from too-high initial lr in early training. It helps the optimizer adapt gradually, preventing destructive large updates during the sensitive startup phase of training.

----------------------

## **Lookahead** – \( \theta_{\text{fast}} \leftarrow \theta - \alpha \nabla_\theta L(\theta),\;\; \theta \leftarrow \theta + \beta \bigl(\theta_{\text{fast}} - \theta \bigr) \) – maintains a slow parameter copy alongside fast updates, periodically syncing them. It often enhances stability and convergence, preventing oscillations and yielding more robust optima across deep architectures with large batch sizes.

----------------------

## **Adadelta** –

```math
\Huge
\Delta \theta_t = -\,\frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}}\; g_t,
\quad
E[\Delta \theta^2]_t = \rho \,E[\Delta \theta^2]_{t-1} \;+\; (1-\rho)\,\Delta \theta_t^{2}
```
– eliminates the need for a manually tuned learning rate. It dynamically adjusts step sizes based on historical gradients, remaining useful even when gradients vary widely between different layers or tasks.

----------------------

## **RAdam (Rectified Adam)** – combines Adam’s moments with a “variance rectification” term. When the running variance of gradients is small, it modifies the effective step size:

```math
\Huge
\theta_{t+1} = \theta_t \;-\; \eta \,\frac{m_t}{\sqrt{\widehat{v}_t} + \epsilon} \times \mathrm{rectification}(t)
```
– often mitigates unstable early steps, improving overall training reliability.

----------------------

## **AdaBound** –

```math
\Huge
\eta_t = \mathrm{clip}\!\Bigl(\tfrac{\eta}{\sqrt{\widehat{v}_t} + \epsilon},\;\alpha_{\text{low}},\,\alpha_{\text{high}}\Bigr),
\quad
\theta_{t+1} = \theta_t \;-\; \eta_t \, m_t
```
– bounds each adaptive learning rate within a predefined range, preventing overly large or tiny updates. This typically blends fast initial learning (like Adam) with the stability of conventional SGD at later stages.

----------------------

## **AccSGD** –

```math
\Huge
\theta_{t+1} = \theta_t \;-\; \eta \,\Bigl( \nabla L(\theta_t) + \gamma \,\sum_{i=1}^k \alpha_i \,(\theta_t - \theta_{t-i})\Bigr)
```
– an accelerated variant of SGD that uses historical parameter states in the update. It can converge faster than plain SGD, especially on convex or smooth problems, by reducing gradient variance over iterations.

----------------------

## **Adafactor** – factorizes second-moment estimation:

```math
\Huge
R_t = \beta_2 R_{t-1} + (1-\beta_2)\,\mathrm{diag}(g_t^2),\quad
C_t = \beta_2 C_{t-1} + (1-\beta_2)\,\mathrm{row\_sum}(g_t^2),
\quad
\theta_{t+1} = \theta_t - \eta \,\frac{m_t}{\sqrt{R_t \,C_t} + \epsilon}
```
– reduces memory usage significantly. Commonly used in large language models (e.g., Transformers), it preserves Adam-like adaptivity with lower computational overhead.

----------------------

## **Apollo** – performs a local quadratic approximation of the objective:

```math
\Huge
\theta_{t+1}
= \theta_t \;-\; \eta \,\Bigl(\alpha_t + \|g_t\|^2\Bigr)^{-1} \,g_t
```
– aims to stabilize updates without storing full second-order statistics. In practice, it can outperform standard first-order methods on some deep networks by better matching curvature.

----------------------

## **MadGrad** – accumulates both gradients and their squares:

```math
\Huge
x_{t+1} = x_t \;-\; \gamma \,\frac{\sum_{\tau=1}^t g_\tau}{\sqrt{\sum_{\tau=1}^t g_\tau^2 + \epsilon}}
```
– designed for momentum-like stability with a unique scaling approach. It can resolve certain training instabilities in modern deep architectures, particularly where adaptive methods underperform.

----------------------

## **Shampoo** – constructs matrix preconditioners for each layer:

```math
\Huge
G_t^{(i)} = \beta\,G_{t-1}^{(i)} + (1-\beta)\,\bigl(\nabla_{\theta^{(i)}} L(\theta_t)\bigr)\bigl(\nabla_{\theta^{(i)}} L(\theta_t)\bigr)^T,
\quad
\theta^{(i)}_{t+1} = \theta^{(i)}_t - \eta\,P_t^{(i)}\,\nabla_{\theta^{(i)}} L(\theta_t)
```
– adaptively adjusts update directions in a full-matrix manner. Though more expensive than scalar methods, it can converge faster in large neural networks by capturing cross-feature curvature.

----------------------

## **K-FAC** (Kronecker-Factored Approximate Curvature) – approximates the Fisher information matrix as a Kronecker product in layer blocks:

```math
\Huge
F(\theta) \approx A(\theta) \otimes G(\theta),
\quad
\theta_{t+1} = \theta_t - \eta\,F(\theta_t)^{-1}\,\nabla_\theta L(\theta_t)
```
– reduces second-order update cost significantly. It is popular in large-scale training (e.g., deep reinforcement learning) for leveraging structured approximations to improve convergence.

----------------------

## **Natural Gradient Descent** –

```math
\Huge
\theta_{t+1} = \theta_t \;-\;\eta \,F(\theta_t)^{-1}\,\nabla_\theta L(\theta_t)
```
where \(F(\theta)\) is the Fisher information matrix. It aligns updates with the underlying manifold of model parameters, often resulting in more stable progress, especially in probabilistic models and policy gradients.

----------------------

## **SVRG** (Stochastic Variance Reduced Gradient) –

```math
\Huge
\theta_{t+1}
= \theta_t \;-\; \eta \,\Bigl(\nabla f(\theta_t) - \nabla f(\tilde{\theta}) \;+\; \tfrac{1}{n}\sum_{i=1}^n \nabla f_i(\tilde{\theta})\Bigr)
```
– periodically calculates a “full” gradient at a reference point \(\tilde{\theta}\). This reduces gradient variance in stochastic updates, often speeding convergence on large datasets.

----------------------

## **SARAH** – maintains a recursive gradient estimate:

```math
\Huge
v_{t+1} = v_t + \nabla f(\theta_{t+1}) \;-\; \nabla f(\theta_t),
\quad
\theta_{t+1} = \theta_t \;-\; \eta\,v_t
```
– fosters variance reduction without storing all gradients. Useful when data are large but also beneficial for smoother, more stable training trajectories than plain SGD.

----------------------

## **SPIDER** – another stochastic variance reduction method:

```math
\Huge
v_t = v_{t-1} \;+\; \tfrac{1}{b}\,\sum_{i=1}^b \bigl(\nabla f_i(\theta_t) - \nabla f_i(\theta_{t-1})\bigr),
\quad
\theta_{t+1} = \theta_t - \eta\,v_t
```
– updates a small batch’s gradient incrementally, stabilizing estimates. Known for strong theoretical convergence rates on finite-sum problems, it can reduce computational overhead compared to full-batch methods.

----------------------

## **PSO (Particle Swarm Optimization)** –

```math
\Huge
v_{t+1} = \omega \,v_t \;+\; c_1 r_1 \,(p_t - x_t) \;+\; c_2 r_2 \,(g_t - x_t),
\quad
x_{t+1} = x_t + v_{t+1}
```
– swarm-inspired population method. Each particle tracks personal best \(p_t\) and global best \(g_t\). It avoids gradient calculations, making it valuable for black-box optimization tasks with complex or noisy objectives.

----------------------

## **CMA-ES (Covariance Matrix Adaptation - Evolution Strategy)** –

```math
\Huge
m_{t+1} = m_t + \mu_{\text{eff}} \sum_{k=1}^\lambda w_k (x_k - m_t),
\quad
C_{t+1} = \mathrm{cov\_update}\bigl(C_t,\dots\bigr)
```
– evolves a Gaussian search distribution, adapting its covariance to explore promising directions. Widely used in reinforcement learning or when gradients are unavailable or unreliable.

----------------------

## **Evolution Strategies** – a broader class of gradient-free algorithms:

```math
\Huge
\theta_{t+1} = \theta_t + \alpha \,\frac{1}{\mu}\sum_{k=1}^\mu r_k \,\nabla_{\theta}\log \pi_\theta(x_k)
```
– rely on sampling and performance-based selection. They optimize parameters by simulating population-based mutations, useful for non-differentiable or deceptive objective surfaces.

----------------------

## **TRPO (Trust Region Policy Optimization)** –

```math
\Huge
\max_\theta \;\mathbb{E}\!\Bigl[\tfrac{\pi_{\theta}(a\mid s)}{\pi_{\theta_{\mathrm{old}}}(a\mid s)}\,A^{\pi_{\theta_{\mathrm{old}}}}(s,a)\Bigr]
\quad
\text{subject to } D_{\mathrm{KL}}\bigl(\pi_{\theta_{\mathrm{old}}}\|\pi_\theta\bigr) < \delta
```
– constrains policy updates in a trust region to prevent large destructive steps. It underpins stable reinforcement learning, often outperforming vanilla policy gradients on complex tasks.

----------------------

## **FTRL (Follow-The-Regularized-Leader)** –

```math
\Huge
\theta_{t+1} = \arg\min_\theta \,\Bigl(\sum_{\tau=1}^t g_\tau^T \theta + \tfrac{1}{\alpha}\,R(\theta)\Bigr)
```
– integrates a regularization term each step to balance accumulated gradient information. Often used in sparse, online learning contexts (like ad click prediction) for controlled and robust parameter updates.

----------------------

## **SAGA** –

```math
\Huge
\theta_{t+1} = \theta_t - \eta \,\Bigl(\nabla f_i(\theta_t) - \nabla f_i(\alpha_i) + \tfrac{1}{n}\!\sum_{j=1}^n \nabla f_j(\alpha_j)\Bigr)
```
– a stochastic gradient method with memory. It maintains a stored gradient for each sample, updating one at a time, achieving fast linear convergence on smooth, strongly convex problems without full batch passes.

=====================================================================
### 📣 Regularization Techniques
=====================================================================

## **L2 Regularization** –

# $` \Omega(w) = \frac{\lambda}{2}\sum_i w_i^2`$

–`$\lambda`$: strength – penalizes large weights (weight decay) – reduces variance and overfitting by shrinking parameters. It also acts as a smooth penalty that can simplify the optimization landscape by encouraging smaller parameter magnitudes.

----------------------

## **L1 Regularization** –

# $` \Omega(w) = \lambda \sum_i |w_i|`$

 – Lasso penalty – encourages sparsity in weights (drives small weights to zero) – feature selection effect. It can produce exact zero coefficients, making models more interpretable and aiding dimensionality reduction.

----------------------

## **Elastic Net** –

# $` \Omega(w) = \lambda_1 \sum_i |w_i| + \lambda_2 \sum_i w_i^2`$

 – convex combo of L1 and L2 – promotes sparsity while preserving groups of correlated features (used in regression). It balances the trade-off between feature selection and coefficient stability, reducing bias from correlated predictors.

----------------------

## **Dropout** –

# $` h_i' = \frac{h_i \cdot m_i}{p} `$` with`$m_i \sim \text{Bernoulli}(p)$

 – randomly drop units (set`$h_i$` to 0) during training, scale outputs by`$1/p`$

 – prevents co-adaptation, improves generalization in neural nets. It effectively enforces an ensemble-like training, reducing reliance on particular neurons for more robust predictions.

----------------------

## **Label Smoothing** –

# $` y^{\text{smooth}}_c = (1-\epsilon) y_c + \frac{\epsilon}{C} `$

–`$\epsilon`$: smoothing coeff – mixes one-hot label with uniform – prevents overconfidence by training on soft targets (improves calibration). It also mitigates label noise issues by distributing probability mass, yielding better uncertainty estimates under noisy conditions.

----------------------

## **Mixup Augmentation** –

# $` \tilde{x} = \lambda x_i + (1-\lambda) x_j,\;\; \tilde{y} = \lambda y_i + (1-\lambda) y_j`$

 – mixes two samples and labels – provides virtual training examples – improves robustness and generalization. It encourages linear behavior in-between training examples, which can help reduce memorization and improve boundary smoothness.

----------------------

## **Knowledge Distillation** –

# $` L = (1-\alpha)H(y,p_s) + \alpha T^2\,D_{KL}(p_t^{(T)}\|p_s^{(T)}) `$

–`$p_s,p_t`$: student/teacher predictions,`$T`$: temperature – student learns from softened teacher outputs + true labels – transfers knowledge to smaller model. Temperature scaling reveals dark knowledge in teacher logits, allowing the student to learn nuanced class relationships.

----------------------

## **Adversarial Training** –

# $` \min_\theta \max_{\|\delta\|\le \epsilon} L(f_\theta(x+\delta),y) `$

 – inner loop finds worst-case perturbation`$\delta$` of input (bounded by`$\epsilon`$), outer loop trains model to resist it – improves robustness to adversarial examples. It can significantly increase computational cost but provides a strong defense against gradient-based attacks in practical settings.

----------------------

## **Max-Norm Constraint** –

# $` w \leftarrow w \cdot \min\!\big(1,\;\frac{c}{\|w\|}\big) `$

–`$c`$: norm cap – projects weight vectors to a maximum length`$c`$

 – prevents overly large weights, stabilizes training (commonly used in dropout networks). This projection step can reduce gradient explosion risk, aiding stable convergence in deep architectures.

----------------------

## **Stochastic Depth** – in each layer:`$x_{out} = \begin{cases}f(x_{in}) & \text{with prob }p\\ x_{in} & \text{with prob }1-p\end{cases}$

 – randomly skip entire layers during training – acts like layer-wise dropout, allows training of very deep networks (used in ResNets). It speeds up training by reducing backprop depth on some passes, preserving representational power while cutting compute.

-------------------

## **Orthogonal Regularization** – \( \Omega(W) = \alpha \,\bigl\|W^\top W - I\bigr\|_F^2 \) – penalizes deviation from orthonormal weights – fosters uncorrelated embeddings, stabilizes backprop. It helps reduce overfitting by limiting redundant features and encouraging diverse representations.

-------------------

## **Group Lasso** – \( \Omega(w) = \sum_{g} \lambda_g \,\|w_{g}\|_2 \) – extension of Lasso that groups coefficients – fosters structured sparsity by jointly shrinking entire feature groups. It is beneficial in models with known group or block structure.

-------------------

## **Cutout** – \( x_{\text{masked}} = x \odot M \) – randomly masks out contiguous patches from the input – forces the model to rely on surrounding context, improving robustness. It combats overfitting by preventing reliance on specific spatial details.

-------------------

## **CutMix** –

```math
\Huge
\tilde{x} = M \odot x_i + (1 - M)\odot x_j, \quad
\tilde{y} = \lambda\,y_i + (1 - \lambda)\,y_j
```
– replaces random patches between images, mixing labels – encourages local invariance and yields stronger regularization than standard cutout or mixup.

-------------------

## **Virtual Adversarial Training** –

```math
\Huge
L_{\text{VAT}} = \Bigl\| f_{\theta}(x) \;-\; f_{\theta}\bigl(x + r_{\text{adv}}\bigr) \Bigr\|^2
```
where \( r_{\text{adv}} \) maximizes divergence – improves robustness to perturbations, especially in semi-supervised contexts.

-------------------

## **Entropy Regularization** – \( \Omega(p) = -\,\beta \sum_{c} p_{c} \log p_{c} \) – encourages broader output distributions by penalizing high-confidence predictions – mitigates overconfidence and can enhance exploration in reinforcement learning or stabilize classification training.

-------------------

## **Early Stopping** –

```math
\Huge
t^{\ast} \;=\; \arg \min_{t}\,\mathcal{L}_{\text{val}}(t)
```
– halts training once validation error plateaus or worsens – prevents the model from overfitting to noise. Its simplicity and effectiveness make it a go-to strategy in practice.

-------------------

## **Stochastic Weight Averaging (SWA)** –

```math
\Huge
w_{\text{SWA}} = \frac{1}{K} \sum_{k=1}^{K} w_k
```
– averages weights collected over training epochs – finds flatter minima for better generalization. It requires minimal code changes yet often yields significant performance gains.

-------------------

## **Zoneout** – with probability \(p\), keep hidden units unchanged; otherwise update – specifically for RNNs – resembles dropout but preserves some hidden states. It enhances sequence model robustness by balancing memory retention with stochastic regularization.

-------------------

## **Shake-Shake Regularization** – forward pass: \( y = \alpha \,f_{1}(x) + (1-\alpha)\,f_{2}(x) \); backward pass uses a different random \(\alpha'\) – injects noise in multi-branch networks – improves generalization in deeply branched architectures.

-------------------

## **Gradient Penalty** –

```math
\Huge
\Omega_{\text{GP}} = \lambda \,\bigl(\|\nabla_{\hat{x}}D(\hat{x})\|_{2} - 1\bigr)^{2}
```
– used in WGAN-GP to enforce the Lipschitz constraint – stabilizes GAN training and reduces mode collapse by penalizing large gradient norms.

-------------------

## **Jacobian Regularization** –

```math
\Huge
\Omega_{\text{Jac}} = \lambda \,\bigl\|\nabla_{x} f_{\theta}(x)\bigr\|_F^2
```
– penalizes high sensitivity of outputs w.r.t. inputs – promotes smoother decision boundaries. It lessens vulnerability to small input perturbations while aiding generalization in deep networks.

=====================================================================
### 📣 Normalization Layers
=====================================================================

## **Batch Normalization** –

# $` \hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},\;\; y = \gamma\,\hat{x} + \beta`$

–`$\mu_B,\sigma_B^2$: batch mean, var;`$\gamma,\beta`$: learnable scale/shift – normalizes layer activations per mini-batch – accelerates training and provides slight regularization. Be cautious with small batch sizes, as unstable estimates of mean/variance can degrade BN's effectiveness.

----------------------

## **Layer Normalization** –

# $` \hat{h}_i = \frac{h_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}},\;\; y_i = \alpha\,\hat{h}_i + \beta`$

 – normalize`$h$` across features for each sample (own mean`$\mu_L`$, var`$\sigma_L^2$) – stabilizes activations in RNNs/Transformers (independent of batch). It can be combined with residual connections to enhance gradient flow and reduce training instability in deep sequences.

----------------------

## **Instance Normalization** –

# $` \hat{x}_{n,c} = \frac{x_{n,c} - \mu_{nc}}{\sqrt{\sigma_{nc}^2+\epsilon}} `$

 – normalize per sample per channel (especially in image style transfer) – keeps sample statistics, removes instance-specific mean/var – used where batch stat not meaningful. It's particularly effective for style transfer tasks, allowing consistent style normalization across varying batch sizes.

----------------------

## **Group Normalization** –

# $` \hat{x}_{(g)} = \frac{x - \mu_{(g)}}{\sqrt{\sigma_{(g)}^2+\epsilon}} `$

 – divide channels into groups, normalize within each group – effective small-batch alternative to BN (used in object detection nets). It strikes a balance between instance and layer normalization, making it robust in multi-GPU and small-batch regimes.

----------------------

## **Weight Normalization** –

# $` \mathbf{w} = \frac{\mathbf{v}}{\|\mathbf{v}\|} g`$

–`$\mathbf{v}$: underlying weights,`$g`$: scale – parameterized re-normalization of weights – decouples length and direction – speeds up training by smoother optimization. By separating magnitude and direction, it can reduce dependency on weight initialization and facilitate better hyperparameter tuning.

----------------------

## **Spectral Normalization** –

# $` W_{\text{SN}} = \frac{W}{\sigma_{\max}(W)} `$

 – divides weights by their largest singular value`$\sigma_{\max}$

 – constrains Lipschitz constant of layers – used in GAN discriminators to stabilize training (controls weight scale). Helps maintain stable training by preventing the discriminator from overpowering the generator, fostering more balanced adversarial interplay.

-------------------

## **Batch Renormalization** –

```math
\Huge
\hat{x}^{(i)} = \frac{x^{(i)} - \hat{\mu}_B}{\sqrt{\hat{\sigma}_B^2 + \epsilon}}
\quad\text{where}\quad
\hat{\mu}_B = r\,\mu_B + d,\quad
\hat{\sigma}_B^2 = \frac{\sigma_B^2}{r}
```
– adjusts batch statistics via running estimates, aiding stable training when batch sizes vary widely. It bridges the gap between batch-level and sample-level normalization, particularly beneficial for small or dynamically changing mini-batches.

---

## **Filter Response Normalization (FRN)** –

```math
\Huge
y = \frac{x}{\sqrt{\eta + \mathrm{mean}(x^2)}}\,\gamma + \beta
```
– normalizes each feature map by its second-moment mean, without explicit subtraction of mean activations. It removes the need for computationally expensive running averages and can improve performance in classification and detection tasks.

---

## **Online Normalization** –

```math
\Huge
\hat{x}_t = \frac{x_t - \widehat{\mu}_t}{\sqrt{\widehat{\sigma}_t^2 + \epsilon}}
```
– incrementally updates mean/variance for each incoming sample. It supports data streaming scenarios with large or unbounded datasets and is useful when full-batch statistics are unavailable or memory-consuming to compute at scale.

---

## **Cross-Channel Normalization (CCN)** –

```math
\Huge
\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w}}{\sqrt{k + \alpha \sum_{c' \in \mathcal{C}} \left(x_{n,c',h,w}\right)^2}}
```
– normalizes a neuron’s response using neighboring channel intensities. Typically found in early CNN architectures (inspired by biological lateral inhibition), it can help reduce redundancy and encourage competition among feature channels.

---

## **Local Contrast Normalization (LCN)** –

```math
\Huge
\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,c,h,w}}{\max\!\Big(\sigma_{n,c,h,w},\,\sigma_0\Big)}
```
– subtracts local mean and divides by local standard deviation within a spatial neighborhood. It enhances local feature discrimination by reducing low-frequency biases, commonly used in early vision-based models.

---

## **RMS Normalization (RMSNorm)** –

```math
\Huge
\mathrm{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
```
– uses the root mean square of activations for normalization, without mean subtraction. This approach can stabilize training in deep sequence models and sometimes outperforms layer normalization for certain Transformer variants.

---

## **Switchable Normalization** –

```math
\Huge
\hat{x} = \alpha_{\mathrm{BN}}\cdot \hat{x}_{\mathrm{BN}} + \alpha_{\mathrm{IN}}\cdot \hat{x}_{\mathrm{IN}} + \alpha_{\mathrm{LN}}\cdot \hat{x}_{\mathrm{LN}},\quad \alpha_{\mathrm{BN}} + \alpha_{\mathrm{IN}} + \alpha_{\mathrm{LN}} = 1
```
– learns to combine Batch, Instance, and Layer Normalization adaptively. This hybrid approach can dynamically select the most suitable normalization statistics, improving model robustness across varied tasks and batch sizes.

---

## **Batch-Instance Normalization (BIN)** –

```math
\Huge
\hat{x} = \lambda \cdot \hat{x}_{\mathrm{BN}} + (1 - \lambda) \cdot \hat{x}_{\mathrm{IN}}
```
– interpolates between batch and instance statistics. Useful in style transfer tasks and content-sensitive applications, offering a balance between batch-level consistency and per-instance adaptation for more flexible feature transformations.

---

## **EvoNorm (EvoNorm-S0 example)** –

```math
\Huge
y = \frac{x}{\max\!\Big(\nu + \mathrm{Combine}(x^2),\,\mathrm{GroupMax}(x^2)\Big)^\beta} + \gamma x + \beta
```
– co-evolves normalization and activation functions. It removes explicit batch dependencies by leveraging feature statistics in a self-contained way, often showing competitive performance without conventional batch-based constraints.

---

## **AdaBatchNorm** –

```math
\Huge
\hat{x} = \gamma \left(\frac{x - \mu_{adapt}}{\sqrt{\sigma_{adapt}^2 + \epsilon}}\right) + \beta
```
– adapts running mean and variance with domain-specific parameters or different tasks. Particularly suited for domain adaptation or multi-task learning, where different data distributions share the same model but require customized normalization statistics.

---

## **Weight Standardization** –

```math
\Huge
\mathbf{W}_{std} = \frac{\mathbf{W} - \mu_w}{\sqrt{\sigma_w^2 + \epsilon}}
```
– normalizes weights channel-wise before convolution/linear operations. Unlike weight normalization, it subtracts the mean and divides by standard deviation for each filter, leading to smoother loss landscapes and often improving performance in CNNs.

---

## **Divisive Normalization** –

```math
\Huge
\hat{x}_{i} = \frac{x_i}{\alpha + \sum_{j} \beta_{ij} x_j^2}
```
– a biologically inspired model that divides activations by a combination of themselves and their neighbors. This reduces global activity levels and has been used to improve contrast and feature competition in early visual layers.

---

## **Power Normalization** –

```math
\Huge
\hat{x} = \left(\frac{x}{\|x\|_p + \epsilon}\right)^{\!\!\alpha}\,\gamma + \beta
```
– rescales features by their \(\ell_p\) norm, raised to a learnable or fixed exponent \(\alpha\). It can emphasize or de-emphasize sparse components and is useful in tasks like fine-grained recognition or robust embedding learning.

---

## **Cross-GPU Batch Normalization (SyncBN)** –

```math
\Huge
\hat{x} = \frac{x - \mu_{\mathrm{sync}}}{\sqrt{\sigma_{\mathrm{sync}}^2 + \epsilon}}
```
– synchronizes mean and variance across multiple GPUs to maintain consistent batch statistics in data-parallel training. It alleviates discrepancies in normalization when per-device batch sizes are small, often boosting multi-GPU performance.

=====================================================================
### 📣 Neural Network Components
=====================================================================

## **Fully-Connected Layer** –

# $` h_i^{(l)} = \sum_{j} W_{ij}^{(l)} a_j^{(l-1)} + b_i^{(l)} `$

–`$W^{(l)},b^{(l)}$: weights, bias;`$a^{(l-1)}$: inputs – computes weighted sum of previous layer outputs – basic neuron operation in dense networks. Fully-Connected Layers can account for the majority of parameters in large networks, often contributing significantly to memory usage and overfitting risk.

----------------------

## **Convolution (2D)** –

# $` (f * g)(i,j) = \sum_{u=0}^{H-1}\sum_{v=0}^{W-1} g(u,v)\,f(i+u,\;j+v) `$

–`$f`$: input image/feature,`$g`$: kernel of size`$H \times W`$

 – sliding dot product filter – extracts local spatial features (shift invariance). They are a cornerstone of modern CNNs, allowing spatial feature extraction that underpins tasks like image classification, detection, and segmentation.

----------------------

## **Depthwise Convolution** –

# $` y_{c}(i,j) = \sum_{u,v} W_{c}(u,v)\,x_{c}(i+u,j+v) `$

 – filter per input channel – convolves each channel independently – used in separable convolution to reduce computation (MobileNets). In some architectures, it drastically cuts computation and memory cost while preserving accuracy, enabling deep models on resource-constrained devices worldwide.

----------------------

## **Pointwise Convolution** –

# $` y_{c'}(i,j) = \sum_{c=1}^{C} W_{c',c}\,x_{c}(i,j) `$

–`$1\times1$` convolution mixing channels – learns cross-channel interactions without spatial context – often follows depthwise conv. This operation is key to channel fusion, often used after depthwise convolutions to form efficient depthwise-separable pipelines in mobile architectures.

----------------------

## **Dilated Convolution** –

# $` y(i) = \sum_{u} x(i + r\cdot u)\,w(u) `$

 – dilation rate`$r`$

 – inserts`$r-1$` zeros between filter taps – expands receptive field without increasing filter size or computations (used in WaveNet, Atrous conv in segmentation). They expand receptive fields exponentially without extra computations, beneficial in tasks requiring contextual understanding like semantic segmentation and audio generation.

----------------------

## **Residual Connection** –

# $` y = F(x) + x`$

–`$F(x)$: output of few layers – skip connection adding input`$x$` to output – mitigates vanishing gradients and enables very deep networks (He et al.’15 ResNet). They enable gradient flow through deep structures and are often crucial for training state-of-the-art architectures such as ResNets and Transformers.

----------------------

## **Global Avg Pooling** –

# $` z_c = \frac{1}{HW}\sum_{i=1}^H\sum_{j=1}^W x_{c}(i,j) `$

 – average feature map`$c$` over spatial dims – produces channel descriptor – used in classification heads (reduces parameters vs fully-connected). Besides reducing parameters, it can alleviate overfitting by enforcing spatial invariance, making features robust for final classification layers in a network.

----------------------

## **Squeeze-and-Excitation** –

# $` s_c = \frac{1}{HW}\sum_{i,j} x_c(i,j),\;\; z = \sigma(W_2\,\delta(W_1 s)),\;\; y_c = z_c \, x_c`$

 – global pooling “squeezes” to channel vector`$s`$, then two FC ($\delta`$: ReLU,`$\sigma`$: sigmoid) to “excite” channels by gating`$x`$

 – adaptively recalibrates feature maps (improves CNN accuracy). This channel attention mechanism can significantly boost accuracy with minimal extra cost, influencing designs in EfficientNet and broader network families.

----------------------

## **Transposed Convolution** – (deconvolution) upsamples feature map by learned kernels – achieved by inserting zeros between inputs or equivalently sliding a kernel with fractional stride – used for learnable upsampling in generators and decoders. When not configured, it can produce checkerboard artifacts in generated images, underscoring the need for appropriate stride and padding choices.

---

## **Instance Normalization** – \( \hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}} \) – \(\mu_{n,c}, \sigma_{n,c}\): mean,std for each instance-channel – common in style transfer. By normalizing channel-wise per-image, it preserves stylistic variations and enables flexible manipulation of artistic features.

---

## **ELU (Exponential Linear Unit)** – \( \text{ELU}(x) = \begin{cases}
x & x \ge 0 \\
\alpha\bigl(e^x - 1\bigr) & x < 0
\end{cases} \) – shifts outputs closer to zero mean – can speed learning. Its negative exponential region helps push mean activations toward zero, potentially smoothing optimization paths compared to ReLU-based activations.

---

## **GELU (Gaussian Error Linear Unit)** – \( \text{GELU}(x) = x \cdot \Phi(x) \) – \(\Phi(x)\): CDF of normal distribution – smoothly gates inputs. Widely used in Transformers, it often yields empirically better performance than ReLU variants by preserving fine-grained information in intermediate layers.

---

## **Transformer Feed-Forward** – \( \text{FFN}(x) = \max(0,\,xW_1 + b_1)\,W_2 + b_2 \) – MLP applied to each token independently – transforms per-position features. This dense projection and nonlinearity help blend extracted attention signals before passing them to subsequent layers.

---

## **Gated Recurrent Unit (GRU)** –

```math
\Huge
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1}),\\
r_t &= \sigma(W_r x_t + U_r h_{t-1}),\\
\tilde{h}_t &= \tanh(W_h x_t + U_h(r_t \odot h_{t-1})),\\
h_t &= (1 - z_t)\,h_{t-1} + z_t\,\tilde{h}_t
\end{aligned}
```
– fewer parameters than LSTM – efficient. It often achieves comparable performance to LSTMs with simpler gating, making it a popular choice for sequence modeling under memory constraints.

---

## **Long Short-Term Memory (LSTM)** –

```math
\Huge
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f),\\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i),\\
\tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C),\\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t,\\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o),\\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
```
– special cell state mitigates vanishing gradients. Its gating structure revolutionized sequence learning by preserving long-range context, becoming a cornerstone of many NLP and speech processing tasks.

---

## **Softmax** – \( \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \) – normalizes logits into probability distribution – typically final layer in classification. By mapping arbitrary real values to [0,1] with sum=1, it enables training networks for multi-class outputs.


---

## **Stochastic Depth** – skips entire residual blocks randomly during training – reduces expected depth – used in ResNets. It improves regularization and training speed by creating an ensemble-like effect, yielding deeper networks that generalize better.

---

## **AlphaDropout** – preserves mean and variance of activations for SELU networks – randomly masks neurons but keeps self-normalizing property. Specially designed for SELU-based architectures, it maintains consistent activation statistics, aiding stable training and convergence in self-normalizing nets.

---

## **Pixel Shuffle** – rearranges elements of a \((C \times r^2, H, W)\) tensor into \((C, H \cdot r, W \cdot r)\) – upsampling without learnable parameters – used in super-resolution. It avoids transposed convolution artifacts and enables higher-quality spatial scaling.

---

## **Channel Shuffle** – rearranges channels across different groups to encourage inter-group information flow – used in ShuffleNet. It strengthens cross-group feature mixing, compensating for group convolution’s channel partitioning and improving efficiency in mobile or real-time applications.

---

## **AdaIN (Adaptive Instance Normalization)** –

```math
\Huge
y_{n,c} = \sigma_s \frac{x_{n,c} - \mu(x_{n,c})}{\sigma(x_{n,c})} + \mu_s
```
– style parameters \((\mu_s,\sigma_s)\) – used in style transfer. It allows learning content from one domain and style from another by adaptively re-scaling instance-normalized features.

---

## **Conditional Batch Normalization** – applies affine transforms conditioned on extra inputs (e.g., class embedding) – typically in conditional GANs – modulates feature statistics by class. It lets the network learn class-dependent normalization, crucial for generating diverse images across categories.

---

## **Gradient Reversal Layer** – multiplies incoming gradient by \(-\lambda\) during backprop – used in adversarial training for domain adaptation. It forces shared features to be domain-invariant by reversing the gradient signal for domain classification, promoting robust generalization to new domains.

---

## **Attention Gate** – \( \alpha = \sigma(\text{Conv}( [x, g] )) ,\; y = \alpha \cdot x \) – learns where to focus in encoder-decoder models – used in medical imaging segmentation. By gating spatial features with a learned attention map, it highlights regions of interest and suppresses irrelevant areas.


=====================================================================
### 📣 Recurrent Networks
=====================================================================

## **Recurrent Neuron (RNN)** –

# $` h_t = \phi(W_{xh} x_t + W_{hh} h_{t-1} + b_h) `$

–`$x_t`$: input,`$h_{t-1}$: prev hidden,`$\phi`$: activation (tanh) – retains state across time – processes sequences by recurrent self-connection. They can struggle with long sequences due to vanishing gradients, requiring specialized architectures or gating to effectively capture extended dependencies.

----------------------

## **LSTM Cell** – *Gating:*`$f_t = \sigma(W_f [h_{t-1},x_t] + b_f),\;\; i_t = \sigma(W_i[h_{t-1},x_t] + b_i),\;\; o_t = \sigma(W_o[h_{t-1},x_t] + b_o)$; *State:*`$\tilde{c}_t = \tanh(W_c[h_{t-1},x_t] + b_c),\;\; c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,\;\; h_t = o_t \odot \tanh(c_t)$

 – uses forget ($f_t`$), input ($i_t`$), output ($o_t`$) gates to control cell memory`$c_t`$

 – preserves long-term dependencies, mitigates vanishing gradients. Its gating structure has become a standard for tasks like language modeling, speech recognition, and time-series forecasting, improving memory retention.

----------------------

## **GRU Cell** –

# $` z_t = \sigma(W_z[h_{t-1},x_t]),\;\; r_t = \sigma(W_r[h_{t-1},x_t]),\;\; \tilde{h}_t = \tanh(W_h[r_t \odot h_{t-1},\;x_t]),\;\; h_t = (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t`$

 – update gate`$z_t$` controls retention of past, reset gate`$r_t$` controls influence of prev`$h_{t-1}$

 – simpler than LSTM, still handles long dependencies. They often perform comparably to LSTMs with fewer parameters, making them popular in resource-limited or production-oriented sequence modeling applications for practical deployments.

-------------------

## **Bidirectional RNN (BiRNN)**

```math
\Huge
\overrightarrow{h}_t = \phi\big(W_{\overrightarrow{h}x}x_t + W_{\overrightarrow{h}\overrightarrow{h}}\overrightarrow{h}_{t-1} + b_{\overrightarrow{h}}\big),
\quad
\overleftarrow{h}_t = \phi\big(W_{\overleftarrow{h}x}x_t + W_{\overleftarrow{h}\overleftarrow{h}}\overleftarrow{h}_{t+1} + b_{\overleftarrow{h}}\big)
```
```math
\Huge
y_t = W_{y\overrightarrow{h}}\overrightarrow{h}_t + W_{y\overleftarrow{h}}\overleftarrow{h}_t + b_y
```
They read sequences forward and backward, capturing past and future context. This yields richer representations for tasks like speech or text processing, providing improved accuracy when future tokens or events help interpret earlier parts.

---

## **Peephole LSTM**

```math
\Huge
f_t = \sigma\big(W_{f}[h_{t-1}, x_t, c_{t-1}] + b_f\big),
\quad
i_t = \sigma\big(W_{i}[h_{t-1}, x_t, c_{t-1}] + b_i\big),
\quad
o_t = \sigma\big(W_{o}[h_{t-1}, x_t, c_{t}] + b_o\big)
```
```math
\Huge
\tilde{c}_t = \tanh\big(W_{c}[h_{t-1}, x_t] + b_c\big),
\quad
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t,
\quad
h_t = o_t \odot \tanh(c_t)
```
They incorporate the internal cell state directly into gating signals. This extra “peek” at \(c_t\) can enhance timing-sensitive tasks, especially in speech recognition or any domain where exact state transitions matter.

---

## **Stacked RNN**

```math
\Huge
h_t^{(l)} = \phi\big(W_{x}^{(l)} h_t^{(l-1)} + W_{h}^{(l)} h_{t-1}^{(l)} + b^{(l)}\big)
\quad \text{for layer } l=1,\dots,L
```
Multiple RNN layers are stacked so each layer’s hidden state feeds into the next. Deeper structures capture more complex patterns in sequences, though they require careful initialization and regularization to avoid vanishing or exploding gradients.

---

## **Residual RNN**

```math
\Huge
h_t^{(l)} = h_t^{(l-1)} + \alpha \,\phi\big(W^{(l)} h_t^{(l-1)} + b^{(l)}\big)
```
They add a residual shortcut across layers, making gradients flow more easily through depth. This reduces training difficulties for deeper recurrent models, allowing them to learn richer sequential dependencies without severe performance degradation.

---

## **Minimal Gated Unit (MGU)**

```math
\Huge
f_t = \sigma\big(W_f [h_{t-1}, x_t] + b_f\big),
\quad
\tilde{h}_t = \tanh\big(W_h [f_t \odot h_{t-1}, x_t] + b_h\big),
\quad
h_t = (1 - f_t)\odot h_{t-1} + f_t \odot \tilde{h}_t
```
It simplifies LSTM-like gating to a single update gate. This minimal design can perform on par with larger gated architectures while using fewer parameters, benefiting applications constrained by memory or computational resources.

---

## **Gated Feedback RNN (GF-RNN)**

```math
\Huge
h_t = \phi\big(W_{xh} x_t + F(h_{t-1}) \odot W_{hh} h_{t-1} + b_h\big)
```
Here, a learned gating function \(F(\cdot)\) modulates recurrent feedback. It can dynamically regulate how past hidden states influence future processing, aiding tasks that require selective retention or suppression of historical information.

---

## **UGRNN (Update Gate RNN)**

```math
\Huge
u_t = \sigma\big(W_{u}[h_{t-1}, x_t]\big),
\quad
\tilde{h}_t = \phi\big(W_{h}[h_{t-1}, x_t]\big),
\quad
h_t = (1-u_t)\odot h_{t-1} + u_t \odot \tilde{h}_t
```
This design merges reset and update concepts into a single gate. It can achieve performance similar to GRU or LSTM with a simpler gating mechanism, often improving training stability on various sequence modeling tasks.

---

## **IndRNN (Independent RNN)**

```math
\Huge
h_t = \phi\big(W \odot h_{t-1} + Ux_t\big)
```
Each neuron’s recurrent weight is restricted to a single scalar, preventing exponential explosion of cross-neuron interactions. It allows for very deep recurrent stacking, making gradient flow more manageable for extended sequence tasks.

---

## **JANET (Just Another NET)**

```math
\Huge
f_t = \sigma\big(W_f [h_{t-1}, x_t]\big),
\quad
\tilde{h}_t = \tanh\big(W_h [f_t \odot h_{t-1}, x_t]\big),
\quad
h_t = h_{t-1} + f_t \odot (\tilde{h}_t - h_{t-1})
```
Its gating design simplifies the LSTM framework with a skip-like connection. This structure can train quickly and handle longer contexts without heavy computation, making it appealing for language and time-series modeling tasks.

---

## **Hierarchical RNN**

```math
\Huge
h_{t}^{(\ell)} = \phi\big(W^{(\ell)} h_{t-1}^{(\ell)} + U^{(\ell)} h_{t}^{(\ell-1)}\big),
\quad \ell=1,\dots,L
```
They process input sequences in multiple temporal resolutions or granularities. Lower layers handle finer timescales, and higher layers capture longer-range dependencies, enabling efficient learning of complex patterns over long sequences and sub-sequence structures.

---

## **SRU (Simple Recurrent Unit)**

```math
\Huge
c_t = f_t \odot c_{t-1} + (1 - f_t) \odot x_t,
\quad
h_t = g_t \odot \tanh(c_t) + (1 - g_t) \odot x_t
```
Separates most computations into parallelizable element-wise operations, significantly speeding up training. Despite its simpler structure, it performs competitively with conventional RNNs, especially on GPUs, and is favored in large-scale language tasks.

---

## **Dilated RNN**

```math
\Huge
h_t = \phi\big(W_{xh} x_t + W_{hh} h_{t-D} + b_h\big)
```
They skip certain timesteps, connecting \(h_t\) with \(h_{t-D}\) instead of \(h_{t-1}\). This “dilation” helps capture longer context without large hidden states, improving efficiency and alleviating vanishing gradient issues in extended sequences.

---

## **Phased LSTM**

```math
\Huge
\alpha_t = \text{pulse}(t; \tau, r),
\quad
\tilde{c}_t = \tanh\big(W_c[h_{t-1}, x_t]\big),
\quad
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \odot \alpha_t
```
Incorporates a time gate that “opens” and “closes” based on an oscillation schedule. This allows asynchronous event processing in time-series data, beneficial for irregularly sampled signals like medical or sensor-based measurements.

---

## **Clockwork RNN**

```math
\Huge
h_{t}^{(k)} =
\begin{cases}
\phi\big(W_{x}^{(k)} x_t + W_{h}^{(k,k)} h_{t-1}^{(k)} + \sum_{j<k} W_{h}^{(k,j)} h_{t}^{(j)}\big) &\text{if } t \mod \tau_k = 0,\\
h_{t-1}^{(k)} &\text{otherwise}
\end{cases}
```
Partitions hidden units into modules running at different clock rates. Faster modules update frequently, slower ones more sparsely. This improves computational efficiency while capturing multi-timescale information in speech or sensor data.

---

## **Echo State Network (ESN)**

```math
\Huge
h_t = \phi\big(W_{in} x_t + W h_{t-1}\big),
\quad
y_t = W_{out} [h_t; x_t]
```
Features a large, fixed randomly connected “reservoir,” with only output weights trained. The reservoir’s rich dynamics handle diverse temporal patterns, making ESNs fast to train and interpretable for tasks like real-time control.

---

## **Liquid State Machine (LSM)**

```math
\Huge
h_t = \text{spiking dynamics in reservoir of neurons},
\quad
y_t = W_{out}\,h_t
```
A biologically inspired spiking network forms a liquid reservoir with time-varying responses. Similar to ESNs, output weights are learned on these evolving states, useful for processing continuous-time signals with event-driven spikes.

---

## **Recurrent Highway Network (RHN)**

```math
\Huge
h_t = h_{t-1} + T_t \odot \big(\phi(W_{H}h_{t-1}) - h_{t-1}\big),
\quad
T_t = \sigma\big(W_{T}h_{t-1}\big)
```
Uses highway “transform” gates to regulate hidden-state updates. This approach stabilizes training in very deep recurrent stacks, enabling models to handle complex, long-range sequence structures more effectively than standard deep RNNs.

---

## **HyperLSTM**

```math
\Huge
\theta_t = LSTM_{\text{hyper}}\big(x_t, \theta_{t-1}\big),
\quad
h_t = LSTM_{\text{main}}\big(x_t, h_{t-1}, \theta_t\big)
```
A secondary LSTM generates context-dependent parameters (\(\theta_t\)) for the main LSTM, allowing dynamic weight adaptation. This improves modeling flexibility and can capture rapidly changing temporal dependencies in language or sequential data.

---

## **Quasi-RNN (QRNN)**

```math
\Huge
f_t, z_t = \sigma(W_f x_t),
\quad
o_t = \tanh(W_o x_t),
\quad
c_t = f_t \odot c_{t-1} + (1 - f_t)\odot z_t,
\quad
h_t = o_t \odot c_t
```
Combines convolutional front-ends with minimal recurrent gating, yielding parallelizable computations. QRNNs can train faster than classic LSTMs or GRUs while preserving long-term context, proving effective for large-scale language and sequence tasks.

---

## **Mogrifier LSTM**

```math
\Huge
x_t^{(k+1)} = x_t^{(k)} \odot \sigma\big(U^{(k)} h_{t-1}^{(k)}\big),
\quad
h_{t-1}^{(k+1)} = h_{t-1}^{(k)} \odot \sigma\big(V^{(k)} x_t^{(k+1)}\big)
```
It iteratively “mogrifies” input and hidden vectors before standard LSTM gating. This gating interplay boosts expressiveness significantly. It often yields better perplexities in language modeling tasks, with relatively modest increases in computational overhead.


=====================================================================
### 📣 Attention Mechanisms
=====================================================================


## **Scaled Dot-Product Attn** –

# $` \text{Attn}(Q,K,V) = \text{softmax}\!\Big(\frac{QK^T}{\sqrt{d}}\Big)\,V`$

–`$Q,K,V`$: query, key, value matrices;`$d`$: key dim – computes weighted sum of values by similarity of queries and keys – core of transformer self-attention. The sqrt(d) scaling helps stabilize training by reducing variance in dot products, ensuring consistent gradients across tokens.

----------------------

## **Multi-Head Attention** –

# $` \text{MHA}(Q,K,V) = [\text{head}_1,\dots,\text{head}_h] W^O,\;\; \text{head}_i = \text{Attn}(Q W_i^Q,\;K W_i^K,\;V W_i^V) `$

 – multiple attention “heads” project input into subspaces with learned`$W_i`$

 – captures information from different representation subspaces and positions in parallel. The heads can highlight distinct relationships, though empirical studies show some redundancy, indicating potential parameter reduction opportunities.

----------------------

## **Additive Attention** –

# $` e_{ij} = v^T \tanh(W_q q_i + W_k k_j) `$, $` \alpha_{ij} = \frac{e^{e_{ij}}}{\sum_{j'} e^{e_{ij'}}} `$,`$c_i = \sum_j \alpha_{ij} v_j`$

–`$q_i`$: query (decoder state),`$k_j,v_j`$: key, value (encoder) – uses a small neural network to compute attention scores`$e_{ij}$

 – original attention mechanism in seq2seq models (Bahdanau). It handles variable-length inputs gracefully, often leading to improved alignment in tasks like translation compared to purely dot-product-based methods.

----------------------

## **Self- vs Cross-Attn** – *Self:*`$Q=K=V=X$` (same sequence attends to itself); *Cross:*`$Q = X_{\text{target}},\;K=V=X_{\text{source}}$

 – self-attention relates elements within a sequence (e.g. words in sentence), cross-attention integrates information from another sequence (e.g. decoder attending encoder outputs in translation). Combining self- and cross-attention in multi-stage processing allows models to refine internal representations before integrating external context from other sequences.

----------------------

## **FlashAttention-2** – *Computes*`$A = \text{softmax}(QK^T/\sqrt{d})V$` *in tiled blocks to reduce memory* – reorders operations to keep only small submatrices in memory at once – achieves linear memory usage in sequence length – enables training with much longer sequences (faster GPU attention by better parallelism). Its block-based memory footprint enables training with sequence lengths previously too large for GPUs, unlocking new breakthroughs in language modeling.

----------------------

## **Linear Attention (Performer)** –

# $` \text{Attn}(Q,K,V) \approx \phi(Q)[\phi(K)^T V]`$

–`$\phi`$: feature map (e.g. random Fourier mapping) – expresses softmax attention as kernel dot-products – reduces complexity to`$O(n)$

 – approximation for efficient long-sequence attention. Using random feature approximations to the softmax kernel, it can handle extremely long sequences without excessive computation or memory overhead.

----------------------

## **Local (Windowed) Attn** –

# $` \alpha_{ij} = 0 $` if`$|i-j| > w $` (only attend within window`$w`$) – restricts attention to fixed local context – linear complexity in sequence length – used in Longformer/BigBird for long documents. Despite reduced global context, strategically overlapping windows can preserve essential dependencies in tasks like question answering over lengthy documents.



## **Positional Encoding**

# $PE_{(pos,2i)} = \sin\!\big(\frac{pos}{10000^{2i/d}}\big),\;\; PE_{(pos,2i+1)} = \cos\!\big(\frac{pos}{10000^{2i/d}}\big)$

–`$pos`$: token index,`$i`$: dimension index – fixed sinusoidal embeddings added to input – injects sequence order information into transformer (since self-attention alone is order-agnostic). Alternative learned positional embeddings can outperform sinusoidal encodings, but require more parameters and can be less interpretable.

----------------------

## **Feed-Forward Layer** –

# $` \text{FFN}(x) = W_2\,\sigma(W_1 x + b_1) + b_2`$

 – two linear transformations with ReLU ($\sigma`$) – position-wise MLP applied to each sequence element – expands model capacity after attention layer (Transformers typically use high FFN dimension). Scaling up the hidden dimension here significantly boosts model capacity, often at relatively low computational cost compared to deeper layers.

----------------------

## **Causal Mask** –

# $` a_{ij} = \begin{cases} \frac{q_i \cdot k_j}{\sqrt{d}} & j \le i \\ -\infty & j>i \end{cases} `$` (before softmax) – masks out future positions ($j>i`$) in self-attention – ensures decoder can’t attend to “future” tokens when generating sequentially (autoregressive property). This strictly enforces left-to-right generation, vital for tasks like language modeling where partial future leakage can degrade performance.

-------------------

## **Luong Dot-Product Attention** –

```math
\Huge
e_{ij} = q_i^T \, W_a \, k_j,\quad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'} \exp(e_{ij'})},\quad
c_i = \sum_j \alpha_{ij}\,v_j
```
– introduced by Luong et al. as a more direct dot-product form than Bahdanau’s additive approach – often faster to compute with fewer parameters.

An interesting fact: Despite its simplicity, Luong’s formulation can match additive attention’s performance, highlighting that architecture nuances (like context gating) can be equally crucial as the attention scoring function itself.

----------------------

## **Hard Monotonic Attention** –

```math
\Huge
\alpha_{ij} = \begin{cases}
\prod_{k=1}^{j-1}(1 - p_{ik}) \cdot p_{ij} & j \le \text{length}(x) \\
0 & \text{otherwise}
\end{cases}
```
– enforces a strict left-to-right alignment by halting once it attends a position – suitable for speech recognition or translation tasks requiring monotonic alignment.

An interesting fact: Hard monotonic attention can reduce computational overhead during inference, since it “locks in” alignment decisions early, mirroring how humans read or listen sequentially without revisiting past context too often.

----------------------

## **Gated Attention** –

```math
\Huge
\text{GA}(Q,K,V) = \sigma\big(U\,[Q\,||\,K]\big) \odot V
```
– concatenates query and key, passes through a learned gate`$\sigma(\cdot)$, then scales values`$V`$

 – emphasizes relevant features while suppressing distractions.

An interesting fact: Gated attention merges the gating concept from LSTM/GRU with attention, offering finer control over what is passed forward, sometimes stabilizing training in challenging tasks like noisy text or multi-modal data integration.

----------------------

## **Hierarchical Attention** –

```math
\Huge
\alpha_{ij}^{(\text{word})} = \frac{\exp\big(u_{ij}\big)}{\sum_{k}\exp\big(u_{ik}\big)},\quad
\alpha_{i}^{(\text{sentence})} = \frac{\exp\big(u_{i}\big)}{\sum_{m}\exp\big(u_{m}\big)}
```
– applies attention at word-level (within a sentence) then at sentence-level (within a document) – helps process long documents by structuring attention.

An interesting fact: Hierarchical attention substantially reduces complexity by focusing on local groupings first, mirroring how humans parse text in paragraphs before extracting key ideas from entire articles.

----------------------

## **Co-Attention (Bilinear)** –

```math
\Huge
S_{ij} = Q_i^T\,W_c\,K_j,\quad
\alpha_i = \text{softmax}(S_{ij}),\quad
\beta_j = \text{softmax}(S_{ij}^T)
```
– jointly learns attention over two sequences (e.g., question and passage) via a shared bilinear term`$W_c`$

 – used in QA and VQA.

An interesting fact: By attending each sequence to the other, co-attention captures cross-dependencies, often boosting reasoning performance in tasks where understanding the interplay between dual inputs is crucial for correct inference.

----------------------

## **Sparse/Top-$k$` Attention** –

```math
\Huge
\alpha_{ij} =
\begin{cases}
\frac{\exp(e_{ij})}{\sum_{k \in \Omega_i}\exp(e_{ik})}, & j \in \Omega_i \\
0, & \text{otherwise}
\end{cases}
```
–`$\Omega_i$` contains only top-$k$` scoring keys for query`$i`$

 – reduces computational cost by pruning low-scoring alignments.

An interesting fact: Sparse attention not only speeds up training for lengthy sequences but can also mimic how humans focus on the most important parts of context, potentially aiding interpretability of attention weights.

----------------------

## **LSH Attention (Reformer)** –

```math
\Huge
\text{Attn}_{\text{LSH}}(Q,K,V)=\text{concat}\Big(\sum_{j\in \mathcal{B}_i}\alpha_{ij}v_j\Big),
```
– uses locality-sensitive hashing to group similar queries/keys into buckets`$\mathcal{B}_i`$

 – computes attention only within each bucket – achieves sub-quadratic complexity.

An interesting fact: LSH attention dramatically reduces memory usage for very long sequences, inspiring research into hashing-based approximations that make transformers feasible on inputs like entire books or lengthy DNA sequences.

----------------------

## **Sinkhorn Attention** –

```math
\Huge
\Pi = \text{Sinkhorn}\big(\exp(QK^T)\big),\quad
\text{Attn}(Q,K,V)=\Pi\,V
```
– replaces softmax with a doubly-stochastic matrix`$\Pi$` computed by Sinkhorn’s algorithm – encourages balanced attention distribution over queries and keys.

An interesting fact: By imposing row/column constraints, Sinkhorn attention can learn balanced assignments, resembling a bipartite matching process that may enhance structure in tasks like sorting or more fine-grained alignment scenarios.

----------------------

## **Routing Attention (Routing Transformer)** –

```math
\Huge
\text{cluster}(Q,K)\to \{\mathcal{C}_1,\dots,\mathcal{C}_r\};\quad
\text{Attn}(Q,K,V)=\sum_{c=1}^r \text{Attn}\big(Q_{\mathcal{C}_c},K_{\mathcal{C}_c},V_{\mathcal{C}_c}\big)
```
– clusters queries and keys, performing attention within each cluster – lowers complexity compared to full attention.

An interesting fact: Routing transforms the attention map into a structured clustering problem, enabling large-scale language models to handle extremely long inputs while preserving crucial local relationships inside each cluster.

----------------------

## **Axial Attention** –

```math
\Huge
\text{AxialAttn}(X) = \text{Attn}\big(X_{\text{rows}}\big)+\text{Attn}\big(X_{\text{cols}}\big)
```
– processes rows and columns separately in grid-like data (e.g. images) – drastically reduces 2D attention cost from`$O(n^2 \times m^2)$` to`$O(n \times m^2 + n^2 \times m)$.

An interesting fact: Axial attention was pivotal for image-based transformers, splitting attention across each dimension, offering a more efficient approach to capturing spatial relations in high-resolution images or tabular data.

----------------------

## **Nystrom Attention (Nystromformer)** –

```math
\Huge
\tilde{A} = P \left(D^{-1} \tilde{K} \tilde{Q}^T D^{-1}\right) U
```
– approximates the softmax kernel via Nystrom method, reducing complexity to`$O(n)$

 – uses a small set of landmark points to reconstruct attention maps.

An interesting fact: Nystrom attention empirically handles sequences of lengths exceeding 8K tokens with manageable memory, expanding the practical horizon for tasks like structured document understanding and multi-document QA.

----------------------

## **Double Attention** –

```math
\Huge
H = \text{softmax}\!\big(QK^T\big)K,\quad
\text{DoubleAttn}(Q,K,V)=\text{softmax}\!\big(QH^T\big)V
```
– factorizes attention into two sequential matrix multiplications – originally used in segmentation tasks – helps capture complex pairwise interactions more flexibly.

An interesting fact: Double attention can highlight multi-step relationships: first focusing on relevant keys, then refocusing queries using that intermediate context, which can improve interpretability on vision tasks like object part segmentation.

----------------------

## **Dynamic Convolution Attention** –

```math
\Huge
\text{DynConv}(x) = \sum_{k=0}^{K-1}\alpha_k \cdot \text{Conv1D}_k(x),\quad
\alpha = \text{softmax}(W\,x)
```
– learns per-timestep convolution filters weighted by attention-like coefficients – alternative to standard self-attention in certain NLP architectures.

An interesting fact: By blending learned convolution kernels dynamically, it fuses local pattern recognition with global weighting, sometimes matching transformers on translation tasks while maintaining improved locality and parameter efficiency.

----------------------

## **Deformable Attention (Deformable DETR)** –

```math
\Huge
\text{DeformAttn}(q) = \sum_{h=1}^{H} \omega_h \sum_{r=1}^{R}\text{softmax}_r(\delta_{r,h}) \cdot V\big(\phi(q)+\Delta p_{r,h}\big)
```
– samples key-value pairs from sparse reference points – designed for object detection in images.

An interesting fact: Deformable attention significantly accelerates DETR-like models by focusing only on relevant image regions, showing that specialized attention sampling can mitigate slow convergence issues in dense vision tasks.

----------------------

## **ProbSparse Attention (Informer)** –

```math
\Huge
\alpha_{ij} \approx
\begin{cases}
\frac{e^{e_{ij}}}{\sum_{k \in \Theta_i} e^{e_{ik}}}, & j \in \Theta_i \\
0, & \text{otherwise}
\end{cases}
```
– selects “dominant” queries for the softmax calculation, approximating the rest as negligible – tailored for long-sequence time-series forecasting.

An interesting fact: By identifying queries with the largest`$\|\mathbf{Q}_i\|$, ProbSparse cuts down attention operations, scaling to huge time-series. This approach underscores the importance of prioritizing “high-energy” tokens in real-world streams.

----------------------

## **Memory Compressed Attention** –

```math
\Huge
K' = M(K), \quad V' = M(V), \quad
\text{MCA}(Q,K,V)=\text{Attn}(Q, K', V')
```
– compresses keys and values into fewer “memory” slots before computing attention – used in models like ETC and Longformer variants.

An interesting fact: Memory compression trades exact token-level alignment for drastically lower computational cost, especially in tasks involving thousands of tokens, showing that partial summarization often suffices for capturing long-range dependencies.


=====================================================================
### 📣 Generative Models
=====================================================================


## **Autoregressive Factorization** –

# $` P(x_1,\ldots,x_n) = \prod_{t=1}^n P(x_t\,|\,x_{<t}) `$

 – chain rule decomposition of joint probability – basis of sequence models (e.g. language modeling: predict next token given all previous). This factorization can slow inference since each new token depends on all previously generated outputs.



## **Reparameterization Trick** –

# $` z = \mu(x) + \sigma(x) \odot \epsilon,\;\; \epsilon \sim \mathcal{N}(0,I) `$

–`$\mu,\sigma`$: encoder outputs – expresses random latent`$z$` as deterministic transform of input`$x$` and noise`$\epsilon`$

 – enables backpropagation through stochastic sampling in VAEs. It also reduces gradient variance, stabilizing the training of variational autoencoders.



## **Normalizing Flow** –

# $` \log p_X(x) = \log p_Z(f(x)) + \log\!\Big|\det \frac{\partial f(x)}{\partial x}\Big|`$

–`$f`$: invertible transform,`$p_Z`$: base density (e.g. Gaussian) – computes exact likelihood via change of variables – used to train flow-based generative models (e.g. NICE, RealNVP, Glow). These invertible transforms can be composed to model complex distributions with exact log-likelihoods.



## **Diffusion Forward** –

# $` q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}\,x_{t-1},\;\beta_t I) `$

–`$\beta_t`$: noise schedule – gradually adds Gaussian noise at each time step`$t`$

 – forward process that destroys data structure (for Denoising Diffusion models). The choice of noise schedule critically affects the fidelity of reconstructed samples in the reverse process.



## **Diffusion Denoising Loss** –

# $` L(\theta) = \mathbb{E}_{t,\epsilon}\big[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\big]`$

 – model`$\epsilon_\theta$` predicts added noise`$\epsilon$` at step`$t`$

 – trains diffusion model to reverse the forward noising by predicting noise and subtracting it. It also serves as a form of score matching, learning the gradient of the log-density at each diffusion step.



## **GAN (Minimax) Objective** –

# $` \min_G \max_D \;\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p(z)}[\log(1 - D(G(z)))]`$

 – adversarial game between generator`$G$` and discriminator`$D`$

–`$G$` tries to produce real-like samples`$G(z)$` to maximize`$D`$’s error,`$D$` tries to correctly classify real vs fake. Convergence can be delicate, often requiring techniques like gradient penalties or spectral normalization.



## **Energy-Based Model** –

# $` p(x) = \frac{1}{Z} \exp(-E(x)) `$

–`$E(x)$: energy function,`$Z`$: partition function – assigns unnormalized probability via energy – used in Boltzmann machines, Hopkins (learn generative model by shaping energy landscape). Estimating or approximating the partition function`$Z$` is a central challenge often tackled with MCMC.



## **Word2Vec Skip-Gram** –

# $` L = -\Big[\log \sigma(u_{w}\!\cdot v_{c^+}) + \sum_{c^-}\log \sigma(-\,u_{w}\!\cdot v_{c^-})\Big]`$

–`$u_w`$: target word vector,`$v_{c^+}$: context vector (positive),`$v_{c^-}$: negatives – trains word embeddings by making dot-product high for word with its context`$c^+$, low with random negatives`$c^-$(negative sampling). Subsampling frequent words further speeds up training while retaining embedding quality.



## **PixelCNN Autoregression** –

# $` p(X) = \prod_{i,j} p(x_{i,j}\,|\,x_{<i,j}, x_{i,<j}) `$

 – image pixels ordered raster-scan – predicts each pixel given those above and to the left – used in PixelCNN generative model for images (autoregressive pixel modeling). Despite high-quality samples, sequential pixel-by-pixel generation can be computationally expensive for large images.



## **Perplexity (LM)** –

# $` \mathrm{PP} = \exp\!\Big(-\frac{1}{N}\sum_{i=1}^N \log P(w_i|w_{<i})\Big) `$

 – exponential of average cross-entropy for a sequence – lower perplexity indicates a better language model (each word is less “surprising” on average). Improvements in perplexity often align with better text generation but may not ensure gains in all downstream tasks.

-------------------

## **ELBO (VAE) Objective** –

```math
\Huge
\mathcal{L}_{\text{ELBO}} \;=\; \mathbb{E}_{q_\phi(z \mid x)}\bigl[\log p_\theta(x \mid z)\bigr] \;-\; D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\,p_\theta(z)\bigr)
```
Enforces a balance between reconstruction fidelity and latent-space regularization. It underpins variational autoencoders, letting us optimize a lower bound on the data log-likelihood via gradient descent, crucial for stable end-to-end training.

---

## **Beta-VAE Objective** –

```math
\Huge
\mathcal{L}_{\beta\text{-VAE}} \;=\; \mathbb{E}_{q_\phi(z \mid x)}\bigl[\log p_\theta(x \mid z)\bigr] \;-\; \beta\,D_{\mathrm{KL}}\bigl(q_\phi(z \mid x)\,\|\,p_\theta(z)\bigr)
```
Promotes disentangled latent factors by weighting the KL term with \(\beta\). Higher \(\beta\) encourages factorized representations of data, aiding interpretability but risking reduced reconstruction quality if \(\beta\) is set too large.

---

## **WGAN Objective** –

```math
\Huge
\min_{G} \max_{D}\;\; \mathbb{E}_{x \sim p_{\text{data}}}\bigl[D(x)\bigr] \;-\; \mathbb{E}_{z \sim p(z)}\bigl[D\bigl(G(z)\bigr)\bigr]
```
Minimizes the Earth Mover (Wasserstein) distance between real and generated distributions, improving training stability over original GANs. It requires a Lipschitz discriminator, often enforced through weight clipping or gradient penalties.

---

## **WGAN-GP Regularizer** –

```math
\Huge
\mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\!\Bigl[\bigl(\|\nabla_{\hat{x}}D(\hat{x})\|\!-\!1\bigr)^2\Bigr]
```
Penalizes gradients to maintain the 1-Lipschitz constraint in WGANs. It stabilizes adversarial training by reducing mode collapse and improving sample diversity, leading to more consistent discriminator updates and robust generator improvements.

---

## **CycleGAN Objective** –

```math
\Huge
\min_{G_X, G_Y}\max_{D_X,D_Y}\;\; \mathcal{L}_{\text{GAN}}(G_X,D_X) \;+\; \mathcal{L}_{\text{GAN}}(G_Y,D_Y) \;+\; \lambda\,\mathcal{L}_{\text{cyc}}(G_X,G_Y)
```
Enables unpaired image-to-image translation by enforcing cycle-consistency losses. Each generator learns to map one domain to another and back, ensuring structural consistency without requiring paired training examples for supervision.

---

## **Gumbel-Softmax Trick** –

```math
\Huge
y \;=\; \mathrm{softmax}\!\Bigl(\frac{\log(\pi) + g}{\tau}\Bigr),
\quad g_i = -\log\bigl(-\log(u_i)\bigr),\;\; u_i \sim \mathrm{Uniform}(0,1)
```
Provides a differentiable approximation for discrete sampling, allowing backpropagation through categorical variables. By adjusting the temperature \(\tau\), one can smoothly interpolate between soft and hard categorical assignments in training.

---

## **Discrete VAE Loss** –

```math
\Huge
\mathcal{L} \;=\; \mathbb{E}_{q_{\phi}(z \mid x)}\bigl[-\log p_{\theta}(x \mid z)\bigr] \;+\; D_{\mathrm{KL}}\!\Bigl(q_{\phi}(z \mid x)\,\Big\|\,p(z)\Bigr)
```
Models data with discrete latent codes instead of continuous vectors. This helps capture categorical features, potentially improving interpretability and robustness when modeling multimodal distributions in image, text, or audio domains.

---

## **Score Matching** –

```math
\Huge
\mathcal{L}(\theta)
\;=\; \tfrac{1}{2} \,\int p_{\text{data}}(x)\,\sum_{i=1}^d \Bigl(\partial_{x_i} \log f_\theta(x)\Bigr)^{2}\,dx
```
Forces the model’s gradient of log-density to match that of the data distribution. It underlies many modern diffusion and noise-conditioned models by learning to predict the gradient of log-probabilities directly from noisy samples.

---

## **Contrastive Divergence (RBM)** –

```math
\Huge
\frac{\partial}{\partial w_{ij}} \,\mathcal{L}_{CD}
\;=\; \langle v_i\,h_j \rangle_{\text{data}}
\;-\; \langle v_i\,h_j \rangle_{\text{model}}
```
Updates weights in Restricted Boltzmann Machines by contrasting positive phase statistics with negative phase reconstructions. This short MCMC-based approximation is often sufficient for practical training without fully sampling equilibrium distributions.

---

## **Wake-Sleep Algorithm** –

```math
\Huge
\text{Wake: } \nabla_\theta \approx \mathbb{E}_{q_\phi(z\mid x)}\bigl[\nabla_\theta \log p_\theta(x,z)\bigr],
\quad
\text{Sleep: } \nabla_\phi \approx \mathbb{E}_{p_\theta(x,z)}\bigl[\nabla_\phi \log q_\phi(z\mid x)\bigr]
```
A two-phase learning scheme for Helmholtz Machines. “Wake” updates the generative parameters by maximizing data likelihood; “Sleep” updates the recognition network by fitting variational parameters to samples from the generative model.

---

## **Deep Belief Network (DBN)** –

```math
\Huge
p(v) \;=\; \sum_{h^{(1)},\,h^{(2)},\,\ldots} p\bigl(h^{(1)},h^{(2)},\ldots\bigr)\;p\bigl(v \mid h^{(1)}\bigr)
```
Stacks RBMs in a hierarchical fashion, initializing each layer in an unsupervised manner. DBNs can learn complex feature representations, later fine-tuned for classification or regression tasks through backpropagation.

---

## **Latent Dirichlet Allocation (LDA)** –

```math
\Huge
\theta_d \!\sim\! \mathrm{Dirichlet}(\alpha),\;\;
z_{d,n} \!\sim\! \mathrm{Categorical}(\theta_d),\;\;
\phi_k \!\sim\! \mathrm{Dirichlet}(\beta),\;\;
w_{d,n} \!\sim\! \mathrm{Categorical}(\phi_{z_{d,n}})
```
A generative model for topic discovery in corpora. Documents are mixtures of topics, and each topic is a distribution over words. This hierarchical Bayesian approach underpins many text-mining and thematic analysis methods.

---

## **GFlowNet Flow Matching** –

```math
\Huge
F(s \to s') \;=\; \frac{P(s')}{\sum_{x\in \mathrm{Ch}(s)} P(x)} \,\sum_{x \in \mathrm{Ch}(s)} F(s \to x)
```
Generates samples through a stochastic policy that factors as “flows” in a directed graph. GFlowNets aim to match a target distribution by learning flow conservation, enabling diverse sampling strategies beyond standard MCMC.

---

## **Gaussian Mixture Model (GMM)** –

```math
\Huge
p(x) \;=\; \sum_{k=1}^K \pi_k \,\mathcal{N}\!\bigl(x \mid \mu_k, \Sigma_k\bigr)
```
Represents data as a weighted sum of Gaussian components. By fitting these parameters via Expectation-Maximization, GMMs discover clusters or subpopulations, making them foundational in many density estimation and latent variable tasks.

---

## **Mixture Density Network (MDN)** –

```math
\Huge
p(y \mid x)
\;=\;
\sum_{k=1}^K \alpha_k(x)\,\mathcal{N}\!\bigl(y \mid \mu_k(x),\Sigma_k(x)\bigr)
```
Combines neural networks with mixture models to predict multi-modal output distributions. Each mixture component’s parameters are produced by a neural network, accommodating complex, non-Gaussian relationships between inputs and outputs.

---

## **Vector Quantized-VAE (VQ-VAE) Loss** –

```math
\Huge
\mathcal{L}
\;=\;
\|x-\hat{x}\|^2 \;+\; \|\mathrm{sg}[z_e(x)] - e\|^2 \;+\; \beta\,\|\mathrm{sg}[e] - z_e(x)\|^2
```
Uses a discrete codebook for latent representations, improving reconstruction while preserving quantized features. “Stop-gradient” prevents codebook collapse, letting the network learn meaningful embeddings without trivial solutions in the discrete bottleneck.

---

## **Nonlinear ICA Factorization** –

```math
\Huge
\mathbf{x} \;=\; f(\mathbf{s}),
\quad
\mathbf{s} \sim p(\mathbf{s}),
```
Recovers independent latent components from observed nonlinear mixtures. Unlike classical linear ICA, it handles more realistic mixing processes, crucial for uncovering structured signals in vision, audio, and multimodal data.

---

## **DDIM Sampling Step** –

```math
\Huge
x_{t-1}
=\;
\sqrt{\alpha_{t-1}}\Bigl(\tfrac{x_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\alpha_t}}\Bigr)
\;+\;
\sqrt{1-\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t)
\;+\;
\sigma_t\,z
```
A deterministic variant of diffusion-based sampling. DDIM allows faster generation and enhanced control over sample characteristics while preserving quality, offering an alternative path through the noise schedule without sacrificing realism.

---

## **Classifier-Free Guidance** –

```math
\Huge
\epsilon_\theta(x_t,\varnothing)
\;=\;
w\,\epsilon_\theta(x_t,y)
\;+\;
(1-w)\,\epsilon_\theta(x_t,\varnothing)
```
Provides conditional diffusion without a separate classifier. Adjusting \(w>1\) amplifies conditional signals (e.g., prompts) and steers generation toward desired attributes, making it essential in text-to-image or controlled synthesis tasks.

---

## **LSTM Language Model** –

```math
\Huge
h_t,\, c_t
=\;
\mathrm{LSTM}\bigl(x_{t-1},\,h_{t-1},\,c_{t-1}\bigr),
\quad
P(x_t \mid x_{<t})
=\;
\mathrm{softmax}(W\,h_t + b)
```
Generates sequences by iterating hidden and cell states. LSTMs capture long-range context, crucial for tasks like language modeling or speech generation, although more modern Transformers often surpass them in large-scale settings.

=====================================================================
### 📣 Evaluation Metrics
=====================================================================

## **Accuracy** – \( \text{Acc} = \frac{TP + TN}{TP + TN + FP + FN} \) – proportion of correct predictions (true positives + true negatives) over total – overall success rate of classification. It can be misleading for highly imbalanced datasets, so always pair it with complementary metrics.

----------------------

## **Precision (Positive Predictive Value)** – \( \text{Prec} = \frac{TP}{TP + FP} \) – fraction of predicted positives that are actually positive – evaluates exactness (few false alarms). It can be drastically affected by class imbalance and is crucial in risk-sensitive tasks like fraud detection.

----------------------

## **Recall (Sensitivity)** – \( \text{Rec} = \frac{TP}{TP + FN} \) – fraction of actual positives that are correctly predicted – evaluates completeness (few misses). It's vital in medical screening, ensuring fewer missed cases, but can penalize over-prediction.

----------------------

## **F1 Score** – \( F1 = \frac{2\,\text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}} \) – harmonic mean of precision and recall – balances the two, useful for imbalanced classes. A single F1 may hide imbalance in sub-classes, so consider per-class F1 for deeper insight.

----------------------

## **ROC AUC** – \( \displaystyle AUC = \int_0^1 \text{TPR}(x)\,dx \) (area under ROC curve) – probability a random positive is ranked above a random negative – metric of classifier separability (1.0 = perfect, 0.5 = chance). It remains consistent under class distribution shifts, but can be deceiving under extreme label imbalance.

----------------------

## **BLEU Score** – \( \displaystyle \text{BLEU} = \text{BP} \cdot \exp\Big(\frac{1}{N}\sum_{n=1}^N \ln p_n\Big) \) – geometric mean of modified n-gram precisions \(p_n\) times brevity penalty BP – evaluates machine translation quality against references. It does not measure semantic coherence, so high BLEU may still yield awkward translations.

----------------------

## **IoU (Jaccard Index)** – \( \text{IoU} = \frac{|A \cap B|}{|A \cup B|} \) – overlap / union of predicted region \(A\) and ground truth \(B\) – used in object detection/segmentation accuracy (1.0 = perfect overlap). Small shifts in predicted boundaries can significantly reduce IoU, revealing sensitivity to localization errors.

----------------------

## **PSNR** – \( \displaystyle \text{PSNR} = 10 \log_{10}\!\frac{MAX^2}{\text{MSE}} \) – \(MAX\): max signal value, MSE: mean squared error – measures reconstructed image quality in dB – higher is better (each 6dB ~ 2× RMSE improvement). Sensors with limited dynamic range can make high PSNR values unrepresentative of perceived quality.

----------------------

## **Specificity** – \( \text{Spec} = \frac{TN}{TN + FP} \) – true negative rate – fraction of actual negatives correctly identified – relevant for binary classification (along with sensitivity). High specificity is essential in ruling out conditions, reducing false positives in medical diagnostics.

----------------------

## **SSIM (Image Similarity)** – \( \displaystyle \text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} \) – compares images \(x\) and \(y\) in terms of luminance (\(\mu\)), contrast (\(\sigma\)), and structure (\(\sigma_{xy}\)) – value in \([0,1]\) (higher = more similar), used for image quality evaluation. SSIM can capture subtle structural distortions that simple MSE-based metrics often overlook.

-------------------

## **Balanced Accuracy** – \( \displaystyle \text{BalancedAcc} = \tfrac{1}{2}\Bigl(\tfrac{TP}{TP + FN} + \tfrac{TN}{TN + FP}\Bigr) \) – average of sensitivity and specificity, tackling class imbalance better than plain accuracy. It provides a fairer measure when positive and negative classes are unequally represented.

Interesting fact (35-40 words): Balanced Accuracy prevents inflated performance estimates on skewed data. It equally weights the true positive rate and true negative rate, making it a popular choice for medical and fraud detection tasks with imbalanced labels.

----------------------

## **Matthews Correlation Coefficient (MCC)** – \( \displaystyle \text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \) – a correlation measure between observed and predicted labels (range: \(-1\) to \(1\)).

Interesting fact (35-40 words): MCC captures all confusion matrix quadrants in one coefficient. It remains reliable even for very skewed class distributions and is often viewed as a comprehensive measure of binary classification performance.

----------------------

## **Average Precision (AP) / AUPRC** – \( \displaystyle \text{AP} = \sum_{k=1}^{n} \bigl(r_k - r_{k-1}\bigr)p_k \) – area under the precision-recall curve, aggregating precision at different recall levels.

Interesting fact (35-40 words): AP is especially informative for imbalanced datasets, where ROC curves can be overly optimistic. It emphasizes performance on correctly identifying positives, making it vital in information retrieval and anomaly detection tasks.

----------------------

## **Dice Coefficient** – \( \displaystyle \text{Dice} = \frac{2\,|A \cap B|}{|A| + |B|} \) – overlap metric ranging from 0 to 1, often used in image segmentation to measure boundary similarity.

Interesting fact (35-40 words): Dice is more forgiving to small contour shifts than IoU but still highlights segmentation errors. It’s popular in biomedical imaging to accurately capture organ or lesion boundaries where pixel-level overlap is crucial.

----------------------

## **Cohen’s Kappa** – \( \displaystyle \kappa = \frac{p_o - p_e}{1 - p_e} \) – compares observed accuracy \(p_o\) with expected accuracy \(p_e\) under random chance, ranging from \(-1\) to \(1\).

Interesting fact (35-40 words): Kappa addresses chance agreement, making it valuable when classes are highly imbalanced or labeled by human annotators. High kappa indicates strong consensus, beyond random guessing, in classification or annotation tasks.

----------------------

## **G-Mean** – \( \displaystyle \text{G-Mean} = \sqrt{\frac{TP}{TP + FN} \;\times\; \frac{TN}{TN + FP}} \) – geometric mean of sensitivity and specificity, capturing balanced classification performance.

Interesting fact (35-40 words): G-Mean preserves the balance between detecting positives and negatives, preventing a high score driven solely by one side. It’s especially helpful in streaming and online learning scenarios with skewed data distributions.

----------------------

## **Log Loss (Cross-Entropy Loss)** – \( \displaystyle \text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \bigl[y_i\ln(\hat{y}_i) + (1 - y_i)\ln(1 - \hat{y}_i)\bigr] \) – penalizes confident but incorrect predictions, measuring probabilistic classification quality.

Interesting fact (35-40 words): Log Loss directly evaluates how well models estimate probabilities. Lower values mean better calibrated predictions, making Log Loss a cornerstone for comparing outputs in competitions like Kaggle, where probability estimates are crucial.

----------------------

## **Brier Score** – \( \displaystyle \text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2 \) – mean squared error of predicted probabilities \(p_i\) for binary outcomes \(o_i\in \{0,1\}\).

Interesting fact (35-40 words): Brier Score is sensitive to both calibration and refinement of probabilistic forecasts. It rewards models that confidently predict correct outcomes and penalizes those that overconfidently predict wrong classes, making it essential in meteorological and clinical forecasts.

----------------------

## **Perplexity** – \( \displaystyle \text{PPL} = \exp\!\Bigl(-\frac{1}{N}\sum_{i=1}^N \ln p(x_i)\Bigr) \) – measures how well a probabilistic model predicts a sample, commonly used in language modeling (lower is better).

Interesting fact (35-40 words): Perplexity quantifies prediction uncertainty in language tasks. It’s a direct exponent of average negative log-likelihood. Large differences in perplexity often translate to substantial improvements in speech recognition or machine translation fluency.

----------------------

## **Mean Absolute Error (MAE)** – \( \displaystyle \text{MAE} = \frac{1}{N} \sum_{i=1}^N \lvert y_i - \hat{y}_i\rvert \) – average magnitude of errors without considering their direction, used in regression tasks.

Interesting fact (35-40 words): MAE yields a linear penalty for each unit of error, making it less sensitive to outliers than MSE. It’s straightforward to interpret for real-world forecasting like housing price predictions or time-series.

----------------------

## **Mean Squared Error (MSE)** – \( \displaystyle \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^{2} \) – average of squared differences between predictions and actual values, ubiquitous in regression.

Interesting fact (35-40 words): MSE heavily penalizes large residuals, making models focus on reducing significant errors. It’s essential in least squares regression and is the basis for fundamental algorithms like linear regression training.

----------------------

## **Root Mean Squared Error (RMSE)** – \( \displaystyle \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} \) – square root of MSE, preserving error units.

Interesting fact (35-40 words): RMSE is more interpretable in the same scale as targets, while still penalizing large deviations. It’s common in geospatial, meteorological, and energy load forecasting where absolute scale matching aids in understanding model performance.

----------------------

## **R² (Coefficient of Determination)** – \( \displaystyle R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2} \) – proportion of variance explained by the model.

Interesting fact (35-40 words): R² can become negative if the model is worse than just predicting the mean. Though popular, relying solely on R² risks missing bias, variance, and outlier effects in regression performance analysis.

----------------------

## **Adjusted Rand Index (ARI)** – \( \displaystyle \text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]} \) – measures clustering similarity, adjusting for chance grouping, ranging from \(-1\) to \(1\).

Interesting fact (35-40 words): ARI rewards exact cluster matches and penalizes random or near-random assignments. It’s widely used in unsupervised learning to compare discovered clusters with ground-truth labels or to compare different clusterings.

----------------------

## **Silhouette Score** – \( \displaystyle \text{Silhouette}(i) = \frac{b_i - a_i}{\max(a_i, b_i)} \) – for each sample \(i\), where \(a_i\) is average intra-cluster distance, \(b_i\) is average nearest-cluster distance.

Interesting fact (35-40 words): The Silhouette Score ranges from \(-1\) to \(+1\). High scores indicate well-separated, cohesive clusters. This metric helps pick optimal cluster numbers and compare clustering algorithms in exploratory data analysis.

----------------------

## **Normalized Mutual Information (NMI)** – \( \displaystyle \text{NMI}(U,V) = \frac{2 \, I(U;V)}{H(U) + H(V)} \) – compares clustering partitions \(U\) and \(V\), normalizing mutual information to [0,1].

Interesting fact (35-40 words): NMI captures shared information between two cluster assignments. A value near 1 suggests near-identical clustering. Unlike raw mutual information, NMI adjusts for cluster size and labeling, making cross-comparison of solutions fairer.

----------------------

## **Word Error Rate (WER)** – \( \displaystyle \text{WER} = \frac{S + D + I}{N} \) – ratio of substitution (S), deletion (D), and insertion (I) errors to total words (N), assessing speech or text recognition accuracy.

Interesting fact (35-40 words): WER spotlights practical usability of speech systems. Even minor improvements matter greatly in applications like voice assistants, where misrecognitions degrade user experience and can propagate errors into downstream tasks like semantic parsing.

----------------------

## **Hamming Loss** – \( \displaystyle \text{HammingLoss} = \frac{1}{N \times L} \sum_{i=1}^N \sum_{j=1}^L \mathbb{I}\bigl(\hat{y}_{ij} \neq y_{ij}\bigr) \) – fraction of misclassified labels in multi-label problems.

Interesting fact (35-40 words): Hamming Loss uniformly penalizes each label misclassification, making it straightforward for multi-label tasks like image tagging or text categorization. It doesn’t capture label correlations, so additional metrics might be needed for nuanced evaluations.

----------------------

## **F\(\boldsymbol{\beta}\)-Score** – \( \displaystyle F_\beta = \bigl(1+\beta^2\bigr)\frac{\text{Precision}\cdot\text{Recall}}{\beta^2 \,\text{Precision} + \text{Recall}} \) – generalization of F1, emphasizing recall if \(\beta>1\) or precision if \(\beta<1\).

Interesting fact (35-40 words): F\(\beta\)-Score allows domain-driven weighting between recall and precision. In medical triage (\(\beta>1\)), missing positives is risky, so recall is prioritized; in spam detection (\(\beta<1\)), precision is crucial to avoid false alarms.

----------------------

## **Mean Reciprocal Rank (MRR)** – \( \displaystyle \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} \) – averages the reciprocal of the first correct answer’s rank across queries \(Q\).

Interesting fact (35-40 words): MRR highlights how quickly a model retrieves the correct result, crucial in information retrieval, question-answering, and recommendation systems. High MRR means users see the relevant answer near the top of results.

----------------------

## **Normalized Discounted Cumulative Gain (nDCG)** – \( \displaystyle \text{nDCG} = \frac{\text{DCG}@k}{\text{IDCG}@k} \), where \(\text{DCG}@k = \sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2(i+1)}\).

Interesting fact (35-40 words): nDCG rewards putting highly relevant items high in ranking. It accounts for the diminishing value of lower-ranked items. Widely used in web search and recommender systems to judge the quality of ordered lists.

----------------------

## **Expected Calibration Error (ECE)** – \( \displaystyle \text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n}\Big\lvert \text{acc}(B_m) - \text{conf}(B_m)\Big\rvert \) – bins predictions by confidence, then measures difference between average confidence and accuracy.

Interesting fact (35-40 words): ECE gauges how well probabilities match observed outcomes. Low ECE indicates well-calibrated predictions, a must in critical applications like autonomous driving or healthcare, where overconfidence or underconfidence can lead to unsafe decisions.

----------------------

## **Jensen-Shannon Divergence (JSD)** – \( \displaystyle \text{JSD}(P\|Q) = \frac{1}{2}D_{\mathrm{KL}}(P\!\parallel\!M) + \frac{1}{2}D_{\mathrm{KL}}(Q\!\parallel\!M)\), \( M = \tfrac{1}{2}(P + Q)\) – symmetrized measure of distribution similarity.

Interesting fact (35-40 words): JSD remains finite even if distributions have zero probabilities for different events, unlike KL divergence. This makes it a stable measure for comparing model outputs in generative modeling or language distribution tasks.

----------------------

## **Kullback–Leibler Divergence (KL Divergence)** – \( \displaystyle D_{\mathrm{KL}}(P\|Q) = \sum_{x} P(x)\,\ln\!\bigl(\tfrac{P(x)}{Q(x)}\bigr) \) – non-symmetric measure of how one probability distribution diverges from another.

Interesting fact (35-40 words): KL divergence underpins variational inference and many modern Bayesian deep learning methods. It penalizes placing probability mass away from true data distributions, guiding models to learn more accurate, data-aligned parameter estimates.

----------------------

## **Earth Mover’s Distance (Wasserstein Distance)** – \( \displaystyle W(P,Q) = \inf_{\gamma\in\Gamma(P,Q)} \mathbb{E}_{(x,y)\sim \gamma}[\,\lVert x - y\rVert\,] \) – minimal “cost” to transform one distribution into another.

Interesting fact (35-40 words): Wasserstein distance has gained popularity in generative adversarial networks (WGANs). It provides meaningful gradients even when distributions don’t overlap, improving training stability and often yielding higher quality generated samples.

----------------------

## **Gini Impurity** – \( \displaystyle \text{Gini} = \sum_{k=1}^{K} p_k(1 - p_k) \) – used in decision trees to measure node impurity, where \(p_k\) is the proportion of class \(k\).

Interesting fact (35-40 words): Gini Impurity prefers binary splits leading to purer nodes. It’s computationally simpler than entropy but similarly guides tree-based models toward features that sharply separate classes, boosting interpretability and predictive performance.

----------------------

## **Top-\(k\) Accuracy** – \( \displaystyle \text{Top-}k \text{ Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{true label} \in \{\text{top }k\text{ predicted labels for }i\}) \) – fraction of samples whose correct label appears in top \(k\) predictions.

Interesting fact (35-40 words): Top-\(k\) metrics acknowledge near-misses in multi-class tasks, notably in ImageNet benchmarking. Models that place correct classes in high-ranked guesses can still be valuable, especially when integrated with further downstream decision steps.


=====================================================================
### 📣 Graph ML & Networks
=====================================================================

## **GCN Layer** – \( h_i^{(l+1)} = \sigma\!\Big( \sum_{j \in N(i)} \frac{1}{\sqrt{|N(i)|\,|N(j)|}} W\,h_j^{(l)} \Big) \) – \(N(i)\): neighbors of node \(i\) – aggregates neighboring features with normalization – semi-supervised Graph Convolutional Network layer (Kipf & Welling). GCNs can suffer from over-smoothing as layers deepen, causing node embeddings to converge to similar values.

----------------------

## **Graph Attention (GAT)** – \( e_{ij} = a^T [Wh_i || Wh_j],\;\; \alpha_{ij} = \frac{\exp(\sigma(e_{ij}))}{\sum_{k \in N(i)}\exp(\sigma(e_{ik}))},\;\; h_i' = \sigma\!\Big(\sum_{j \in N(i)} \alpha_{ij} W_v h_j\Big) \) – learns attention \( \alpha_{ij} \) on neighbors \(j\) of node \(i\) – weights neighbors in aggregation via a learned function \(a^T\) (self-attention on graph nodes). By tuning attention coefficients, GAT can reveal which neighbor relationships matter most for each node's representation.

----------------------

## **GraphSAGE** – \( h_i^{(l+1)} = \sigma\!\Big( W \big[ h_i^{(l)} || \text{AGG}_{j\in N(i)}(h_j^{(l)}) \big] \Big) \) – concatenates node’s current representation with an aggregate (mean/max/LSTM) of neighbor features, then linear & nonlinear – enables inductive node embedding by sampling neighbors (Hamilton et al.). GraphSAGE allows unseen nodes to be embedded by sampling neighbors, offering scalability to large dynamic graphs.

----------------------

## **PageRank** – \( r_{i}^{(t+1)} = \frac{1-\alpha}{N} + \alpha \sum_{j \in \text{in}(i)} \frac{r_j^{(t)}}{\deg(j)} \) – \(\alpha\): damping, \(N\): nodes count – iterative power method – ranks nodes by importance in graph (stationary distribution of random walk). PageRank can be personalized by adjusting the teleport vector to emphasize certain nodes or user interests.

-------------------

## **Chebyshev Graph Convolution (ChebNet)**

```math
\Huge
h^{(l+1)} = \sum_{k=0}^{K} \theta_{k}\,T_{k}(\tilde{L})\,h^{(l)},
\quad \tilde{L} = 2L/\lambda_{\max} - I
```
Chebyshev polynomials \(T_{k}\) enable fast localized convolution on graphs without explicit eigen-decomposition. This spectral method reduces computational overhead and handles large-scale graphs by truncating convolutions at \(K\)-hop neighborhoods.

---

## **Graph Isomorphism Network (GIN)**

```math
\Huge
h_{i}^{(l+1)}
= \text{MLP}\!\Big(\big(1 + \epsilon\big)\,h_{i}^{(l)} \;+\; \sum_{j \in N(i)} h_{j}^{(l)}\Big)
```
GIN achieves the expressive power of the Weisfeiler-Lehman test for graph isomorphism. It discriminates subtle graph structures by adaptively combining node features and ensuring injective neighborhood aggregations.

---

## **Relational Graph Convolutional Network (R-GCN)**

```math
\Huge
h_{i}^{(l+1)}
= \sigma\!\Bigg(
\sum_{r \in \mathcal{R}} \sum_{j \in N_{r}(i)} \frac{1}{\lvert N_{r}(i)\rvert} W_{r} \, h_{j}^{(l)}
\;+\; W_{0}\,h_{i}^{(l)}
\Bigg)
```
R-GCN handles multi-relational graphs with different edge types. Each relation \(r\) uses a separate transform \(W_{r}\). It is pivotal in knowledge graph completion and modeling heterogeneous domains.

---

## **Mixture Model Network (MoNet)**

```math
\Huge
h_{i}^{(l+1)}
= \sum_{j \in N(i)} \sum_{k=1}^{K} w_k(\mathbf{u}_{ij}) \,\Theta_k\,h_{j}^{(l)}
```
MoNet generalizes convolution via learnable kernel functions \(w_k\). It captures local neighborhood structures by parameterizing continuous kernel-weighting, enabling flexible adaptation to diverse graph geometries or irregular data domains.

---

## **Graph Autoencoder (GAE)**

```math
\Huge
\hat{A} = \sigma\!\bigl(Z \,Z^{T}\bigr),
\quad
Z = \text{Encoder}(X,A)
```
GAEs learn low-dimensional node embeddings \(Z\) by reconstructing adjacency \(\hat{A}\). They reveal latent connectivity patterns, aiding link prediction and community detection in unsupervised graph representation settings.

---

## **Variational Graph Autoencoder (VGAE)**

```math
\Huge
q(\mathbf{Z}\mid X,A)
= \prod_{i=1}^{N} q\bigl(\mathbf{z}_{i}\mid X,A\bigr),
\quad
\mathbf{z}_{i} \sim \mathcal{N}\!\bigl(\boldsymbol{\mu}_{i},\,\mathrm{diag}(\boldsymbol{\sigma}_{i}^{2})\bigr)
```
VGAE introduces probabilistic embeddings, capturing uncertainty in node representations. Its latent variables follow Gaussian distributions, enabling more robust link prediction and anomaly detection on noisy or uncertain graphs.

---

## **Deep Graph Infomax (DGI)**

```math
\Huge
\mathcal{L}_{\text{DGI}}
= \mathbb{E}_{\tilde{A},\tilde{X}}\bigl[\log D(\mathbf{h}, \mathbf{h}_{\tilde{A},\tilde{X}})\bigr]
+ \mathbb{E}_{A,X}\bigl[\log \bigl(1 - D(\mathbf{h}, \mathbf{h}_{A,X})\bigr)\bigr]
```
DGI maximizes mutual information between global and local node embeddings. It discerns informative structural features in an unsupervised manner, enabling powerful pretraining strategies for downstream graph tasks.

---

## **DiffPool**

```math
\Huge
S^{(l)} = \mathrm{softmax}\!\bigl(\mathrm{GNN}_{S}^{(l)}(A^{(l)}, X^{(l)})\bigr),\quad
X^{(l+1)} = S^{(l)T} X^{(l)},\quad
A^{(l+1)} = S^{(l)T} A^{(l)} S^{(l)}
```
DiffPool provides end-to-end hierarchical pooling by learning a cluster assignment matrix \(S^{(l)}\). It yields differentiable coarsening of graphs, enabling graph-level classification and interpretable multi-level structures.

---

## **Graph U-Net**

```math
\Huge
\bigl(X^{(l+1)}, A^{(l+1)}\bigr)
= \text{Unpool}\!\Big(\text{Pool}\bigl(X^{(l)}, A^{(l)}\bigr)\Big)
```
Graph U-Net uses pooling and unpooling operators analogous to image U-Nets. It selectively downsamples and upsamples node representations to capture multi-resolution features while preserving essential graph topology.

---

## **Node2Vec**

```math
\Huge
\min_{\Phi} \sum_{(u,v)\in D}
-\log \sigma\bigl(\Phi(u)^{\top}\,\Phi(v)\bigr)
\;-\;\sum_{n\in N_{\mathrm{neg}}}\log \sigma\bigl(-\Phi(u)^{\top}\,\Phi(n)\bigr)
```
Node2Vec extends random walks with biased sampling to capture homophily and structural equivalences. Its flexible neighborhood exploration strategy helps balance community-sensitive embeddings and long-range structural patterns.

---

## **DeepWalk**

```math
\Huge
\min_{\Phi} \sum_{(u,v)\in\text{Context}}
-\log p\bigl(v\mid u\bigr),
\quad
p\bigl(v\mid u\bigr)
= \frac{\exp\bigl(\Phi(u)^{\top}\,\Phi(v)\bigr)}{\sum_{x}\exp\bigl(\Phi(u)^{\top}\,\Phi(x)\bigr)}
```
DeepWalk treats truncated random walks on a graph as word-like sequences. By applying Skip-Gram, it learns embeddings where nodes co-occurring in walks share similar vector representations, aiding clustering and recommendation.

---

## **LINE**

```math
\Huge
\mathcal{L}_{\mathrm{1st}}
= -\sum_{(i,j)\in E} w_{ij}\,\log p_{j}(i),
\quad
p_{j}(i) = \sigma\bigl(\Phi(i)^{\top}\,\Phi(j)\bigr)
```
LINE handles large-scale graphs with first- and second-order proximity capturing local and global node relationships. Its separate objectives combine to enhance embeddings for various link prediction and node classification tasks.

---

## **Gated Graph Neural Networks (GGNN)**

```math
\Huge
h^{(l+1)}
= \mathrm{GRU}\!\Bigl(h^{(l)},\;A\,h^{(l)}\,W\Bigr)
```
GGNN uses gated recurrent units (GRU) to accumulate neighbor information in multiple propagation steps. This iterative message passing stabilizes training and captures complex dependencies in directed or undirected graphs.

---

## **Graph Recurrent Network (GRN)**

```math
\Huge
h_{i}^{(t+1)}
= \mathrm{GRU}\!\Bigl(
h_{i}^{(t)},\; \sum_{j \in N(i)}W\,h_{j}^{(t)}
\Bigr)
```
GRN fuses recurrent neural networks with graph adjacency for temporal node features. It updates node states using past hidden states and neighbor interactions, making it applicable in dynamic or streaming graph tasks.

---

## **ARMA Convolutions**

```math
\Huge
H^{(l+1)}
= \sum_{k=1}^{K} \alpha_{k}\,\bigl(I - \beta_{k}\,\tilde{L}\bigr)^{-1}\,H^{(l)}
```
ARMA filters approximate rational polynomial spectral kernels with fewer parameters. This achieves flexible frequency responses and deeper stable convolutions, potentially reducing over-smoothing in multi-layer graph networks.

---

## **GNN Explainer**

```math
\Huge
\min_{\Omega}
\;\mathcal{L}\bigl(\mathrm{GNN}(A \odot M_\Omega,\;X \odot F_\Omega),\,y\bigr)
\;+\;\lambda\,\|\Omega\|_{1}
```
GNN Explainer learns edge and feature masks \(\Omega\) that best justify a GNN’s prediction. By highlighting crucial substructures, it offers transparency and trust in high-stakes applications like biology or social networks.

---

## **SEAL (Subgraph Embedding-based Link Prediction)**

```math
\Huge
\mathrm{Score}(i,j)
= \mathrm{GNN}\!\bigl(\mathcal{G}_{i,j}, \,\text{Embeddings}\bigr),
\quad
\mathcal{G}_{i,j} = \text{enclosing subgraph around }(i,j)
```
SEAL extracts subgraphs around candidate edges and applies GNN-based embeddings for link prediction. Modeling localized topology gives strong predictive performance, revealing structural roles crucial for forming edges.

---

## **EvolveGCN**

```math
\Huge
W^{(t+1)} = \mathrm{RNN}\!\bigl(W^{(t)},\,\ldots\bigr)
```
EvolveGCN captures dynamic graphs by recurrently updating GCN weights \(W\). It adapts to temporal changes in topology or features, handling scenarios like evolving social networks or time-varying sensor data.

---

## **Temporal Graph Convolutional Network (T-GCN)**

```math
\Huge
h^{(t+1)}
= \mathrm{GRU}\!\Bigl(\mathrm{GCN}(A, h^{(t)}),\;h^{(t)}\Bigr)
```
T-GCN processes sequential graph-structured data using GCN for spatial dependencies and GRU for temporal patterns. It is widely used for forecasting in traffic or event sequences with graph-based correlations.

---

## **ST-GCN (Spatial-Temporal GCN)**

```math
\Huge
h^{(t+1)}
= \mathrm{Conv2D}\!\Bigl(\mathrm{GraphConv}\bigl(A,\,h^{(t)}\bigr)\Bigr)
```
ST-GCN interleaves graph convolutions for spatial relations with standard 2D convolutions for temporal dynamics. This architecture excels in skeleton-based action recognition, capturing how node positions evolve over time.

---

## **Graph Transformer**

```math
\Huge
\alpha_{ij}
= \mathrm{Softmax}\!\Bigl(
\frac{\bigl(x_{i}W_Q\bigr)\bigl(x_{j}W_K\bigr)^{\top}}{\sqrt{d}}
\Bigr),
\quad
x_{i}'
= \sum_{j}\alpha_{ij}\bigl(x_{j}W_V\bigr)
```
Graph Transformers apply self-attention on node pairs. They learn global dependency patterns and can handle irregular connectivity, making them suitable for tasks requiring broad context beyond local neighborhoods.

---

## **NetGAN**

```math
\Huge
\min_{G} \max_{D} \,\mathcal{L}(G,D)
= \mathbb{E}_{A\sim p(A)}\bigl[\log D(A)\bigr]
\;+\;
\mathbb{E}_{\hat{A}\sim G}\bigl[\log \bigl(1-D(\hat{A})\bigr)\bigr]
```
NetGAN generates realistic synthetic graphs via adversarial training. By modeling random walks, it preserves global and local structures. It is useful for anonymization, graph augmentation, or bridging data gaps.

---

## **Graph Memory Networks**

```math
\Huge
\mathbf{m}_{i}
= \mathrm{Attention}\bigl(h_{i}, M\bigr),
\quad
h_{i}'
= f\bigl(h_{i}, \mathbf{m}_{i}\bigr)
```
These networks maintain a memory matrix \(M\) capturing long-term relational cues. By retrieving relevant memory slices via attention, they enhance node representations in tasks like question answering over knowledge graphs.

---

## **Graph MLP**

```math
\Huge
h_{i}^{(l+1)}
= \mathrm{MLP}\!\bigl(h_{i}^{(l)}\bigr)
\;+\;
\mathrm{Msg}\!\bigl(\{h_{j}^{(l)} \mid j \in N(i)\}\bigr)
```
Graph MLP simplifies node-level updates by combining a local message aggregation \(\mathrm{Msg}\) with an MLP transform. It can serve as a lightweight baseline or be integrated with attention for advanced representation.

---

## **Grand**

```math
\Huge
H^{(l+1)}
= \alpha\,\bigl(I - \tilde{L}\bigr)H^{(l)}
\;+\;
\bigl(1-\alpha\bigr)\,H^{(l)},
\quad
\tilde{L}: \text{normalized Laplacian}
```
Grand stochastically augments node features through diffusion and dropout-like techniques. It mitigates over-smoothing by randomly propagating information, retaining node individuality in deeper networks.

---

## **Graph Heat Kernel**

```math
\Huge
e^{-tL}
= \sum_{k=0}^{\infty}
\frac{(-t)^{k}}{k!} \, L^{k},
\quad
L: \text{graph Laplacian}
```
The heat kernel diffuses signals over a graph. It reveals multi-scale node similarity, used in semi-supervised learning and kernel methods. Tuning diffusion time \(t\) controls the spread of information.

---

## **PDE-GCN**

```math
\Huge
\frac{\partial H}{\partial t}
= -\kappa\,\tilde{L}\,H \;+\; F\!\bigl(H, A\bigr)
```
PDE-GCN interprets graph convolution as partial differential equations over graph domains. This continuous viewpoint unifies smoothing and feature transformation, potentially improving stability and multi-scale representation power.

=====================================================================
### 📣 Ensemble Methods
=====================================================================

## **Bagging (Bootstrap Aggreg.)** –

# $` \displaystyle \hat{y} = \frac{1}{M}\sum_{m=1}^M f^{(m)}(x) `$

 – average of`$M$` model predictions`$f^{(m)}$` (each trained on random data subset) – reduces variance by model averaging (random forest uses bagged decision trees). Even a small correlation among base learners can hamper bagging’s benefit, making diversity crucial for performance.

----------------------

## **AdaBoost Weight** –

# $` \displaystyle \alpha^{(m)} = \frac{1}{2}\ln\!\frac{1 - err^{(m)}}{err^{(m)}} `$

–`$err^{(m)}$: error of weak learner`$m`$

 – computes model weight`$\alpha$` for ensemble – higher weight for more accurate weak models (AdaBoost). It can sometimes overemphasize misclassified examples, leading to potential overfitting if not carefully regularized.

----------------------

## **Gradient Boosting** –

# $` F_0(x) = \bar{y},\quad F_{m}(x) = F_{m-1}(x) + \nu\,h_m(x) `$

–`$h_m`$: new weak learner fit to residuals,`$\nu`$: shrinkage – stagewise additive model – each tree corrects errors of previous ensemble – powerful ensemble method (XGBoost). Early stopping is crucial to prevent models from memorizing noise in residuals, preserving generalization performance.

----------------------

## **Random Forest Vote** –

# $` \displaystyle \hat{y} = \arg\max_c \sum_{m=1}^M \mathbf{1}(h_m(x)=c) `$

–`$h_m`$:`$m`$th tree prediction – majority voting across ensemble of randomized decision trees – improves generalization and stability over single tree. Increasing the number of trees eventually saturates accuracy but also boosts computational overhead, so balancing is key.

-------------------

## **Stacking** – \( \displaystyle \hat{y} = f_{\text{meta}}\!\bigl(f_1(x), f_2(x), \ldots, f_M(x)\bigr) \) – trains a meta-learner on base predictions to combine them effectively – popular in competitions (Kaggle). Stacking can overfit if meta-learner sees the same data as base models, so proper cross-validation is essential.

-------------------

## **Blending** – \( \displaystyle \hat{y} = \beta\,f_{A}(x) + (1-\beta)\,f_{B}(x) \) – a simpler two-model ensemble that relies on a small held-out set to find blending weight \(\beta\) – faster than full stacking. Hidden data leakage in the blending phase can bias results, so strict separation is recommended.

-------------------

## **Soft Voting** – \( \displaystyle \hat{y} = \arg\max_{c}\,\sum_{m=1}^{M} p_{m}(x,c) \) – uses average predicted probabilities \(p_{m}(x,c)\) from each model – especially effective for probabilistic classifiers. Poorly calibrated probabilities can mislead soft voting, so regularizing or re-calibrating base models often helps.

-------------------

## **Mixture of Experts** – \( \displaystyle y = \sum_{m=1}^{M} g_{m}(x)\,f_{m}(x)\quad \text{with} \quad \sum_{m=1}^{M} g_{m}(x) = 1 \) – a gating function \(g_{m}\) distributes inputs to specialized experts. Properly trained gating functions can partition complex feature space, but unbalanced gating sometimes causes a single expert to dominate.

-------------------

## **Bayesian Model Averaging** – \( \displaystyle p(y \mid x) = \sum_{m=1}^M p\bigl(y \mid x, M_m\bigr)\,p\bigl(M_m \mid D\bigr) \) – weights each model by posterior probability – elegantly handles model uncertainty. Overly complex model sets risk posterior dilution, making careful prior selection and evidence-based pruning valuable.

-------------------

## **Ensemble Distillation** – \( \displaystyle \mathcal{L} = \alpha\,\mathcal{L}_{\mathrm{CE}}(y, s) + (1-\alpha)\,\mathrm{KL}\!\bigl(p_{\mathrm{ensemble}} \,\|\, p_{\mathrm{student}}\bigr) \) – transfers ensemble “knowledge” into a single student network – reduces deployment complexity. If the ensemble is too large or heterogeneous, distillation can become cumbersome, diminishing the speed gain.

-------------------

## **Rotation Forest** – \( \displaystyle \hat{y} = \arg\max_{c}\,\sum_{m=1}^M \mathbf{1}\bigl(h_m(R_m\,x) = c\bigr) \) – each tree sees rotated features via random PCA-based transformations – increases diversity among trees. Correlation among rotated subspaces can still arise, so random partitioning and rotation strategies matter greatly.

-------------------

## **Negative Correlation Learning** – \( \displaystyle \mathrm{NCL\,Loss} = \sum_{m=1}^M L\bigl(f_m(x), y\bigr) \;+\; \lambda\,\sum_{m=1}^M \sum_{j \neq m}\bigl(f_m(x)-f_j(x)\bigr)^2 \) – penalizes correlated errors. Choosing \(\lambda\) too high disrupts individual learner quality, while too low reduces the ensemble’s benefit.

-------------------

## **Snapshot Ensemble** – \( \displaystyle \hat{y} = \frac{1}{M}\sum_{m=1}^M f^{(m)}(x) \) – collects multiple neural network “snapshots” saved at different local minima during cyclic learning rate scheduling – ensemble in one training run. Wide learning rate oscillations can produce more diverse snapshots, boosting final accuracy.

-------------------

## **Ensemble Selection** – \( \displaystyle \hat{y} = \arg\max_{c}\sum_{m \,\in\,S}w_m\,\mathbf{1}\bigl(h_m(x) = c\bigr) \;\text{with}\;\sum_{m \in S} w_m = 1 \) – picks a subset \(S\) of candidate models with optimal weights – popular for large model libraries. Overzealous inclusion of redundant models may cause overfitting and increased computation.

=====================================================================
### 📣 Advanced & Miscellaneous
=====================================================================

## **LoRA (Low-Rank Adaptation)** –

# $` \Delta W = B\,A $` with`$B \in \mathbb{R}^{d\times r}, A \in \mathbb{R}^{r\times k}$

 – rank-$r$` decomposition of weight update – during fine-tuning,`$W = W_0 + \Delta W`$

 – drastically reduces trainable parameters for large language models by updating via low-rank matrices. Enables quick domain shifts in LLM fine-tuning with minimal overhead, fostering efficient adaptation for specialized tasks.

----------------------

## **Mixture-of-Experts** –

# $` y = \sum_{i=1}^M \pi_i(x)\,f_i(x) `$

–`$\pi_i(x)$: gating soft assignment (sums to 1),`$f_i`$: expert model outputs – conditionally routes input to a few expert networks – enables extremely large models with sparse activation (Switch Transformers). Allows distributed training across many experts, improving scalability and model capacity with reduced computational footprint.

----------------------

## **Neural ODE** –

# $` \frac{dh(t)}{dt} = f(h(t),\,t;\theta) `$

 – treats hidden states`$h(t)$` as continuous dynamical system defined by ODE – solutions via integrators (e.g. Runge-Kutta) – enables continuous-depth models (memory efficient, adaptive computation). Allows dynamic trade-offs between speed and accuracy by adjusting ODE solver steps during inference.

----------------------

## **Backpropagation** –

# $` \displaystyle \frac{\partial L}{\partial w_{ij}^{(l)}} = \delta_j^{(l)}\,a_i^{(l-1)},\;\; \text{with}\;\; \delta^L = \nabla_a L,\;\; \delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \phi'(z^{(l)}) `$

 – recursively computes gradients from output layer`$L$` back to weights – chain rule applied through network layers – core algorithm for training neural nets. Paved the way for deep networks by efficiently computing gradients, inspiring algorithms like forward-mode auto-differentiation and second-order methods.

----------------------

## **Xavier Initialization** –

# $` \mathrm{Var}(w) = \frac{2}{n_{in}+n_{out}} `$

 – set initial weights with variance`$2/(fan_{in}+fan_{out})$` (often via`$U[-\sqrt{6}/\sqrt{fan_{in}+fan_{out}},\,+\sqrt{6}/\sqrt{fan_{in}+fan_{out}}]$) – keeps activations variance stable across layers (good for sigmoid/tanh). Avoids early saturation of activation functions, stabilizing gradient flow and improving training speed in deeper networks.

----------------------

## **He Initialization** –

# $` \mathrm{Var}(w) = \frac{2}{n_{in}} `$

 – larger init variance for ReLU-based layers – preserves signal magnitude through layers by compensating ReLU’s zero output on half inputs – helps avoid vanishing/exploding activations at start. Also suitable for leaky ReLUs, maintaining balanced variance and accelerating convergence in architectures beyond standard ReLU-based layers.

----------------------

## **Bellman Optimality (Q)** –

# $` Q^*(s,a) = r(s,a) + \gamma \mathbb{E}_{s'}[\max_{a'} Q^*(s',a')]`$

–`$r(s,a)$: immediate reward,`$\gamma`$: discount – recursive definition of optimal Q-value – foundation of dynamic programming in RL (solved via value iteration). Forms the theoretical backbone for Q-learning, ensuring that iteratively applying the relation converges to the optimal policy.

----------------------

## **Q-Learning Update** –

# $` Q_{new}(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]`$

–`$\alpha`$: learning rate – off-policy TD algorithm – iteratively updates Q towards Bellman target`$r+\gamma \max Q(s',\cdot)$

 – learns optimal action-value function from experience. Converges even with off-policy data, making it robust for environments where exploring all actions is impractical.

----------------------

## **Policy Gradient (REINFORCE)** –

# $` \nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta} \Big[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)\,G_t \Big]`$

–`$G_t`$: cumulative reward from`$t`$

 – update weights`$\theta$` in direction that increases log-probability of actions proportional to observed return – basic policy optimization in RL. Can suffer from high variance, often mitigated by baselines or variance-reduction techniques like advantage functions and reward normalization.

----------------------

## **Bellman Expectation (Value)** –

# $` V^\pi(s) = \mathbb{E}_{a\sim\pi} [\,r(s,a) + \gamma V^\pi(s')\,]`$

 – under policy`$\pi`$

 – expected value = immediate reward + discounted next state value – consistency equation used in policy evaluation. Underpins iterative prediction methods like TD(0), shaping how agents estimate long-term returns from sampled experience.

----------------------

## **Advantage Function** –

# $` A(s,a) = Q(s,a) - V(s) `$

 – extra reward advantage of action`$a$` over average – measures how much better an action is than typical – used in actor-critic and PPO to reduce variance of policy gradient (by subtracting baseline`$V(s)$). Encapsulates how each action compares to average performance, guiding efficient exploration and faster convergence in policy optimization.

----------------------

## **Rotary Positional Embed.** –

# $` (q_{2i}, q_{2i+1}) \mapsto (q_{2i}\cos \theta + q_{2i+1}\sin \theta,\;\; -\,q_{2i}\sin \theta + q_{2i+1}\cos \theta) `$` with`$\theta = \frac{pos}{10000^{2i/d}}$

 – applies a rotation to each query/key pair of dimensions`$(2i,2i+1)$

 – injects relative position information in transformers (used in GPT-NeoX, LLama for long context handling). Facilitates extrapolation to longer sequences, making it helpful for tasks like document summarization or extended dialogue generation.

----------------------

## **AdaDelta** –

# $` s_t = \rho s_{t-1} + (1-\rho) g_t^2,\;\; \Delta \theta_t = -\,\frac{\sqrt{\Delta \theta_{t-1}^2+\epsilon}}{\sqrt{s_t+\epsilon}}\,g_t`$

 – no external learning rate – adapts step`$\Delta \theta$` using running RMS of gradients`$s_t$` and updates – improvement over AdaGrad for continuing training without decay. Eliminates the need for a global learning rate, allowing more robust parameter updates across a wide range of training scenarios.

----------------------

## **FGSM Attack** –

# $` \delta = \epsilon\,\mathrm{sign}(\nabla_x L(x,y)) `$

–`$\epsilon`$: small step – Fast Gradient Sign Method – generates adversarial example`$x+\delta$` by taking a single step in input space along gradient sign – exposes model vulnerabilities, used to augment training for robustness. Demonstrates how small, adversarial perturbations exploit local gradients, prompting development of stronger defenses like adversarial training.

----------------------

## **Weight Quantization** –

# $` q = \text{round}\!\Big(\frac{w}{\Delta}\Big) \cdot \Delta`$

–`$\Delta`$: quantization step (scale) – converts full-precision weight`$w$` to nearest discrete level – reduces model to low-bit representation (e.g. int8) for faster inference and smaller storage at slight accuracy cost. May involve post-training calibration or quantization-aware training, balancing hardware constraints with acceptable performance trade-offs.

----------------------

## **Model Pruning** – remove weights where`$|w| < \tau$` (below threshold) or lowest-magnitude`$p\%$` weights – eliminates least important connections – yields sparse model with fewer parameters – trade-off: compress model with minimal loss in accuracy (often combined with fine-tuning). Structured pruning targets entire filters or channels, providing more efficient inference on hardware with parallel processing constraints.

----------------------

## **Natural Gradient** –

# $` \theta_{t+1} = \theta_t - \eta\,F^{-1} \nabla_\theta L`$

–`$F`$: Fisher information matrix – preconditions gradient by curvature (FIM) – moves in parameter space invariant to reparameterization – used in advanced optimizers (e.g. in policy gradient methods for RL to improve convergence). Can be computationally expensive for large models, but approximations like Kronecker-Factored Curvature reduce overhead significantly.

----------------------

## **One-Hot Encoding** –

# $` \text{onehot}(y)_i = \mathbf{1}\{y = i\} `$

 – represents categorical variable`$y$` as binary vector (1 at index of category, 0 elsewhere) – enables categorical inputs to be used in ML models (converted to numerical features). Expands feature space quickly for high-cardinality data, motivating alternatives like embeddings or hashing tricks.

----------------------

## **SVM Dual Form** –

# $` L_D(\alpha) = \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i,x_j) `$` s.t.`$\sum_i \alpha_i y_i = 0,\;\alpha_i \ge 0$

 – Lagrange dual of SVM

 –`$\alpha_i$` are support values – solution gives`$\alpha_i$` for support vectors defining decision boundary (enables kernel trick). Allows nonlinear decision boundaries in high-dimensional spaces through kernels, boosting performance without explicitly mapping data.

----------------------

## **t-SNE KL Loss** –

# $` C = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}} `$

–`$p_{ij}$: high-dim Gaussian affinity,`$q_{ij}$: low-dim Student t affinity – t-SNE finds low-dimensional embedding that minimizes KL divergence between original and embedded pairwise similarities – used for visualizing high-dimensional data clusters. Careful selection of perplexity and learning rate is crucial to avoid misleading cluster patterns or excessive crowding in plots.

-------------------

## **Adam** –

```math
\Huge
\hat{m}_t = \frac{\beta_1 \hat{m}_{t-1} + (1-\beta_1)\,g_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{\beta_2 \hat{v}_{t-1} + (1-\beta_2)\,g_t^2}{1 - \beta_2^t}, \quad
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

Merges momentum and adaptive learning rates, offering fast convergence and stability. Widely used in deep networks, it dynamically rescales gradients, making it robust to noisy objectives and sparse gradients.

---


## **Gumbel-Softmax** –

```math
\Huge
y_i = \frac{\exp\bigl((\log \pi_i + g_i)/\tau\bigr)}{\sum_j \exp\bigl((\log \pi_j + g_j)/\tau\bigr)}
```
Samples from a categorical distribution via continuous relaxation. The Gumbel noise \(g_i\) transforms discrete sampling into a differentiable approximation, enabling gradient-based training of models that need to choose discrete classes.

---

## **Fisher’s Linear Discriminant** –

```math
\Huge
w^* = S_W^{-1} \bigl(\mu_1 - \mu_2\bigr)
```
Projects data onto a line that maximizes class separability. \(S_W\) is the within-class scatter, while \(\mu_1, \mu_2\) are class means. It underlies Linear Discriminant Analysis, balancing scatter between and within classes.

---

## **DropConnect** –

```math
\Huge
\widetilde{W} = M \odot W
```
Randomly drops weights (instead of activations, as in Dropout). This encourages strong regularization by enforcing redundancy in parameter space, potentially improving generalization. Often used in fully-connected layers to mitigate overfitting.

---

## **Proximal Policy Optimization (PPO)** –

```math
\Huge
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t \Bigl[\min\bigl(r_t(\theta)\,A_t,\;\mathrm{clip}\bigl(r_t(\theta),1-\epsilon,\,1+\epsilon\bigr)\,A_t\bigr)\Bigr]
```
Clip-based policy gradient that penalizes large deviations from old policies. Balances policy improvement with stability, making RL training more robust and sample-efficient, widely adopted in continuous control and large-scale reinforcement tasks.

---

## **Double Q-Learning** –

```math
\Huge
Q^A(s,a)\leftarrow Q^A(s,a)+\alpha\Bigl[r + \gamma\,Q^B\bigl(s',\,\arg\max_a Q^A(s',a)\bigr) - Q^A(s,a)\Bigr]
```
Addresses overestimation bias in single Q-learning by maintaining two Q-functions \(Q^A\) and \(Q^B\). Each updates using the other’s best action, yielding more accurate value estimates and better stability in many off-policy RL settings.

---

## **Bayesian Inference (Posterior)** –

```math
\Huge
p(\theta\mid X) \;=\; \frac{\,p(X\mid \theta)\,p(\theta)\,}{\int p(X \mid \theta)\,p(\theta)\,d\theta}
```
Computes posterior distributions over parameters \(\theta\) given data \(X\). Forms the foundation for Bayesian ML, providing principled uncertainty estimates and the ability to incorporate prior beliefs into model training.

---

## **Ensemble Averaging** –

```math
\Huge
\hat{y} \;=\; \frac{1}{M}\sum_{m=1}^{M} f_m(x)
```
Averaging outputs from multiple independently trained models. Reduces variance, often boosting accuracy and robustness to outliers. Commonly used in competitions, where blending diverse model architectures yields strong final predictions.

---

## **Monte Carlo Dropout** –

```math
\Huge
p(y\mid x)\;\approx\;\frac{1}{T}\,\sum_{t=1}^{T} p\bigl(y\mid x,\boldsymbol{\omega}^{(t)}\bigr)
```
Interprets Dropout at test time as approximate Bayesian inference by sampling different dropout masks. Produces predictive uncertainty without expensive sampling of full posterior distributions, helpful for safety-critical or risk-sensitive tasks.

---

## **Graph Neural Network Update** –

```math
\Huge
h_i^{(l+1)} = \sigma\Bigl(W^{(l)}\,h_i^{(l)} \;+\; \sum_{j \in \mathcal{N}(i)} U^{(l)}\,h_j^{(l)}\Bigr)
```

Aggregates neighbor features in graph-structured data. Iterative message passing captures local connectivity information, enabling tasks like node classification, link prediction, or graph-level classification across social networks, molecules, and beyond.

---

## **Hopfield Network Energy** –

```math
\Huge
E(\mathbf{s}) = -\tfrac{1}{2}\,\sum_{i,j} w_{ij}\,s_i\,s_j \;+\;\sum_i \theta_i\,s_i
```
Recurrent network storing patterns as stable attractor states. Energy minimization dynamics converge to these states, enabling associative memory retrieval. Modern attention mechanisms echo Hopfield’s idea of embedding patterns as attractor basins.

---

## **Contrastive Loss (InfoNCE)** –

```math
\Huge
L_{\mathrm{InfoNCE}} = -\log\!\Bigl(\frac{\exp(\mathrm{sim}(h_i,h_j)/\tau)}{\sum_{k}\exp(\mathrm{sim}(h_i,h_k)/\tau)}\Bigr)
```
Encourages similar representations for positive pairs and dissimilar for negatives, widely used in contrastive self-supervised learning. It uses a softmax over similarity scores, fueling breakthroughs in representation learning (e.g., SimCLR, MoCo).


---


## **SWAG (Stochastic Weight Averaging-Gaussian)** –

```math
\Huge
\theta_{\mathrm{SWAG}} \sim \mathcal{N}\Bigl(\bar{\theta}, \,\tfrac12 \,\Sigma\Bigr),
\quad
\bar{\theta} = \tfrac{1}{T}\sum_{t=1}^{T}\theta_t
```
Approximates a Gaussian over neural network parameters. Improves calibration and predictive uncertainty by capturing first- and second-moment statistics of SGD iterates, producing a simple Bayesian-like ensemble from a single training run.

---

## **Sparsemax** –


```math
\Huge
\mathrm{sparsemax}(\mathbf{z}) = \arg\min_{\mathbf{p}\in\Delta} \|\mathbf{p} - \mathbf{z}\|^2
\quad\text{subject to } \mathbf{p}\ge 0,\;\sum_i p_i=1
```


Maps logits to a probability distribution with possible zero entries. Unlike softmax, it can produce sparse outputs, making it suitable for attention mechanisms or classification tasks where selective focus or fewer active labels is desired.