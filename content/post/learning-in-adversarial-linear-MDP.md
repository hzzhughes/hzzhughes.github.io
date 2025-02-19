---
date : '2025-02-18'
draft : true
title : 'Learning in Adversarial Linear MDP'
katex : true
tags : ['learing theory','reinforcement learning']
---

Linear function approximation is a common approach in theoretical RL to understand generalization. As mentioned in [previous post](/post/generalization-in-rl/), there are already a line of works in such area. But learning in adversatial setting still remains an open question. The best solution so far is by [[Liu et.al. 2023]](https://arxiv.org/pdf/2310.11550), which proposed one computationally inefficient algorithm that ensures a regret of $\tilde{O}(\sqrt{K})$ and an efficient algorithm that guarantees a regret of $\tilde{O}(K^{3/4})$.

## Preliminaries

## Some Simple Examples

Let's first by consider some easy scenario in such setting to help us better understand it.

**Example 1**(Adversarial Linear Bandit):
Let's say we are considering the case when $H=1$.

In such case, we have

$$
\begin{align*}
{\cal R_{\it K}}=&{\Bbb E}[\sum_{k=1}^K V^{\pi_k}(s_1;l_k)]-\min_\pi\sum_{k=1}^KV^\pi(s_1;l_k)\\\\
=&\sum_{k=1}^K\Bbb E[\phi^T(s_1,a_{k,1})\theta_{k,1}]
-\min_a\sum_{k=1}^K\phi^T(s_1,a)\theta_{k,1}\\\\
=&\sum_{k=1}^K\langle \Bbb E[\phi(s_1,a_{k,1})]-\phi(s_1,a^\ast),\theta_{k,1} \rangle\\\\
=&\sum_{k=1}^K\langle \Bbb E[\phi_k]-\phi^\ast,\theta_{k,1} \rangle
\end{align*}
$$
where $\phi_k=\phi(s_1,a_{k,1})\in\Phi$,$\Phi=\\{\phi(s_1,a)\mid a\in{\cal A}\\}$, $\phi^\ast=\arg\min_{\phi\in\Phi}\sum_{k=1}^K\phi^T\theta_{k,1}$
<!-- 
If the theta is known to us, then the problem becomes simple
$$
{\cal V}=\min_{\phi_{1:K}}\max_{\theta_{1:K}}
\sum_{k=1}^K\langle \phi_k-\phi^*,\theta_{k,1} \rangle
$$

Since it can be easily proved that the regret is affine in $\phi_k$ and convex in $\theta_k$, we can apply [Von Neumann's minimax theorem](https://en.wikipedia.org/wiki/Minimax_theorem) here and obtain

$$
{\cal V}=\max_{\theta_{1:K}}\min_{\phi_{1:K}}
\sum_{k=1}^K\langle \phi_k-\phi^*,\theta_{k,1} \rangle=0
$$ 
-->
Then the whole problem becomes an adversarial linear bandit problem which already have [theoretical bounds](https://banditalgs.com/2016/11/24/adversarial-linear-bandits/) in regret of $O(\sqrt{dK\log(\vert\cal A\vert)})$

**Example 2**(Planning Problen):
Let's say we are considering the case when $\Bbb P$ is known.

**Example 3**(Oblivious One):
Let's say we are considering the case when the loss function is identical through out all the episodes.
