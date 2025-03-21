---
date : '2025-02-18'
draft : false
title : 'Learning in Adversarial Linear MDP'
katex : true
tags : ['learing theory','reinforcement learning']
---

Linear function approximation is a common approach in theoretical RL to understand generalization.
As mentioned in [previous post](/post/generalization-in-rl/),
there are already a line of works in such area.
But how to learn efficiently in adversarial setting still remains an open question.
The best solution so far is by [[Liu et.al. 2023]](https://arxiv.org/pdf/2310.11550),
which proposed one computationally inefficient algorithm that ensures a regret of $\tilde{O}(\sqrt{K})$
and an efficient algorithm that guarantees a regret of $\tilde{O}(K^{3/4})$.

## Preliminaries

$$
\begin{align*}
    {\cal S}\quad&\text{state space}\\\\
    {\cal A}\quad&\text{action space}\\\\
    H\quad&\text{horizon}\\\\
    s\quad&\text{state}\\\\
    a\quad&\text{action}\\\\
    h\quad&\text{step}\\\\
    l_h(s,a)\quad&\text{loss}\\\\
    \Bbb P_h(\cdot\vert s,a)\quad&\text{transition probability}\\\\
    K\quad&\text{number of episodes}\\\\
    k\quad&\text{episode}\\\\
\end{align*}
$$

**Definition**(Regret):
$$
{\cal R_{\it K}}
={\Bbb E}[\sum_{k=1}^K V_1^{\pi_k}(s_1;l_k)]
-\min_\pi\sum_{k=1}^KV_1^\pi(s_1;l_k)
$$

**Definition**(Linear MDP):
MDP $\cal M$ is a *linear MDP* with a feature map $\phi:\cal S\times A\mapsto \Bbb R^d$,
if for any $h\in[H]$,
there exists $d$ unknown measures $\psi_h=(\psi_h^{(1)},\ldots,\psi_h^{(d)})$ over $\cal S$
and an unknown vector $\theta_h\in \Bbb R^d$,
such that for any $(s,a)\in\cal S\times A$,
we have

$$
l_h(s,a)=\theta_h^T\phi(s,a)\\\\
\Bbb P_h(\cdot\vert s,a)=\psi_h(\cdot)^T\phi(s,a),
$$

**Assumption A**:(Adversarial Loss):
Before the game starts,
an adversary arbitrarily chooses the loss functions for all episodes
$l_h^k:{\cal S\times A}\mapsto[0,1],k\in[K],h\in[H]$,
and does not reveal them to the learner.

## Some Simple Examples

Let's first consider some easy scenarios in such setting to help us better understand it.

**Example 1**(Adversarial Linear Bandit):
Let's say we are considering the case where $H=1$.

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
Then the whole problem becomes an adversarial linear bandit problem
which already have [theoretical bounds](https://banditalgs.com/2016/11/24/adversarial-linear-bandits/)
in regret of $O(\sqrt{dK\log(\vert\cal A\vert)})$

**Algorithm 1**(Exp3 for Adversarial Linear MAB):

- FOR $k=1,\ldots,K$
  - $\tilde p(a)\propto \exp(-\eta\sum_{\tau=1}^{k-1}\hat l_\tau(a))$
  - $p=\gamma\pi+(1-\gamma)\tilde{p}$ where $\pi$ is some exploration strategy
  - take action $a$ following mixed strategy $p$
  - receive loss $l_k$
  - $\Lambda=\sum_{a\in\cal A}p(a)aa^T$
  - $\theta^k=\Lambda^{-1}al_k$
  - $\hat l_k(\cdot)=\langle\theta_k,\cdot\rangle$

**Theorem 1**:
*Assume $\cal A$ is non-empty and let $A=|\cal A|$.
For any exploration distribution $\pi$,
for some parameters $\eta$ and $\gamma$,
for all $\theta_{1:K}$ with $\theta_k\in\Theta$,
the regret of Algorithm 1 satisfies*

$$
{\cal R_{\it K}}
\le2\sqrt{(2g(\pi)+d)K\log(A)}
$$

**Example 2**(Planning Problen):
Let's say we are considering the case where $\Bbb P$ is known.

**Example 3**(Oblivious Reward):
Let's say we are considering the case where the loss function is identical through out all the episodes.

**Algorithm 2**(LSVI-UCB):

As discussed in [previous post](/post/generalization-in-rl/#linear-mdp),
we already have some algorithms(e.g. LSVI-UCB) that can guarantee a regret of $O(\sqrt{d^3H^3T})$

- FOR $k=1,\ldots,K$
  - receive initiala state $s_1$
  - FOR $h=H,\ldots,1$
    - $\Lambda=\sum_{\tau=1}^{k-1}\phi_h^\tau(\phi_h^\tau)^T+\lambda I$
      ($\lambda I$ here is some small term to keep $\Lambda$ full rank)
    - $w_h=\Lambda^{-1}\sum_{\tau=1}^{k-1}\phi_h^\tau(r_h^\tau+\max_aQ_{h+1}(s_{h+1}^\tau,a))$
    - $Q_h(\cdot,\cdot)=\max\\{H,w_h^T\phi(\cdot,\cdot)+\beta\sqrt{\phi^T(\cdot,\cdot)\Lambda^{-1}\phi(\cdot,\cdot)}\\}$
  - FOR $h=1,\ldots,H$
    - take action $a_h^k=\arg\max_a Q_h(s_h^k,a)$
    - observe $s_{h+1}^k$ and receive reward $r_h^k$

**Proposition 1**:
*For a linear MDP, for any policy $\pi$,
there exist weights $\\{w_h^\pi\\}_{h\in[H]}$
such that for any $(s,a,h)\in{\cal S\times A}\times[H]$,
we have $Q_h^\pi(s,a)=\langle\phi(s,a),w_h^\pi\rangle$*

## Implementing Exp3

**Algorihtm 3**:

- FOR $k=1,\ldots,K$
  - receive initiala state $s_1$
  - FOR $h=1,\ldots,H$
    - $\tilde\pi_h^k(a\mid s)\propto \exp(-\eta\sum_{\tau=1}^{k-1}\hat Q_h^\tau(s,a))$
    - $\pi_h^k=\gamma\beta_h^k+(1-\gamma)\tilde\pi_h^k$, where $\beta_h^k\in\Delta(\cal A_h)$ is some exploration distribution
    - take action $a$ following mixed strategy $\pi_h^k$
    - observe $s_{h+1}^k$ and receive loss $l_h^k$
  - FOR $h=H,\ldots,1$
    - $\Lambda=\sum_{a\in\cal A_h}\pi_h^k(a)\phi(s_h^k,a)\phi(s_h^k,a)^T$
    <!-- - $\hat w_h^k=\Lambda^{-1}\phi_h^k(l_h^k+\hat V_{h+1}^k(s_{h+1}^k))$ -->
    - $\hat w_h^k=\Lambda^{-1}\phi_h^k(\sum_{h'=h}^Hl_{h'}^k)$
    - $\hat Q_h^k(\cdot,\cdot)=\hat w_h^T\phi(\cdot,\cdot)$
    <!-- - $\hat V_h^k(\cdot)=\sum_{a\in\cal A}\pi_h^k(a)\hat Q_h^k(\cdot,a)$ -->

By choosing some proper exploration distribution $\pi$,
we assures that $\Lambda$ is non-singular.

**Proposition 2**:
*For any $k\in [K]$, $h\in[H]$ and policy $\pi^k$,
we have $\Bbb E [\hat Q_h^k]=Q_h^k$*

*Proof.*
Here we use induction to prove this proposition.

<!-- First we claim that $\hat V_{H+1}^k=V_{H+1}^k=0$ is surely unbiased -->

Then we claim that if $w_{h+1}^k$ is unbiased,
we can also prove $w_h^k$ is unbiased.
The proof of this claim is as following
$$
\begin{align*}
  \Bbb E[\hat w^k_h]
  =&\Bbb E[
    \Lambda^{-1}
    \phi_h^k
    (
      \sum_{h'=h}^Hl_{h'}^k
    )]\\\\
  =&
    \Bbb E[
      \Lambda^{-1}
      {\phi_h^k}Q_h^k(s_h^k,a_h^k)
    ]&\text{}\\\\
  =&
    \Bbb E[
      \Lambda^{-1}
      {\phi_h^k}{\phi_h^k}^T
    ]w_h^k
    &\text{by Proposition 1}\\\\
  =&w_h^k&\text{by choice of }\Lambda
\end{align*}
$$

By induction,
we know the proposition is true,

$$\tag*{$\blacksquare$}$$
which completes the proof.

**Theorem 2**(Main Result):
*Assume $\cal A$ is non-empty and let $A=|\cal A|$.
For any exploration distribution $\pi$,
for some parameters $\eta$ and $\gamma$,
for all $\theta_{1:K}$ with $\theta_k\in\Theta$,
the regret of Algorithm satisfies*

$$
{\cal R_{\it K}}
\le2\sqrt{(2g(\pi)+d)H^4K\log(A)}
$$

**Lemma**(Hedge):
*For any $h\in[H],s\in\cal S_h$*
$$
\sum_{k=1}^K
  \Bbb E_{a\sim\pi_h^k}\hat Q_h^k(s,a)/H
  -\min_a\sum_{k=1}^K\hat Q_h^k(s,a)/H
\le
  \frac{\log(A)}{\eta_h}
  +2\gamma_h K
  +\eta_h\sum_{k=1}^K\Bbb E_{a\sim\pi_h^k}[
    \hat Q_h^k(s,a)^2/H^2]
$$

Note that both $\eta$ and $\gamma$ here can depend on $h$ and $s$.
<!-- $$
\begin{align*}
  \hat V_h^k(s_h^k)
  &=
    \sum_{a\in\cal A_h}\pi_h^k(a\mid s_h^k)\hat Q_h^k(s_h^k,a)\\\\
  &=
    \sum_{a\in\cal A_h}\pi_h^k(a\mid s_h^k){\hat w_h^k}^T\phi(s_h^k,a)\\\\
  &=
    \sum_{a\in\cal A_h}\pi_h^k(a\mid s_h^k)\phi(s_h^k,a_h^k)^T\Lambda^{-1}\phi(s_h^k,a)
    (l_h^k+\hat V_{h+1}^k(s_{h+1}^k))\\\\
  &=l_h^k+\hat V_{h+1}^k(s_{h+1}^k)
\end{align*}
$$ -->
*Proof of Theorem.*
We know
$$
\begin{align*}
{\cal R_{\it K}}
=&\Bbb E[\sum_{k=1}^K V^k_1(s_1)-\sum_{k=1}^K V^\ast_1(s_1)]\\\\
=&\Bbb E\left[[
  \min_a\sum_{k=1}^KQ_1^k(s_1,a)-\min_a\sum_{k=1}^KQ_1^\ast(s_1,a)
  ]+[
  \sum_{k=1}^K\Bbb E_{a\sim\pi_1^k}Q_1^k(s_1,a)-\min_a\sum_{k=1}^KQ_1^k(s_1,a)]\right]\\\\
\le&
  \Bbb E\left[
    \Bbb E_{a\sim\pi^\ast_1}[\sum_{k=1}^KQ_1^k(s_1,a)-\sum_{k=1}^KQ_1^\ast(s_1,a)]
    +[\sum_{k=1}^K\Bbb E_{a\sim\pi_1^k}Q_1^k(s_1,a)-\min_a\sum_{k=1}^KQ_1^k(s_1,a)]\right]\\\\
=&\Bbb E\left[
  \Bbb E_{a_1\sim\pi_1^\ast,s_2\sim\Bbb P(s_1,a_1)}[
    \sum_{k=1}^KV_2^k(s_2)-
    \sum_{k=1}^KV_2^\ast(s_2)
  ]+[
    \sum_{k=1}^K\Bbb E_{a\sim\pi_1^k}Q_1^k(s_1,a)
    -\min_a\sum_{k=1}^KQ_1^k(s_1,a)
  ]
\right]\\\\
\end{align*}
$$

by recursively using such technique we finally prove that

$$
{\cal R_{\it K}}
\le\Bbb E[\sum_{h=1}^H\Bbb E_{s_h^k\sim\mu_h^*}[\sum_{k=1}^K
  \Bbb E_{a\sim\pi_h^k}Q_h^k(s_h^k,a)
  -\min_{a\in\cal A_h}\sum_{k=1}^KQ_h^k(s_h^k,a)]
]
$$

where $\mu_h^\ast$ is the occupancy measure of $\pi^\ast$

At this point we can finally use the consistency of our estimator and hedge lemma.

We know when $\eta_h\hat Q_h^k/H\ge-1$, we have
$$
\begin{align*}
  \Bbb E[\sum_{k=1}^K
  \Bbb E_{a\sim\pi_h^k}Q_h^k(s_h^k,a)
  -\min_{a\in\cal A_h}\sum_{k=1}^KQ_h^k(s_h^k,a)]
  =&\Bbb E[\sum_{k=1}^K
  \Bbb E_{a\sim\pi_h^k}\hat Q_h^k(s_h^k,a)
  -\min_{a\in\cal A_h}\sum_{k=1}^K\hat Q_h^k(s_h^k,a)]\\\\
  \le&
    H(\frac{\log(A)}{\eta_h}
    +2\gamma_h K
    +\eta_h\sum_{k=1}^K\Bbb E_{a\sim\pi_h^k}[
      \hat Q_h^k(s_h^k,a)^2/H^2])\\\\
\end{align*}
$$

And

$$
\Bbb E_{a\sim\pi_h^k}[\hat Q_h^k(s,a)^2/H^2]
=(\sum_{h'=h}^Hl_{h'}^k/H)^2{\phi_h^k}^T\Lambda^{-1}\phi_h^k
\le{\phi_h^k}^T\Lambda^{-1}\phi_h^k
=\text{trace}(\phi_h^k{\phi_h^k}^T\Lambda^{-1})
=d
$$

And by choosing a proper $\gamma_h$,
we assure that $|\eta_h\hat Q_h^k/H|\le 1$

Finally,
by choosing some proper $\eta_{1:H}$,
we prove the regret bound of $\sqrt{(2g(\pi)+d)H^4K\log(A)}$

$$\tag*{$\blacksquare$}$$
which completes the proof.

## Improvements

Since exp3 algorithm is computationally inefficient,
we can still make some improvements.
For instance,
we may be able to implement other FTRL or BLO algorithms following similar technique we used above.
