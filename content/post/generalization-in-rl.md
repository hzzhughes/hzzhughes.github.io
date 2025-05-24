---
date : '2025-02-10'
draft : false
title : 'Some Notes on Generalization in RL'
katex : true
tags : ['learing theory','reinforcement learning']
---
>This is a personal note on some of the ideas in RL the writer learned so far. Many of them could be incorrect and I'd be happy if you could let me know.
>

In this article,
we are going to include some of these ideas proposed in such field.
Note that most of the content actually comes from [this book](https://rltheorybook.github.io/).

## Background

First of all,
let's define the notations and settings.
$$
\begin{align*}
    {\cal S}\quad&\text{state space}\\\\
    {\cal A}\quad&\text{action space}\\\\
    H\quad&\text{horizon}\\\\
    s\quad&\text{state}\\\\
    a\quad&\text{action}\\\\
    h\quad&\text{step}\\\\
    r_h(s,a)\quad&\text{reward}\\\\
    \Bbb P_h(\cdot\vert s,a)\quad&\text{transition probability}\\\\
    K\quad&\text{number of episodes}\\\\
    k\quad&\text{episode}\\\\
\end{align*}
$$
We say we are in episodic setting and consider the finite horizon MDP ${\cal M}=({\cal S}, {\cal A}, H, \Bbb P, r)$.
We are also going to use this setting in most parts of this article.

### Limits in tabular setting

In tabular setting,
we are faced with finite state and action space.
However, sometimes in practice, we have to deal with infinite time state or action space.
Methods we used in tabular setting like UCB-VI can not be directly used in such cases.

### Challenges in agnostic learning

The second question is that often we do not have much assumption on the MDP in tabular setting.
But sometimes we do want to incorporate our prior knowledge in our algorithm to achieve smaller regret and faster convergence even in tabular settings.

**Lemma**(unbiased estimation of $V_0^\pi(\mu)$):
$$
V_1^\pi(\mu)=|{\cal A}|^H\cdot\Bbb E_{\tau\sim Pr_{Unif_{\cal A}}}\left[\Bbb I(\pi(s_1)=a_1,\ldots,\pi(s_H)=a_H)\sum_{h=1}^Hr(s_h,a_h)\right]
$$

**Proposition**(An "Occam's razor bound" on RL):
with probability at least $1-\delta$, we have that
$$
V_0^{\hat{\pi}}(\mu)\ge\max_{\pi\in\Pi} V_0^\pi(\mu)-H|{\cal A}|^H\sqrt{\frac{2}{N}\log\frac{2|\Pi|}{\delta}}
$$
which is equivalent to say,
to achieve a $\epsilon$-optimal policy with probability at least $1-\delta$,
we need

$$
N\ge H|A|^H\frac{2\log(2|\Pi|/\delta)}{\epsilon^2}
$$

We are facing a sample complexity bound exponential in $H$

So here comes the question

>What are the minimal structural assumptions that empower sample-efficient RL?
>
>---*[[Jin et.al. 2021]](https://arxiv.org/pdf/2102.00815)*

## A Naive Thought: Discretization

A simple way to address the infinite state/action space may be discretization,
just like what we do to the infinite reference class in supervised learing.

And for sure discretization can also be treated as somewhat a generalization method
as it generalizes one points to its surrounding area.

But even though you can approximate a value function to any error under Lipschitzness,
you still only utilize continuity.

## Linear Approximation

A simple yet powerful way is using linear approximation.
A line of work have been done under similar ideas.

### Linear MDP

Specifically, in linear MDP we assume the following conditions hold.

**Definition**(linear MDP):
MDP $\cal M$ is a *linear MDP* with a feature map $\phi:\cal S\times A\mapsto \Bbb R^d$,
if for any $h\in[H]$,
there exists $d$ unknown measures $\mu_h=(\mu_h^{(1)},\ldots,\mu_h^{(d)})$ over $\cal S$
and an unknown vector $\theta_h\in \Bbb R^d$,
such that for any $(s,a)\in\cal S\times A$,
we have

$$
r_h(s,a)=\theta_h^T\phi(s,a)\\\\
\Bbb P_h(\cdot\vert s,a)=\mu_h(\cdot)^T\phi(s,a),
$$
which is to say, by mapping state-action pairs $(s,a)$ to a vector $\phi(s,a)$ in $d$-dimentional feature space, we can represent both rewards and transition probability with some $d$-dimentional parameters.


**Proposition**:
*For a linear MDP,
for any policy $\pi$,
there exists weights $\\{w_h^\pi\\}_{h=1}^H$
s.t. for any $(s,a,h)\in{\cal S\times A}\times [H]$
we have $Q_h^\pi(s,a)=\langle\phi(s,a),w_h\rangle$*

*Proof.*
By Bellman equality
$$
\begin{align*}
Q_h^\pi(s,a)
&=r_h(s,a)+(\Bbb P_hV_{h+1}^\pi)(s,a)\\\\
&=\phi^T(s,a)\theta_h
+\phi^T(s,a)\int V_{h+1}^\pi(s')d\mu(s')\\\\
&=\phi^T(s,a)\left(\theta_h+\int V_{h+1}^\pi(s')d\mu(s')\right)
\end{align*}
$$

Then just set $w_h$ to be the left part,
we show $Q_h^\pi(s,a)$ is linear.
$$\tag*{$\blacksquare$}$$
which completes the proof.

**Algorithm**(LSVI-UCB):

- FOR $k=1,\ldots,K$
  - receive initiala state $s_1$
  - FOR $h=H,\ldots,1$
    - $\Lambda=\sum_{\tau=1}^{k-1}\phi_h^\tau(\phi_h^\tau)^T+\lambda I$ ($\lambda I$ here is some small term to keep $\Lambda$ full rank)
    - $w_h=\Lambda^{-1}\sum_{\tau=1}^{k-1}\phi_h^\tau(r_h^\tau+\max_aQ_{h+1}(s_{h+1}^\tau,a))$
    - $Q_h(\cdot,\cdot)=\max\\{H,w_h^T\phi(\cdot,\cdot)+\beta\sqrt{\phi^T(\cdot,\cdot)\Lambda^{-1}\phi(\cdot,\cdot)}\\}$
  - FOR $h=1,\ldots,H$
    - take action $a_h^k=\arg\max_a Q_h(s_h^k,a)$
    - observe $s_{h+1}^k$ and receive reward $r_h^k$

Let's imagine we are estimating a hyperplane in $\Bbb R^d$.
We know that we can assure such a hyperplane with affinely independent $d$ points.
So what we do here is just exploring more in directions that are more (in some sense) affinely independent from the recorded ones.

One can prove that LSVI-UCB has regret $\sqrt{d^3H^3T}$,
see [[Jin et al. 2019]](https://arxiv.org/pdf/1907.05388).

### Linear realizability

Instead of assuming value functions are all linear,
we can just suppose that $Q_h^\star$, $V_h^\star$ are functions which are in the span of some given features.

**Algorithm**(ELEANOR, from [[Zanette et.al. 2020]](https://arxiv.org/pdf/2003.00153)):

- FOR episode $k=1,\ldots,K$
  - receive initial state $s_1^k$
  - construct confidence set: $\Theta_k=\\{(\bar\theta_1,\ldots,\bar\theta_H)\mid\Vert\bar\theta_h-\hat\theta_h \Vert_{\Lambda_h}^2\le\beta_k\\}$ where $\bar\theta_h$ is weight vector at step $h$
  - $\Lambda_h=\sum_{\tau=1}^{k-1}\phi_h^\tau(\phi_h^\tau)^T+\lambda I$
  - $\hat\theta_h=\Lambda^{-1}\sum_{\tau=1}^{k-1}\phi_h^\tau(r_h^\tau+\max_a\bar\theta_{h+1}^\tau\phi(s_{h+1}^\tau,a))$
  - $(\bar\theta_1,\ldots,\bar\theta_H)=\arg\max_{(\bar\theta_1,\ldots,\bar\theta_H)\in\Theta_k}\max_a\bar\theta_{h+1}^\tau\phi(s_1^k,a))$
  - FOR $h=1,\ldots,H$
    - take action $a_h^k=\arg\max_a\phi^T(s_h^k,a)\bar\theta_h$
    - transit to next step

Can also guarantee a regret polynomial in $H$

- computationally inefficient
- need to specify feature mapping
- sometimes linear is too strong an asssumption

## General Function Approximation

In empirical field, people may prefer deep RL techniques that are non-linear.
To fit such practice in our analysis framework,
we may need consider general function approximation
where we focus on hypothesis class $\cal H$ just like
what we do in supervised learning.

instead of mapping state-actoin pairs
it maps functions in hypothesis class

we may have model-based and value-based hypothesis classes.

### Low Bellman rank

**Definition**(average bellman error with roll-in policy):
$\forall f\in\cal F$ and
any roll-in policy $\pi$,
define

$$
\epsilon_h(f,\pi)
=\Bbb E_{a_{1:h-1}\sim\pi,a_h\sim\pi_{f}}
[f_h(s_h,a_h)-(\cal T_h f_{h+1})(s_h,a_h)]
$$

**Definition**(Bellman rank):
Let $\epsilon_h$ be a matrix in $\Bbb R^{\cal |F|\times|F|}$
where its $(f,f')$ entry is $\epsilon_h(f,\pi_{f'})$.
The *Bellman rank* of MDP $\cal M$ and function class $\cal F$ is defined as
$$
\max_{h\in[H]}rank(\epsilon_h)
$$
equivalently, $\cal M$ has Bellman rank $d$ means
there exists maps $\phi:\cal F\mapsto\Bbb R^d$
and $\psi:\cal F\mapsto\Bbb R^d$ s.t. $\forall f,f'$
$$
\epsilon(f,\pi_{f'})=\langle\phi(f),\psi(f')\rangle
$$

**Algorithm**(OLIVE):

### Low Bilinear rank

**Algorithm**(BLin-UCB):

### Low Bellman-Eluder rank

**Algorihtm**(Golobal Optimism based on Local Fitting, GOLF):

- Initialize ${\cal D}_1,\ldots,{\cal D}_H=\emptyset$,$\cal B^0=F$
- FOR episode $k=1,\ldots,K$
  - Choose policy $\pi^k=\pi_{f^k}$ where $f^k=\arg\max_{f\in\cal B^{k-1}}f(s_1,\pi_f(s_1))$
  - Collect a trajectory $(s_1,a_1,r_1,\ldots,s_H,a_H,r_H,s_{H+1})$ by following $\pi^k$
  - Augment $D_h=D_h\cup\\{s_h,a_h,s_{h+1}\\}$ for $h\in[H]$
  - Update ${\cal B}^k=\\{f\in {\cal F}: L_{\cal D_h}(f_h,f_{h+1})\le\inf_{f\in\cal F_n}L_{\cal D_h}(f,f_{h+1})+\beta,\forall h\in[H]\\}$

Basically speaking, it's a generalized version of ELEANOR algorithm.

- may be computationaly hard

can we say seperate the problem of RL into standalone generalization,planning and exploration tasks?

can we go beyond model-based and value based methods,
just optimize over policy space with a theoretical bound on regret?
