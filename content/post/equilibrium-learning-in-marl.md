---
date : '2025-05-28'
draft : true
title : 'Equilibrium Learning in MARL'
katex : true
tags : ['learing theory','reinforcement learning']
---
## Introduction

## Settings

Normal-form/extensive-form game

**Definition**(markov game):

> Markov Games with m players: (state, m actions) -> (m rewards, next state).

markov potential game

extensive form game

simultaneous/turn-based

## Equilibrium Learning

### Equilibria in Game

Nash equilibrium

>  Nash equilibrium: Each player plays the best response to all other player’s policies.

correlated equilibrium

coarse correlated equilibrium
$$
\max_{i\in[m]}\max_{\pi_i'}(V_{i,1}^{\pi_i',\pi_{-i}}-V_{i,1}^\pi)=0
$$


### Learning Goals

**Definition**:$\epsilon$-approximate
$$
\max_{i\in[m]}\max_{\pi_i'}(V_{i,1}^{\pi_i',\pi_{-i}}-V_{i,1}^\pi)\le\epsilon
$$
external regret
$$
\max_{a^i\in \cal{A}_i}\sum_t[r^i(a^i,a_t^{-i})-r^i(a_t^i,a_t^{-i})]
$$


Swap regret
$$
\max_{\phi\in\cal{A}_i\mapsto\cal{A}_i}\sum_t[r^i(\phi(a_t^i),a_t^{-i})-r^i(a_t^i,a_t^{-i})]
$$


**Proposition**: A no-external/swap regret algorithm could output a $\epsilon$-approximate CCE/CE policy

*Proof*. Let 

$$
\bar{p}_T(a)=\frac{1}{T}\sum_t \Bbb I\{a_t=a\}
$$

Then once we have that

$$
\begin{align}
	\max_{a^i\in \cal{A}_i}\sum_t[r^i(a^i,a_t^{-i})-r^i(a_t^i,a_t^{-i})] & \le o(T)\\
	\max_{a^i\in \cal{A}_i}\sum_t[\frac{1}{T}r^i(a^i,a_t^{-i})-\frac{1}{T}r^i(a_t^i,a_t^{-i})]& \le o(1)
\end{align}
$$

we can say

$$
\max_{\tilde{a}^i\in\cal{A}_i}\Bbb{E}_{a\sim\bar{p}_T}[r^i(\tilde{a}^i,a^{-i})-r^i(a)]\le o(1)
$$

random-iterate

Last-iterate

OMWU & OGDA

FTRL & OMD

### Algorithms in Markov Game

|                                                              | Simulator | Decentralized | NE   | CE                                                   | CCE                                                |
| ------------------------------------------------------------ | :-------- | ------------- | ---- | ---------------------------------------------------- | -------------------------------------------------- |
| [Song et al. 2021](http://arxiv.org/abs/2110.04184)(CE/CCE-V-Learning) | No        | -             | -    | $\tilde{\cal{O}}(H^6S\max_{i\le m}A_i^2/\epsilon^2)$ | $\tilde{\cal{O}}(H^5S\max_{i\le m}A_i/\epsilon^2)$ |
| [Jin et al. 2021c](http://arxiv.org/abs/2110.14555)(V-Learning) | No        | Yes           | -    | $\tilde{\cal{O}}(H^5S\max_{i\le m}A_i^2/\epsilon^2)$ | $\tilde{\cal{O}}(H^5S\max_{i\le m}A_i/\epsilon^2)$ |
| [Li et al. 2022](https://arxiv.org/abs/2208.10458v2)(Q-FTRL) | Yes       | -             | -    | -                                                    | $\tilde{\cal{O}}(H^4S\sum_{i\le m}A_i/\epsilon^2)$ |

Potential Improvements

- improve dependency on $H$ to be optimal
- Can we achieve a last-iterate convergence in Markov game? 

### Algorithms in Markov Game with linear function approximation 

|                              | Simulator | Decentralized | NE   | CE   | CCE                                                         |
| ---------------------------- | :-------- | ------------- | ---- | ---- | ----------------------------------------------------------- |
| [Wang et al., 2023]()(AVLPR) | No        | -             | -    | -    | $\tilde{\cal{O}}(m^2d^4H^{6}\max_{i\le m}A_i^5/\epsilon^2)$ |
| [Dai et al.,2024]()          | No        | -             | -    | -    | $\tilde{\cal{O}}(m^4d^5H^6\log S/\epsilon^2)$               |
| [Fan et al., 2024]()         | Yes       | -             | -    | -    | $\tilde{\cal{O}}(m^2d^2H^6/\epsilon^2)$                     |

Potential Improvements:

- remove undesired dependency on $S$ and $A$
- improve dependency on $H$ and $d$ to be optimal

## Beyond Equilibrium Learning

equal-share
