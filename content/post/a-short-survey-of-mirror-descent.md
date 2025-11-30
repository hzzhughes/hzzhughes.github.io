---
date : '2025-10-20'
draft : true
title : 'A Short Survey of Online Mirror Descent in Adversarial MAB'
katex : true
tags : ['learing theory','online learning']
---

FTRL, Online Mirror, Descent,
Bandit algorithm

## Stochastic Gradient Descent

guarantee of convergence to optimal point

## Proximal Gradient Descent

definition: proximal operator

for convex (not necessarily smooth) function $f$
we define proximal operator as
$$
    \text{prox}_{f}(x)
:=
    \arg \min_z
    f(z)+
    \frac{1}{2}
    ||x-z||_2^2
$$

algorithm: PGD
for convex unconstraint problem

$$
    \min_x f(x):=f_0(x)+f_1(x)
$$

where $f_0 $ is convex (not necessarily smooth),
$f_1$ is convex and smooth function.

$$
    x^{r+1}=x^r-\alpha^r\tilde{\nabla} f(x^r)
$$

where we define
$\tilde{\nabla}f(x):=x-\text{prox}_{f_0}(x-\nabla f_1 (x))$

**equivalent convergence**.

equivalence of optimality condition.

guarantee of convergence

**easier computation**.

closed-form solution

equivalence of minimizing a quadratic function
e.g. gradient descent

$$
    x^{r+1}=x^r-\alpha^r
    \nabla f(x^r)
=
    \argmin_x
    \lang
        x-x',
        \nabla f(x')
    \rang+
    \frac{1}{2\alpha^r}
    ||x-x^r||^2

$$

We can observe that doing gradient descent is minimizing a quadratic function that is tangent to the original function at current point $x^r$.
Next we'll show that such observation also exists in proximal gradient descent.

$$
    x^{r+1}
=
    x^r-
    \alpha^r
    \tilde{\nabla} f(x^r)
=
    \text{prox}_{\alpha^r f_0}(x^r-\alpha^r\nabla f_1 (x))\\\\
=
    \argmin_x
    f_0(x)+
    \frac{1}{2}
    ||x-x^r-\alpha^r\nabla f_1 (x)||^2\\\\
=
    \argmin_x
    f_0(x)+
    \lang
        x-x^r,
        \nabla f_1(x^r)
    \rang+
    \frac{1}{2\alpha^r}
    ||x-x^r||^2

$$


bregman divergence: 
$$
    B_\eta(x,x')
:=
    \eta(x)-
    \eta(x')-
    \lang
        x-x',
        \nabla\eta(x')
    \rang
$$

It's obvious that $B_\eta(x',x')=0$ and that $\frac{\partial}{\partial x}\mid_{x=x'} B_\eta(x,x')=0$

$$
    x^{r+1}
=
    \argmin_x
    f_0(x)+
    \lang
        x-x^r,
        \nabla f_1(x^r)
    \rang+
    \frac{1}{\alpha^r}
    B_\eta(x,x^r)

$$

## PGD in Online Expert Problem

from online linear optimization to decision making problem

Assume we are minimizing the loss of our policy over $n$ options.
The underlying loss vector for each round is all $l$ in expectation.

online expert problem

$l\in[0,1]^n$

$l^r\in[0,1]^n,\Bbb{E}[l^r]=l$

$\theta\in\Delta_{[n]}$

$\lang\theta,l^r\rang$

$$
    \min_\theta
    \Bbb{E}
    [\lang\theta,l^r\rang]\\\\
    s.t.\quad
    \theta_i\ge 0,\forall i \in [n]\\\\
    \sum_{i\in [n]} \theta_i=1
$$

objective: $\epsilon$-optimal policy,
sublinear regret

simplex constraint
$f_0=I_{\chi},\chi=\Delta_{[n]}$

linear objective 
$f_1=\lang\theta,l\rang$

choosing eta to be entropy
$\eta(x)=\sum_i x_i\log x_i$

we'll then have a closed-form update
$$
\theta_i^{r+1}\propto\theta_i^{r}\exp(\alpha l_i^r),\forall i \in [n]
$$

convergence to optimal policy

equivalence between PGD/OMD/FTRL

stronger notion of convergence: regret

## Restricted Information Feedback: Multi-Armed Bandits

importance sampling estimator

## One More Application Example: Learning Equilibrium in Games

matrix game

nash equilibrium

independently run OMD

e.g. routing game