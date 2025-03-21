---
date : '2025-03-08'
draft : true
title : 'Introducing Bonus into Adversarial Online Learning'
katex : true
tags : ['learing theory','reinforcement learning']
---

As mentioned in [last post](/post/learning-in-adversarial-linear-mdp/),
due to the fact that we no longer enjoy the effect of variance cancellation,
we cannot directly apply MAB algorithm to MDP problem with performance difference lemma.
But some past research([[Luo et.al.]](https://arxiv.org/pdf/2107.08346)) shows that by introducing some well designed bonus term,
we can not only bound the variance of our natural estimator,
but also guarantee that the bias term introduced along with bonus will be bounded.

Actually, from my personal perspective,
the reason why introducing a bonus term can help is that we are still handling stochastic transition (though we face adversarial loss) which allows and also drives us to explore more like in the stochastic MAB setting.

**Algorithm**:

- FOR $k=1,\ldots,K$
  - receive initiala state $x_1$
  - FOR $h=1,\ldots,H$
    - $\pi_{t,h}(a\mid x)\propto \exp(-\eta\sum_{\tau=1}^{k-1}\hat Q_{\tau,h}(x,a)-B_{\tau,h}(x,a))$
    <!-- - $\pi_{t,h}=\gamma\beta_{t,h}+(1-\gamma)\tilde\pi_{t,h}$, where $\beta_{t,h}\in\Delta(\cal A_h)$ is some exploration distribution -->
    - take action $a$ following mixed strategy $\pi_{t,h}$
    - observe $x_{h+1}^k$ and receive loss $l_{t,h}$
  - FOR $h=H,\ldots,1$
    - $\Lambda=\lambda I+\sum_{a\in\cal A_h}\pi_{t,h}(a\mid x_{t,h})\phi(x_{t,h},a)\phi(x_{t,h},a)^T$
    - $\hat w_{t,h}=\Lambda^{-1}\phi_{t,h}(l_{t,h}+\hat V_{h+1}^k(x_{h+1}^k))$
    <!-- - $\hat w_{t,h}=\Lambda^{-1}\phi_{t,h}(\sum_{h'=h}^Hl_{h'}^k)$ -->
    - $\hat Q_{t,h}(\cdot,\cdot)=\hat w_{t,h}^T\phi(\cdot,\cdot)$
    - $\hat V_{t,h}(\cdot)=\sum_{a\in\cal A}\pi_{t,h}(a)\hat Q_{t,h}(\cdot,a)$
    - $b_{t,h}(x)=\langle\pi_t(\cdot\mid x),\beta\Vert\phi(x,\cdot)\Vert_{(\Lambda)^{-1}}\rangle$
    - $B_{t,h}(x,a)=b_{t,h}(x)+\phi(x,a)\Lambda^{-1}\phi_{t,h}\sum_{a'\in\cal A}\pi_{t,h+1}(a'\mid x_{t,h+1})B_{t,h+1}(x_{t,h+1},a')$

First,
following the proving process from [[Luo et.al.]](https://arxiv.org/pdf/2107.08346),
we show that if we can design a bonus term that satisfies the following conditions,
which is,

$$
\sum_tV^{\pi_t}(x_0;l_t-b_t)-V^{\pi^\ast}(x_0;l_t-b_t)
\lesssim o(T)+V^{\pi^\ast}(x_0;b_t)
$$

then it becomes easy to bound our regret term,
since by simple rearrangement we can change the above inequality into

$$
\sum_tV^{\pi_t}(x_0;l_t)-V^{\pi^\ast}(x_0;l_t)
\lesssim o(T)+V^{\pi_t}(x_0;b_t)
$$
which is easily bounded.

Therefore, to prove the theoretical bound of our design of algorithm,
we firstly need to prove the first inequality we mentioned above holds with high probability.

Again, following the proving process from [[Luo et.al.]](https://arxiv.org/pdf/2107.08346),
we decompose the LHF of the first inequality into three terms

$$
\begin{align*}
    \sum_tV^{\pi_t}(x_0;l_t-b_t)-V^{\pi^\ast}(x_0;l_t-b_t)
    =&\sum_xq^\ast(x)\sum_t\langle (\pi_t-\pi^\ast)(\cdot\mid x),Q^{\pi_t}(x,\cdot;l_t-b_t)\rangle\\\\
    =&\sum_xq^\ast(x)\sum_t\langle (\pi_t-\pi^\ast)(\cdot\mid x),(Q_t-B_t)(x,\cdot)\rangle\\\\
    =&\sum_xq^\ast(x)\sum_t\langle \pi_t(\cdot\mid x),(Q_t-\hat Q_t)(x,\cdot)\rangle
    &\text{bias-1}\\\\
    &+\sum_xq^\ast(x)\sum_t\langle \pi^\ast(\cdot\mid x),(\hat Q_t-Q_t)(x,\cdot)\rangle
    &\text{bias-2}\\\\
    &+\sum_xq^\ast(x)\sum_t\langle (\pi_t-\pi^\ast)(\cdot\mid x),(\hat Q_t-B_t)(x,\cdot)\rangle
    &\text{reg-term}\\\\
\end{align*}
$$

To bound $\text{bias-1}$ term,
we follow the proof from [[Jin et.al.]](http://arxiv.org/abs/1907.05388).

We know
$$
\begin{align*}
    w_{t,h}-\hat w_{t,h}
    =&w_{t,h}-(\Lambda)^{-1}\phi_{t,h}[l_{t,h}+\hat V_{t,h+1}(x_{t,h+1})]\\\\
    =&(\Lambda)^{-1}\left(\lambda w_{t,h}+\langle\pi_t(\cdot\mid x_{t,h}),[\phi(l_{t,h}+\Bbb P_hV_{h+1})] (x_{t,h},\cdot)\rangle- \phi_{t,h}[l_{t,h}+\hat V_{t,h+1}(x_{t,h+1})]\right)
    \\\\
    =&\lambda(\Lambda)^{-1}w_{t,h}
    &q_1\\\\
    &-(\Lambda)^{-1}(\phi_{t,h}l_{t,h}-\langle\pi(\cdot\mid x_{t,h}),(\phi l_{t,h})(x_{t,h},\cdot)\rangle)
    &q_2\\\\
    &-(\Lambda)^{-1}\langle\pi(\cdot\mid x_{t,h}),\phi_{t,h}\hat V_{t,h+1}(x_{t,h+1})-(\phi\Bbb P_h\hat V_{t,h+1})(x_{t,h},\cdot)\rangle
    &q_3\\\\
    &+(\Lambda)^{-1}\langle\pi(\cdot\mid x_{t,h}),[\phi\Bbb P_h(V_{t,h+1}-\hat V_{t,h+1})] (x_{t,h},\cdot)\rangle
    &q_4\\\\
\end{align*}
$$
Then we bound $q1,q2,q3$ term respectively.

For $q_1$,
$$
\vert\phi(x,a)^Tq_1\vert
=\vert\lambda\phi(x,a)^T(\Lambda)^{-1}w_{t,h}\vert
\le\sqrt{\lambda}\Vert w_{t,h}\Vert\Vert\phi(s,a)\Vert_{(\Lambda)^{-1}}
$$

The last step is by weighted Cauchy-Schwarz and the fact that $\Lambda\succeq\lambda I$.

For $q_2$, using similar technique, we have
$$
\vert\phi(x,a)^Tq_2\vert
=\vert\phi(x,a)^T(\Lambda)^{-1}(\phi_{t,h}l_{t,h}-\langle\pi(\cdot\mid x_{t,h}),(\phi l_{t,h})(x_{t,h},\cdot)\rangle)\vert
\le\sqrt{\frac{4}{\lambda}}\Vert\phi(s,a)\Vert_{(\Lambda)^{-1}}
$$

For $q_3$, we have
$$
\vert\phi(x,a)^Tq_3\vert
=\vert\phi(x,a)^T(\Lambda)^{-1}\langle\pi(\cdot\mid x_{t,h}),\phi_{t,h}\hat V_{t,h+1}(x_{t,h+1})-(\phi\Bbb P_h\hat V_{t,h+1})(x_{t,h},\cdot)\rangle\vert
\le H\sqrt{\frac{4}{\lambda}}\Vert\phi(s,a)\Vert_{(\Lambda)^{-1}}
$$

For $q_4$,

$$
\begin{align*}
    \phi(x,a)^T q_4
    =&
    \phi(x,a)^T
    (\Lambda)^{-1}
    \langle
        \pi(\cdot\mid x_{t,h}),
        [
            \phi
            \Bbb P_h(
                V_{t,h+1}
                -\hat V_{t,h+1}
            )
        ]
        (x_{t,h},\cdot)
    \rangle
    \\\\
    =&
    \phi(x,a)^T
    (\Lambda)^{-1}
    \langle
        \pi(\cdot\mid x_{t,h}),
        (\phi\phi^T)(x_{t,h},\cdot)
    \rangle
    \int(V_{t,h+1}-\hat V_{t,h+1})(x_{h+1})d\mu(x_{h+1})
    \\\\
    =&
    \Bbb P_h(V_{t,h+1}-\hat V_{t,h+1})(x,a)
    &p_1\\\\
    &-\lambda\phi(x,a)^T(\Lambda)^{-1}\int(V_{t,h+1}-\hat V_{t,h+1})(x_{h+1})d\mu(x_{h+1})
    &p_2
\end{align*}
$$
where $\vert p_2\vert\le\sqrt{\lambda}\Vert\int(V_{t,h+1}-\hat V_{t,h+1})(x_{h+1})d\mu(x_{h+1})\Vert\Vert\phi(s,a)\Vert_{(\Lambda)^{-1}}$

To bound $\Vert\int(V_{t,h+1}-\hat V_{t,h+1})(x_{h+1})d\mu(x_{h+1})\Vert$,
we need to ensure that $\hat V_{t,h}\le H$.
Actually, this is true if we choose $\lambda\ge 1$ since
$$
\begin{align*}
    \hat Q_{t,h}
    =&\phi(\Lambda)^{-1}\phi_{t,h}[l_{t,h}+\hat V_{t,h+1}(x_{t,h+1})]\\\\
    \le&\frac{1}{\lambda}[l_{t,h}+\hat V_{t,h+1}(x_{t,h+1})]\\\\
\end{align*}
$$
and we know $\hat Q_{t,H}=\phi(\Lambda)^{-1}\phi_{t,H}l_{t,H}\le\frac{1}{\lambda}$,
and by induction we can easily prove that for any $(x,a)\in \cal X\times A_h$,
we have $Q_{t,h}(x,a)\le H-h+1\le H$ if we choose $\lambda\ge 1$

Then we have $\Vert\int(V_{t,h+1}-\hat V_{t,h+1})(x_{h+1})d\mu(x_{h+1})\Vert\le2H\sqrt{d\lambda}$

Combining upper bounds we obtained on $p1,p2,p3,p4$,
we can prove that
for any $(x,a)\in \cal S_h\times A_h$
$$
\mid\phi(x,a)^T\hat w_{t,h}-Q_t(x,a)-\Bbb P_h(V_{t,h+1}-\hat V_{t,h+1})(x,a)\mid
\le\beta\Vert\phi(x,a)\Vert_{(\Lambda)^{-1}}
$$
holds with probability,
where $\beta=cH\sqrt{d}$

Then we turn to the $\text{bias-2}$,

Then we turn to the $\text{reg-term}$,

By Hedge lemma, we have
$$
\sum_t\langle
        (\pi_t-\pi^\ast)(\cdot\mid x),
        (\hat Q_t-B_t)(x,\cdot)
    \rangle
\le\frac{\ln A}{\eta}
+\eta\sum_t\langle
    \pi_t(\cdot\mid x),
    (\hat Q_t-B_t)(x,\cdot)^2
\rangle
$$

if we can guarantee that $\vert\eta(\hat Q_t-B_t)(x,a)\vert\le1$.

Further, by choosing some proper $\eta$,
we can bound RHS with some $O(\sqrt{H^2T})$ term which can sum up to $O(\sqrt{H^4T})$ over step $h\in[H]$.

Finally, combining upper bounds we obtained on $\text{bias-1, bias-2, reg-term}$,
we have
$$
\begin{align*}
    &\text{bias-1 + bias-2 + reg-term}\\\\
    &\le \tilde{O}(\frac{H}{\eta})
    +\sum_t\sum_{x,a}q^\ast(x)\pi_t(a\mid x)
    \beta\Vert\phi(x,a)\Vert_{(\Lambda)^{-1}}\\\\
    &=\tilde{O}(\frac{H}{\eta})
    +\sum_t\sum_{x,a}q^\ast(x)\pi^\ast(a\mid x)b_t(x)
\end{align*}
$$

Then as we discussed in the beginning of this blog,
we have

$$
\text{Reg}
\le\tilde{O}(\frac{H}{\eta})
+V^{\pi_t}(x_0;b_t)
+\sum_xq^\ast(x)\sum_t\langle (\pi_t-\pi^\ast)(\cdot\mid x),(\hat B_t-B_t)(x,\cdot)\rangle
$$

since the last term in RHS is also an underestimator,
we can bound it with constant once we take expectation over both side of above inequality.

$$
\begin{align*}
    \Bbb E_{x\sim q_h^\ast}[
        \langle
            (\pi_t-\pi^\ast)(\cdot\mid x),
            Q_{t,h}(x,\cdot)
        \rangle
    ]
    =&\Bbb E_{x\sim q_h^\ast}[
        \langle
            \pi_t(\cdot\mid x),
            Q_{t,h}(x,\cdot)
        \rangle
    ]\\\\
    &-\Bbb E_{x\sim q_h^\ast}[
        \langle
            \pi^\ast(\cdot\mid x),
            r_{t,h}(x,\cdot)
        \rangle
    ]\\\\
    &-\Bbb E_{x\sim q_h^\ast}[
        \langle
            \pi^\ast(\cdot\mid x),
            \Bbb P^{\pi_t}Q_{t,h+1}(x,\cdot)
        \rangle
    ]\\\\
    =&\Bbb E_{x\sim q_h^\ast}[
        \langle
            \pi_t(\cdot\mid x),
            Q_{t,h}(x,\cdot)
        \rangle
    ]
    -\Bbb E_{x\sim q_{h+1}^\ast}[
        \langle
            \pi_t(\cdot\mid x),
            Q_{t,h+1}(x,\cdot)
        \rangle
    ]\\\\
    &-\Bbb E_{x\sim q_h^\ast}[
        \langle
            \pi^\ast(\cdot\mid x),
            r_{t,h}(x,\cdot)
        \rangle
    ]\\\\
\end{align*}
$$
