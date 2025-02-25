---
date : '2024-12-10'
draft : false
title : 'Settings in Theoretical Machine Learning'
katex : true
tags : ['learing theory']
---
> This is a personal learing note on some of the baisc settings I have encountered so far in a learning theory course. If there is any typo or other kind of mistake, I'd be happy if you could let me know.

One of the main idea behind learing theory is to figure out why machine learning algorithms work from a statistical or mathematical perspective. This article talks about the basic settings used in learning theory.

## Supervised Learing

From the very beginning, we formalize our setting with supervised learning.

In supervised learning, we have input space $\cal X$, output space $\cal Y$ and a training set of size n $(x_1,y_1),\ldots,(x_n,y_n)\in \cal X\times Y$ that are drawn ***[iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)*** from a certain distribution $\cal P$. Our goal is to find a predictor $\hat y:\cal X\mapsto Y\in Y^X$ that can predict precisely. In practice, to measure how precise a predictor is (or how well does it perform), we use a loss function $l:\cal Y^X\times(X\times Y)\mapsto \Bbb R$ and compute the average loss on samples that is indepent from the training set. This is a common technique to quantify our goal, but in theory we use expected loss on population instead of averaged loss on a certain test.

Therefore, we formalize our objective as the expected loss in population and call it ***risk***.

**Definition**(risk):
$$
L(\hat y):=\Bbb E_{z\sim\cal{P}}[l(\hat y,z)]
$$

Note that here we use the notation $z$ instead of $(x,y)$ to represent data points. Also, it is worth to mention that $L(\hat y)$ is still a random variable as $\hat y$ depends on algorithm $\cal A$ (could be randomized algorithm) and training dataset $z_{1:n}$ (set of random variable drawn from distribution $\cal P$).

Next, we introduce reference class $\cal F\sube Y^X$ that is normally the space that we choose our predictor from. E.g. in OLS, we have reference class $\cal F=\\\{f:f(x)=w^{\it T}x+b\vert w\in \Bbb R^m,b\in \Bbb R\\\}$ which is the set of all linear functions.

However, there are also cases in which algorithms can derive predictor that does not belong to $\cal F$. In such cases, we say these algorithms are ***improper***. And inversely, we say an algorithm is ***proper*** if its output space is within $\cal F$.

This may cause confusion to some readers since we usually use the concept of parameter space instead of reference class. One reason we use it here is that we want to be more generalized. As parameter space does not reflect how the prectors are and we want to fit our framework to different learning problems(such as supervised learing, unsupervised learing and density estimation).

Another reason is that we want to be ***agnostic***. In most cases discussed in this article, we do not make much assumptions on distribution $\cal P$. Instead, we incorporate our prior knowledge into the choice of reference class $\cal F$.

Although our ultimate goal is to minize risk as we described above, the main subject of these article is actually just part of it. Since what we are interested in is how could machine learning algorithms derive a good predictor given a certain reference class(we care less about how they are choosed), we care more about ***generalization error***.

**Definition**(excess risk):
$$
\Bbb E[L(\hat y)] - \inf_{f\in \cal F} L(f)
$$

Other parts of the risk is known as ***representation error*** and ***optimization error***. Representation error depends on how representative a reference class is and is naturally identified as $\inf_{f\in \cal F} L(f)$. Optimization error is introduced when we talks about optimization processes in algorithms.

We say there exists at least one policy(or say algorithm) $\pi$ that its excess risk is at least $\cal V^{iid}(F,n)$ for any $\cal P$.

**Definition**(the value of the game):
$$
\cal V^{iid}(F,n):=\inf_{\pi}\sup_{P}\left(\Bbb E[\it L(\hat y)] - \inf_{f\in \cal F} L(f)\right)
$$
So the value of the game actually measures what an one-fits-all algorithm can do: it will do as worst as this value in adversrial environment but no worse than this value in any other environment.

<!-- ### PAC Setting

$$
\cal R
$$

## Generalized Version -->

## Online Learning

### Online Learning with Partial Information
