+++
date = '2024-11-30T14:52:24-08:00'
draft = true
title = 'My First Post'
katex = true
+++
## Introduction

This is **bold** text, and this is *emphasized* text.

Visit the [Hugo](https://gohugo.io) website!

# Quiz 03

## Q1

compute the 9.5 years work perience people's annual salary

(get 4.7 out of 5.0)

## Q2

form trimmed least square problem into MIP Problem

One possible version:
$$
\min_{z,x} \sum_i(1-z_i)(y_i-a^T x)^2\\\
s.t.\sum_i z_i =k\\\
z_i\in\{0,1\}
$$

Better version(Gurobi):

$$
\min_{z,x,w} \sum_i(y_i-a^T x+w_i)^2\\\
s.t.\sum_i z_i=k\\\
-Mz_i \leq w_i \leq Mz_i\\\
W\in \Bbb R,z\in \{0,1\}^n
$$

\\[\int_a^b f(x)\\]

```Python
import torch
import numpy as np
```