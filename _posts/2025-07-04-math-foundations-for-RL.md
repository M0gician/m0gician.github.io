---
layout: post
title: Reinforcement Learning—Mathematical Foundations
date: 2025-07-04 11:12:00-0800
description: A very, very, very large chunk of math
tags: RL math
categories: reinforcement-learning
related_posts: false
---

<style>
.definition {
  background-color: #e7f3fe;
  border-left: 6px solid #2196F3;
  padding: 10px;
  margin-bottom: 15px;
}
</style>

## MDP (Markov Decision Process)

We usually define an MDP as a tuple $$(\mathcal{S}, \mathcal{A}, p, R, \gamma)$$.

<div class="definition" markdown="1">
**State Set ($$\mathcal{S}$$):** The set of all possible states of the environment.
* The state at time $$t$$, $$S_t$$, always takes values in $$\mathcal{S}$$.
</div>

<div class="definition" markdown="1">
**Action Set ($$\mathcal{A}$$):** The set of all possible actions the agent can take.
* The action at time $$t$$, $$A_t$$, always takes values in $$\mathcal{A}$$.
</div>

<div class="definition" markdown="1">
**Transition Function ($$p$$):** Describes how the state of the environment changes.

$$
p: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]
$$

For all $$s \in \mathcal{S}$$, $$a \in \mathcal{A}$$, $$s’ \in \mathcal{S}$$, and $$t \in \mathbb{N}_{\geq 0}$$:

$$
p(s,a,s') := \text{Pr}(S_{t+1}=s' | S_t=s, A_t=a)
$$

A transition function is deterministic if $$p(s,a,s’) \in \{0,1\}$$ for all s, a, and s’
</div>

<div class="definition" markdown="1">
$$d_R$$ describes how rewards are generated.

$$
R_t \sim d_r(S_t, A_t, S_{t+1})
$$

</div>

<div class="definition" markdown="1">
**Reward Function ($$R$$):** A function implicitly defined by the reward distribution $$d_R$$, which describes how rewards are generated.

$$
R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}
$$

$$
R(s,a) := \mathrm{R}[R_t|S_t = s, A_t = a]
$$

</div>

<div class="definition" markdown="1">
**Initial State Distribution ($$d_0$$):**

$$
d_0: \mathcal{S} \rightarrow [0,1]
$$

$$
d_0(s) = \text{Pr}(S_0=s)
$$

</div>

<div class="definition" markdown="1">
**Discount Factor ($$\gamma$$):** A parameter in $$[0,1]$$ that discounts future rewards.
</div>

---

### Objective

The goal is to find an optimal policy $$\pi^*$$ that maximizes the expected total amount of discounted reward.

-   $$G^i$$ denotes the return of the i-th episode.
-   $$R^i_t$$ denotes the reward at time $$t$$ during episode $$i$$.

<div class="definition" markdown="1">
**Objective function ($$J$$):**

$$
J : \Pi \rightarrow \mathbb{R}, \text{where for all } \pi \in \Pi
$$

$$
\begin{aligned}
&J(\pi) := \mathrm{E}\left[\sum_{t=1}^{\infty} \gamma^tR_t \bigg| \pi\right] \\
&\hat{J}(\pi) := \frac{1}{N}\sum_{i=1}^{N}G^i = \frac{1}{n}\sum_{i=1}^{N}\sum_{t=0}^{\infty}\gamma^t R_t^i
\end{aligned}
$$

</div>

<div class="definition" markdown="1">
**Optimal Policy ($$\pi^*$$):**

$$
\pi^* \in \underset{\pi \in \Pi}{\text{argmax}}\,J(\pi)
$$

</div>

<details>
  <summary>Is the optimal policy always unique when it exists?</summary>

  No. There can exist multiple optimal policies that are equally good.

</details>

---

### Properties

<div class="definition" markdown="1">
**Horizon ($$L$$):** The smallest integer $$L$$ such that for all $$t \geq L$$, the probability of being in a terminal state $$s_\infty$$ is 1.

$$
\forall t \geq L, \text{Pr}(S_t = s_\infty) = 1
$$

- The MDP is **finite horizon** (episodic) if $$L < \infty$$ for all policies.
- The MDP is **infinite horizon** (continuous) when $$L = \infty$$.
</div>

<div class="definition" markdown="1">
**Markov Property:** A property of the state representation. It assumes that the future is independent of the past given the present.
* $$S_{t+1}$$ is conditionally independent of the history $$H_{t-1}$$ given the current state $$S_t$$.
</div>

---

## Policy

<div class="definition" markdown="1">
A **policy** is a decision rule—a way that the agent can select actions.

$$
\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]
$$

$$
\pi(s,a) := \text{Pr}(A_t=a | S_t=s)
$$

</div>

---

## Value Functions

<div class="definition" markdown="1">
**State-Value Function ($$v^\pi$$)**

The state-value function $$v^\pi : \mathcal{S} \rightarrow \mathbb{R}$$ measures the expected return starting from a state $$s$$ and following policy $$\pi$$.

$$
\begin{aligned}
v^\pi(s) &:= \mathbf{E}\left[\sum_{k=1}^{\infty}\gamma^k R_{t+k} \bigg| S_t=s, \pi\right] \\
&:= \mathbf{E}[G_t|S_t=s, \pi]
\end{aligned}
$$

</div>

<div class="definition" markdown="1">
**Action-Value Function (Q-function, $$q^\pi$$)**

The action-value function $$q^\pi : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$ measures the expected return starting from state $$s$$, taking action $$a$$, and then following policy $$\pi$$.

$$
q^\pi(s,a) := \mathbf{E}[G_t | S_t=s, A_t=a, \pi]
$$

</div>

---

### Bellman Equations

<div class="definition" markdown="1">
**Bellman Equation for the State-Value Function ($$v^\pi$$)**

$$
\begin{aligned}
v^\pi(s) &= \mathbf{E}\left[\underbrace{R(s,A_t)}_{\text{immediate reward}} + \gamma \underbrace{v^\pi(S_{t+1}) }_{\text{value of next state}} \bigg| S_t = s, \pi\right] \\[0.3cm]
&= \sum_{a\in\mathcal{A}}\pi(s,a)\sum_{s' \in \mathcal{S}}p(s,a,s')(R(s,a) + \gamma v^\pi(s'))
\end{aligned}
$$

- The Bellman equation only needs to look forward one time step into the future.
- The optimal state-value function, $$v^*$$, is unique—all optimal policies share the same state-value function.
</div>

<div class="definition" markdown="1">
**Bellman Equation for the Action-Value Function ($$q^\pi$$)**

$$
q^\pi(s,a) = R(s,a) + \gamma\sum_{s' \in \mathcal{S}}p(s,a,s')\sum_{a' \in \mathcal{A}}\pi(s',a')q^\pi(s',a')
$$

- The optimal action-value function, $$q^*$$, is also unique among all optimal policies.
</div>

### Bellman Optimality Equations

1.  If a policy $$\pi$$ satisfies the Bellman optimality equation, then $$\pi$$ is an optimal policy.
2.  If the state and action sets are finite, rewards are bounded, and $$\gamma < 1$$, then there exists a policy $$\pi$$ that satisfies the Bellman optimality equation.

<div class="definition" markdown="1">
**Bellman Optimality Equation for $$v^*$$**

A policy $$\pi$$ satisfies the Bellman optimality equation if for all states $$s\in\mathcal{S}$$:

$$
v^\pi(s) = \max_{a\in\mathcal{A}}\sum_{s'\in\mathcal{S}}p(s,a,s')[R(s,a)+\gamma v^\pi(s')]
$$

</div>

<div class="definition" markdown="1">
**Bellman Optimality Equation for $$q^*$$**

$$
q^*(s,a) = \sum_{s'\in\mathcal{S}}p(s,a,s')\left[R(s,a) + \gamma\max_{a'\in\mathcal{A}}q^*(s',a')\right]
$$

</div>

---

## Policy Iteration

Policy iteration is an algorithm that finds an optimal policy by alternating between two steps: policy evaluation and policy improvement.

-   Even though policy evaluation using dynamic programming is guaranteed to converge to $$v^\pi$$, it is not guaranteed to reach $$v^\pi$$ in a finite amount of computation.

<div class="definition" markdown="1">
**Policy Improvement Theorem**

For any policy $$\pi$$, if $$\pi’$$ is a deterministic policy such that $$\forall s \in \mathcal{S}$$:

$$
q^\pi(s, \pi'(s)) \geq v^\pi(s)
$$

then $$\pi’ \geq \pi$$.
</div>

<div class="definition" markdown="1">
**Policy Improvement Theorem for Stochastic Policies**

For any policy $$\pi$$, if $$\pi’$$ satisfies:

$$
\sum_{a\in\mathcal{A}}\pi'(s,a) q^\pi(s,a) \geq v^\pi(s),
$$

for all $$s \in \mathcal{S}$$, then $$\pi' \geq \pi$$.
</div>

---

## Value Iteration

Value iteration is an algorithm that finds the optimal state-value function by iteratively applying the Bellman optimality update.

<div class="definition" markdown="1">
**Banach Fixed-Point Theorem**

If $$f$$ is a contraction mapping on a non-empty complete normed vector space, then $$f$$ has a unique fixed point, $$x^*$$, and the sequence defined by $$x_{k+1} = f(x_k)$$, with $$x_0$$ chosen arbitrarily, converges to $$x^*$$.
</div>

<div class="definition" markdown="1">
**Bellman Operator is a Contraction Mapping**

The Bellman operator is a contraction mapping on $$\mathbb{R}^{\vert\mathcal{S}\vert}$$ with distance metric $$d(v,v’) := \max_{s\in\mathcal{S}}\vert v(s)-v’(s) \vert$$ if $$\gamma < 1$$. 
</div>

-   Value iteration **converges** to a unique fixed point $$v^\infty$$ for all MDPs with finite state and action sets, bounded rewards, and $$\gamma < 1$$.
-   All MDPs with finite state and action sets, bounded rewards, and $$\gamma < 1$$ **have at least one optimal policy**.

---

## Law of Large Numbers

<div class="definition" markdown="1">
**Khintchine's Strong Law of Large Numbers**

Let $$\{X_i\}_{i=1}^{\infty}$$ be **independent and identically distributed (i.i.d.) random variables**. Then the sequence of sample averages $$(\frac{1}{n} \sum_{i=1}^{n} X_i)_{n=1}^\infty$$ converges **almost surely** to the expected value $$\mathbf{E}[X_1]$$.

i.e., $$\displaystyle \frac{1}{n}\sum_{i=1}^{\infty} X_i \overset{a.s.}{\rightarrow}\mathbf{E}[X_1]$$

</div>

<div class="definition" markdown="1">
**Kolmogorov's Strong Law of Large Numbers**

Let $$\{X_i\}^\infty_{i=1}$$ be **independent (not necessarily identically distributed) random variables**. If all $$X_i$$ have the **same mean and bounded variance**, then the sequence of sample averages $$(\frac{1}{n}\sum_{i=1}^n X_i)^\infty_{n=1}$$ converges almost surely to $$\mathbf{E}[X_1]$$.
</div>
