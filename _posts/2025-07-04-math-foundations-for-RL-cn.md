---
layout: post
title: 强化学习—数学基础
date: 2025-07-04 11:12:00-0800
description: 超大一坨数学公式
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

## MDP（马尔可夫决策过程）

我们通常把一个 MDP 定义为一个元组 $$(\mathcal{S}, \mathcal{A}, p, R, \gamma)$$。

<div class="definition" markdown="1">
**状态集合（$$\mathcal{S}$$）：** 环境所有可能状态的集合。  
* 时刻 $$t$$ 的状态 $$S_t$$ 总是取值于 $$\mathcal{S}$$。
</div>

<div class="definition" markdown="1">
**动作集合（$$\mathcal{A}$$）：** 智能体可以采取的所有可能动作的集合。  
* 时刻 $$t$$ 的动作 $$A_t$$ 总是取值于 $$\mathcal{A}$$。
</div>

<div class="definition" markdown="1">
**转移函数（$$p$$）：** 描述环境状态如何变化。

$$
p: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]
$$

对于所有 $$s \in \mathcal{S}$$、$$a \in \mathcal{A}$$、$$s' \in \mathcal{S}$$ 以及 $$t \in \mathbb{N}_{\ge 0}$$：

$$
p(s,a,s') := \Pr(S_{t+1}=s' \mid S_t=s, A_t=a)
$$

当 $$p(s,a,s') \in \{0,1\}$$ 对所有的 $$s,a,s'$$ 成立时，转移函数是确定性的。
</div>

<div class="definition" markdown="1">
$$d_R$$ 描述奖励的生成方式。

$$
R_t \sim d_R(S_t, A_t, S_{t+1})
$$

</div>

<div class="definition" markdown="1">
**奖励函数（$$R$$）：** 由奖励分布 $$d_R$$ 隐式定义的函数，描述奖励如何生成。

$$
R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}
$$

$$
R(s,a) := \mathrm{E}[R_t \mid S_t = s, A_t = a]
$$

</div>

<div class="definition" markdown="1">
**初始状态分布（$$d_0$$）：**

$$
d_0: \mathcal{S} \rightarrow [0,1]
$$

$$
d_0(s) = \Pr(S_0 = s)
$$

</div>

<div class="definition" markdown="1">
**折扣因子（$$\gamma$$）：** 取值范围 $$[0,1]$$，用于折扣未来奖励。
</div>

---

### 目标

我们的目标是找到一条最优策略 $$\pi^*$$，使得期望累计折扣奖励最大化。

- $$G^i$$ 表示第 $$i$$ 个回合的回报（return）。  
- $$R_t^i$$ 表示第 $$i$$ 个回合时刻 $$t$$ 的奖励。

<div class="definition" markdown="1">
**目标函数（$$J$$）：**

$$
J : \Pi \rightarrow \mathbb{R}, \quad \text{对于所有 }\pi \in \Pi
$$

$$
\begin{aligned}
J(\pi) &:= \mathrm{E}\Bigg[\sum_{t=1}^{\infty} \gamma^t R_t \,\Big|\, \pi\Bigg] \\[0.2cm]
\hat{J}(\pi) &:= \frac{1}{N}\sum_{i=1}^{N} G^i 
            = \frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{\infty}\gamma^t R_t^i
\end{aligned}
$$

</div>

<div class="definition" markdown="1">
**最优策略（$$\pi^*$$）：**

$$
\pi^* \in \arg\max_{\pi \in \Pi} J(\pi)
$$

</div>

<details>
  <summary>当最优策略存在时它总是唯一的吗？</summary>

  不一定，可能存在多条同样优秀的最优策略。
</details>

---

### 性质

<div class="definition" markdown="1">
**时限（Horizon，$$L$$）：** 最小的整数 $$L$$，使得对所有 $$t \ge L$$，处于终止状态 $$s_\infty$$ 的概率为 1。

$$
\forall t \ge L,\; \Pr(S_t = s_\infty) = 1
$$

- 若 $$L < \infty$$（对所有策略均成立），则 MDP 为 **有限时限**（回合式）。  
- 若 $$L = \infty$$，则 MDP 为 **无限时限**（连续式）。
</div>

<div class="definition" markdown="1">
**马尔可夫性（Markov Property）：** 一种关于状态表示的性质，假设在给定当前状态的条件下，未来与过去独立。  
* 给定当前状态 $$S_t$$，$$S_{t+1}$$ 与历史 $$H_{t-1}$$ 条件独立。
</div>

---

## 策略（Policy）

<div class="definition" markdown="1">
**策略** 是一种决策规则——智能体选择动作的方式。

$$
\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]
$$

$$
\pi(s,a) := \Pr(A_t = a \mid S_t = s)
$$

</div>

---

## 价值函数（Value Functions）

<div class="definition" markdown="1">
**状态价值函数（$$v^\pi$$）**

状态价值函数 $$v^\pi : \mathcal{S} \rightarrow \mathbb{R}$$ 衡量从状态 $$s$$ 出发并遵循策略 $$\pi$$ 时的期望回报。

$$
\begin{aligned}
v^\pi(s) &:= \mathbf{E}\Bigg[\sum_{k=1}^{\infty}\gamma^k R_{t+k} \,\Big|\, S_t = s, \pi\Bigg] \\[0.2cm]
         &:= \mathbf{E}[G_t \mid S_t = s, \pi]
\end{aligned}
$$

</div>

<div class="definition" markdown="1">
**动作价值函数（Q 函数，$$q^\pi$$）**

动作价值函数 $$q^\pi : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$ 衡量在状态 $$s$$ 采取动作 $$a$$ 后再遵循策略 $$\pi$$ 所得到的期望回报。

$$
q^\pi(s,a) := \mathbf{E}[G_t \mid S_t = s, A_t = a, \pi]
$$

</div>

---

### Bellman 方程

<div class="definition" markdown="1">
**状态价值函数的 Bellman 方程（$$v^\pi$$）**

$$
\begin{aligned}
v^\pi(s) &= \mathbf{E}\Big[R(s,A_t) + \gamma v^\pi(S_{t+1}) \,\Big|\, S_t = s, \pi\Big] \\[0.3cm]
         &= \sum_{a \in \mathcal{A}} \pi(s,a) 
            \sum_{s' \in \mathcal{S}} p(s,a,s')\big(R(s,a) + \gamma v^\pi(s')\big)
\end{aligned}
$$

- Bellman 方程只需向前看一步。  
- 最优状态价值函数 $$v^*$$ 是唯一的——所有最优策略共享同一 $$v^*$$。
</div>

<div class="definition" markdown="1">
**动作价值函数的 Bellman 方程（$$q^\pi$$）**

$$
q^\pi(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} p(s,a,s') 
             \sum_{a' \in \mathcal{A}} \pi(s',a') q^\pi(s',a')
$$

- 最优动作价值函数 $$q^*$$ 对所有最优策略也是唯一的。
</div>

### Bellman 最优方程

1. 若一条策略 $$\pi$$ 满足 Bellman 最优方程，则 $$\pi$$ 是最优策略。  
2. 若状态、动作集合有限，奖励有界且 $$\gamma < 1$$，那么存在满足 Bellman 最优方程的策略 $$\pi$$。

<div class="definition" markdown="1">
**$$v^*$$ 的 Bellman 最优方程**

对所有状态 $$s \in \mathcal{S}$$：

$$
v^\pi(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} p(s,a,s') \big[R(s,a) + \gamma v^\pi(s')\big]
$$

</div>

<div class="definition" markdown="1">
**$$q^*$$ 的 Bellman 最优方程**

$$
q^*(s,a) = \sum_{s' \in \mathcal{S}} p(s,a,s') 
           \big[R(s,a) + \gamma \max_{a' \in \mathcal{A}} q^*(s',a')\big]
$$

</div>

---

## 策略迭代（Policy Iteration）

策略迭代通过交替执行两步——策略评估与策略改进——来寻找最优策略。

- 通过动态规划进行的策略评估虽然保证收敛到 $$v^\pi$$，但并不保证在有限计算内就能到达。

<div class="definition" markdown="1">
**策略改进定理**

对于任意策略 $$\pi$$，若存在确定性策略 $$\pi'$$ 使得 $$\forall s \in \mathcal{S}$$：

$$
q^\pi(s, \pi'(s)) \ge v^\pi(s)
$$

则有 $$\pi' \ge \pi$$。
</div>

<div class="definition" markdown="1">
**随机策略的策略改进定理**

对于任意策略 $$\pi$$，若 $$\pi'$$ 满足：

$$
\sum_{a \in \mathcal{A}} \pi'(s,a) q^\pi(s,a) \ge v^\pi(s),
$$

则 $$\forall s \in \mathcal{S}$$，都有 $$\pi' \ge \pi$$。
</div>

---

## 价值迭代（Value Iteration）

价值迭代通过迭代应用 Bellman 最优更新来寻找最优状态价值函数。

<div class="definition" markdown="1">
**Banach 不动点定理**

若映射 $$f$$ 在非空完备赋范向量空间上是收缩映射，则存在唯一不动点 $$x^*$$，且以任意 $$x_0$$ 为初始点、按照 $$x_{k+1} = f(x_k)$$ 生成的序列收敛到 $$x^*$$。
</div>

<div class="definition" markdown="1">
**Bellman 算子是收缩映射**

当 $$\gamma < 1$$ 时，在度量 $$d(v,v') := \max_{s \in \mathcal{S}} |v(s)-v'(s)|$$ 下，Bellman 算子在 $$\mathbb{R}^{|\mathcal{S}|}$$ 上是收缩映射。
</div>

- 对于有限状态动作集、有界奖励且 $$\gamma < 1$$ 的 MDP，价值迭代 **收敛** 到唯一的固定点 $$v^\infty$$。  
- 这类 MDP **至少存在** 一条最优策略。

---

## 大数定律（Law of Large Numbers）

<div class="definition" markdown="1">
**辛钦强大数定律（Khintchine's Strong Law of Large Numbers）**

设 $$\{X_i\}_{i=1}^{\infty}$$ 为 **独立同分布（i.i.d.）随机变量**。则样本平均序列 $$(\frac{1}{n} \sum_{i=1}^{n} X_i)_{n=1}^\infty$$ **几乎必然** 收敛到期望 $$\mathbf{E}[X_1]$$。

即 $$\displaystyle \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{a.s.} \mathbf{E}[X_1]$$

</div>

<div class="definition" markdown="1">
**Kolmogorov 强大数定律**

设 $$\{X_i\}_{i=1}^{\infty}$$ 为 **独立（不要求同分布）随机变量**。若所有 $$X_i$$ 具有 **相同均值且方差有界**，则样本平均序列 $$(\frac{1}{n}\sum_{i=1}^{n} X_i)^\infty_{n=1}$$ 亦几乎必然收敛到 $$\mathbf{E}[X_1]$$。
</div>
