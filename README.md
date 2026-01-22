# Projects for the Multi-Agent Systems Course

## AI Teacher using Reinforcement Learning

**Setting**: Reinforcement Learning (RL) is increasingly used to adaptively guide learning processes in educational technology. Imagine an AI teacher tasked with helping a student master two mathematical skills: arithmetic and algebra. At each step, the teacher must choose one of three assignment types — arithmetic-only, algebra-only, or mixed — based on the student’s performance history. The goal is to learn a teaching policy that accelerates the student’s mastery of both skills. This setup can be modeled as a Markov Decision Process (MDP) or extended to a multi-armed bandit framework with evolving reward distributions.

**Objectives**:

- Formulate the AI teaching task as an RL problem where the teacher is an agent interacting with a student model.
- Implement a simulation environment where the student’s skill level evolves probabilistically based on the type of assignments received.
- Use standard RL algorithms (e.g., Q-learning, policy gradients) to learn a policy that minimizes the expected time until the student reaches 90% mastery in both skills.
- Compare the performance of different strategies (e.g., greedy, epsilon-greedy, RL-based) in terms of learning efficiency.
- Discuss implications for adaptive education systems and potential ethical considerations of automated teaching.

## Automated Mechanism Design for Bartering

**Setting**: One-on-one Bartering

Two agents have an initial set of 3 goods each. They also have a valuation for every subset of the whole set of 6 goods they have together. It is possible that both agents can get better off by trading some goods. Suppose, however, that the agents cannot make any form of payment. All they can do is trade goods. This is known as bartering. Additionally, suppose that one agent (agent 1) can dictate the rules of the bartering process. Agent 1 can credibly say to agent 2: “we will barter by my rules, or not at all”. This makes agent 1 the mechanism designer and agent 2 the only player. The set of outcomes O is the set of all allocations of the goods (the partitioning of the 6 goods, of which there are 26 = |O|). Agent 2 is to report (possibly lying) his preferences over the goods (a value in {1, 2, 3, 4, 5} for each item). Based on the report an outcome is chosen. Then, the valuation u : Θ × O → Z that agent 2 has of each allocation is the sum of the corresponding values, minus the valuation of what he owned to begin with. So, the mechanism is direct, where the preferences are both the type and the possible actions of agent 2. The outcome function, which is selected by agent 1 in advance, must be truthful so
that agent 2 has no incentive to lie. Also, the outcome function must ensure that agent 2 does not incur a loss as a result of participating in the mechanism (known as individual rationality constraint), otherwise he would not trade at all. Under these constraints, agent 1 wants to maximise her own valuation g : O → Z (defined in the same manner as agent 2, but based on her own preferences, assumed to be fixed) of her resulting allocation of goods under the mechanism.

**Objectives**:

Implement the branch and bound depth-first search algorithm for automated mechanism design introduced by [CS04] and reported below.
```
SEARCH1(X , Y , w, d):
  if d > |O|:
    CB = X
    L = w
  else :
    if v(X ∪ {od }, Y ) > L:
      SEARCH1(X ∪ {od }, Y , v(X ∪ {od}, Y ), d + 1)
    if ∀ θ ∈ Θ. ∃ o ∈ O ∖ (Y ∪ {od }). u(θ, o) ≥ 0 and v(X, Y ∪ od) > L :
      SEARCH1(X , Y ∪ od , v(X, Y ∪ od ), d + 1)
BnB-DFS():
  CB = None
  L = -∞
  SEARCH1(∅, ∅, 0, 1)
  return CB
```
As in the paper, the definition of function v on disjoint subsets of outcomes X, Y ⊆ O is
```
v(X,Y) = sum[θ∈Θ] p(θ) · max {g(o) | o ∈ O ∖ Y ∧ ∀x ∈ X. u(θ,o) ≥ u(θ,x)}
```
where p is the probability distribution over types Θ, so p(θ) = 1/|Θ| if uniform. Also, notice that outcomes are indexed by d, so that od is the d-th outcome, for 1 ≤ d ≤ |O|.

---

Apply the algorithm to the above setting to find a truthful individually rational deterministic mechanism without payments. To do this, fix beforehand 50 = |Θ| different possible types Θ for agent 2 uniformly distributed, by randomly drawing a preference value for each good for each type. Similarly, uniformly randomly draw and fix the preferences for agent 1 beforehand (just once). If the computation takes too long, first try with agents having 2 goods each, 4 in total, and |Θ| = 15.

Then, verify that the obtained mechanism is truthful and individually rational. Note that, given output CB of the algorithm, the corresponding mechanism is
```
M_CB (θ) = argmax[o∈CB] {g(o)o∈CB |∀x ∈ CB . u(θ, o) ≥ u(θ, x)}
```
which chooses an optimal outcome for agent 1 among those in CB providing the highest utility to agent 2.

### References

[CS04] V. Conitzer and T. Sandholm (2004) An algorithm for automatically designing deterministic mechanisms without payments. Proceedings of AAMAS’04.
