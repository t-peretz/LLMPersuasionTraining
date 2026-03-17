# Argumentative Reasoning with Alternating RL Agents
This project explores whether reasoning can emerge through argument and critique between LLM’s, inspired by the argumentative theory of reasoning proposed by Hugo Mercier and Dan Sperber. Their theory suggests that human reasoning evolved primarily to produce and evaluate arguments in social contexts, rather than to solve problems in isolation.

The system trains two interacting agents:

- Agent A generates answers and reasoning to a question.
- Agent B evaluates whether the answer is correct.

Training proceeds in alternating phases:

1. Agent A generates answers and reasoning.
2. Agent B judges the answer as correct or incorrect.
3. Agent A receives reward based on whether it persuades B.
4. Agent B is then trained to improve its judging ability.

Agent A is optimized using Group Relative Policy Optimization (GRPO), while Agent B is trained using a GRPO-style reinforcement objective derived from its judging trajectories. This creates a feedback loop where argumentation and evaluation co-evolve.
The goal is to investigate whether argumentative interaction between models can improve mathematical reasoning performance.

## Results

After training for one epoch on the GSM8K training set, the following results are obtained on the GSM8K test set:

- **Persuasion:** Agent B agrees with Agent A's answer in **94.16%** of cases.
- **Evaluation:** Agent B correctly evaluates Agent A's answer in **83.32** of cases.
- **Accuracy:** Agent A answers the problem correctly in **78.70%** of cases.

## References

- Mercier, H., & Sperber, D. (2011). *Why do humans reason? Arguments for an argumentative theory*. Behavioral and Brain Sciences.
- Mercier, H., & Sperber, D. (2017). *The Enigma of Reason*.
- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. 
