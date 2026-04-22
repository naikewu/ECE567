# Fixed-Clip CPPOPID

## Motivation

PPOLag is a common baseline for safe reinforcement learning because it handles the reward-cost tradeoff by learning a Lagrange multiplier. However, the standard PPOLag multiplier update may react slowly when the constraint violation is large, while a full PID-style penalty controller can sometimes change the multiplier too aggressively from one epoch to the next. Large one-step changes in the penalty may over-correct the policy, destabilize training, or make the controller overly conservative.

To address this issue, we propose **Fixed-Clip CPPOPID**, a clipped-multiplier variant of CPPOPID. The key idea is simple: we keep the original PID-based penalty controller, but we clip the *per-update change* of the Lagrange multiplier to a fixed range. Instead of directly jumping to the raw PID target, the multiplier can move by at most a constant amount in each epoch. This keeps the controller responsive while preventing abrupt penalty spikes.

## Method

Fixed-Clip CPPOPID preserves the original PPO actor-critic optimization and the standard PID-Lagrangian structure. As in PPO-Lag and CPPOPID, the policy is optimized using a mixed advantage

$$
A_t = \frac{A^r_t - \lambda_t A^c_t}{1 + \lambda_t},
$$

where $A^r_t$ and $A^c_t$ denote the reward and cost advantages, respectively, and $\lambda_t \ge 0$ is the Lagrange multiplier. The actor is then updated using the PPO clipped objective

$$
\mathcal{L}^{\text{clip}}(\theta)
=
\mathbb{E}_t \Big[
\min \big(
r_t(\theta) A_t,\,
\operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\big)
\Big],
$$

where

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
$$

In standard CPPOPID, the penalty is produced by a PID controller driven by the average episode cost. Let $J_c^{(k)}$ be the mean episode cost at epoch $k$, and let $d$ be the cost limit. We first define the cost error

$$
e_k = J_c^{(k)} - d.
$$

The integral term is updated as

$$
I_{k+1} = \max \big(0,\; I_k + K_I e_k \big).
$$

To smooth the proportional and derivative signals, we use exponential moving averages:

$$
P_{k+1} = \alpha_p P_k + (1-\alpha_p)e_k,
$$

$$
C_{k+1} = \alpha_d C_k + (1-\alpha_d)J_c^{(k)}.
$$

The derivative contribution is computed with a delayed difference,

$$
D_{k+1} = \max \big(0,\; C_{k+1} - C_{k+1-\tau} \big),
$$

and the raw PID target multiplier is

$$
\tilde{\lambda}_{k+1}
=
\max \big(0,\; K_P P_{k+1} + I_{k+1} + K_D D_{k+1} \big).
$$

Our modification is to clip the **change** in the multiplier rather than its absolute value. Let $\Delta_\lambda^{\max}$ denote the fixed per-epoch change limit. Then the final multiplier update becomes

$$
\lambda_{k+1}
=
\max \Big(
0,\;
\lambda_k + \operatorname{clip}\big(
\tilde{\lambda}_{k+1} - \lambda_k,\;
-\Delta_\lambda^{\max},\;
\Delta_\lambda^{\max}
\big)
\Big).
$$

In our implementation, we set $\Delta_\lambda^{\max} = 1.0$. This design keeps the penalty nonnegative and prevents large jumps in the Lagrange multiplier. Intuitively, the controller can still respond to sustained violations, but it must do so gradually. Compared with the unclipped PID controller, the fixed-clip rule is less likely to overreact to transient cost spikes and therefore can produce a smoother penalty schedule.

## Environment

We evaluate Fixed-Clip CPPOPID on two standard safe RL environments, **SafetyPointGoal1-v0** and **SafetySwimmerVelocity-v1**, using **5 random seeds**. Each run uses **4M total environment steps** with **20k steps per epoch**. The cost limit is set to **25**, and both the actor and critic use two hidden layers of size **256**. The controller hyperparameters are fixed across the experiments: $K_P = 0.67$, $K_I = 0.0028$, $K_D = 0.168$, $\alpha_p = 0.90$, $\alpha_d = 0.90$, initial multiplier $\lambda_0 = 0.05$, and fixed clip $\Delta_\lambda^{\max} = 1.0$.

The PointGoal baseline is PPO-Lag with 5-seed averaged time series from the project repository. The Swimmer baseline is also PPO-Lag averaged over 5 seeds. We compare the mean reward and mean cost trajectories of Fixed-Clip CPPOPID against the PPO-Lag baseline.

**[Insert Figure 1 here: SafetyPointGoal1-v0, Fixed-Clip CPPOPID vs PPO-Lag mean curves.]**

**[Insert Figure 2 here: SafetySwimmerVelocity-v1, Fixed-Clip CPPOPID vs PPO-Lag mean curves.]**

## Results

Table 1 reports the final 5-seed mean performance at 4M steps.

| Environment | Method | Final Return Mean | Final Return Std | Final Cost Mean | Final Cost Std |
| --- | --- | ---: | ---: | ---: | ---: |
| SafetyPointGoal1-v0 | PPO-Lag | 11.742 | 2.495 | 23.866 | 2.317 |
| SafetyPointGoal1-v0 | Fixed-Clip CPPOPID | 11.538 | 1.731 | 28.018 | 5.021 |
| SafetySwimmerVelocity-v1 | PPO-Lag | 59.314 | 16.090 | 28.096 | 2.246 |
| SafetySwimmerVelocity-v1 | Fixed-Clip CPPOPID | 106.352 | 60.466 | 23.740 | 17.809 |

On **SafetyPointGoal1-v0**, the proposed method reduces the mean cost much faster in the early phase of training. The mean cost of Fixed-Clip CPPOPID first drops below the cost limit at approximately **0.32M** steps, while the PPO-Lag baseline does not do so until about **3.36M** steps. This suggests that clipping the per-update multiplier change still allows the PID controller to react early and enforce the safety constraint faster than PPO-Lag. However, the final outcome is mixed: Fixed-Clip CPPOPID ends with nearly the same mean return as PPO-Lag (11.538 vs 11.742, a **1.7%** decrease), but with a higher final mean cost (28.018 vs 23.866, a **17.4%** increase). Therefore, on PointGoal the fixed-clip rule improves early-stage cost reduction but does not maintain the same final safety level as the baseline.

On **SafetySwimmerVelocity-v1**, the effect is much more favorable. Fixed-Clip CPPOPID drives the mean cost below the limit at approximately **0.18M** steps, while the PPO-Lag baseline does **not** reach the cost limit by the end of the 4M-step training horizon. At 4M steps, Fixed-Clip CPPOPID achieves a much higher final mean return than PPO-Lag (106.352 vs 59.314, a **79.3%** increase) and also a lower final mean cost (23.740 vs 28.096, a **15.5%** reduction). This shows that in Swimmer, the clipped PID update gives a substantially better reward-safety tradeoff than the baseline.

Taken together, these results suggest that clipping the *change* in the multiplier is a useful stabilization mechanism for PID-based safe RL, but its benefits are environment dependent. In both environments, the method makes the penalty controller react early enough to reduce cost quickly. On Swimmer, this leads to both better return and better safety. On PointGoal, the same mechanism appears to make the controller responsive early on but not conservative enough later in training, so the final cost drifts above the constraint even though the final return remains competitive. This indicates that fixed clipping can be a strong and simple improvement when the baseline penalty update is too slow, but it may still need environment-specific tuning to maintain the best long-horizon safety performance.
