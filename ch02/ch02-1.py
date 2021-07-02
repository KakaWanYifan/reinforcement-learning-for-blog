import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env = env.unwrapped
env.seed(0)
print('观察空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测空间大小 = {}'.format(env.unwrapped.nS))
print('动作空间大小 = {}'.format(env.unwrapped.nA))
env.render()

print(env.P[0][1])


def play_policy(env, policy, render=False):
    # 回报
    total_reward = 0.
    # 状态
    observation = env.reset()
    while True:
        if render:
            env.render()  # 此行可显示
        action = np.random.choice(env.action_space.n, p=policy[observation])
        # 状态、奖励、是否完成
        observation, reward, done, _ = env.step(action)
        # 累积奖励
        total_reward += reward
        # 游戏结束
        if done:
            break
    return total_reward


# 随机策略
random_policy = np.ones((env.nS, env.nA)) / env.nA

episode_rewards = [play_policy(env, random_policy) for _ in range(100)]
print("随机策略 平均奖励：{}".format(np.mean(episode_rewards)))


def v2q(env, v, s=None, gamma=1.):
    """
    根据状态价值函数计算动作价值函数
    :param env: 环境
    :param v: 状态价值
    :param s: 状态
    :param gamma: 衰减率
    :return: 动作价值
    """
    # 针对单个状态求解
    if s is not None:
        q = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q[a] += prob * (reward + gamma * v[next_state] * (1. - done))
    # 针对所有状态求解
    else:
        q = np.zeros((env.unwrapped.nS, env.unwrapped.nA))
        for s in range(env.unwrapped.nS):
            q[s] = v2q(env, v, s, gamma)
    return q


def evaluate_policy(env, policy, gamma=1., theta=1e-6, kmax=1000000):
    """
    评估策略的状态价值
    :param env: 环境
    :param policy: 策略
    :param gamma: 衰减率
    :param theta: 最小更新值
    :param kmax: 最大迭代次数
    :return:
    """
    # 初始化状态价值函数
    v = np.zeros(env.unwrapped.nS)
    for k in range(kmax):
        delta = 0
        for s in range(env.unwrapped.nS):
            # 更新状态价值函数
            vs = sum(policy[s] * v2q(env, v, s, gamma))
            # 更新最大误差
            delta = max(delta, abs(v[s] - vs))
            # 更新状态价值函数
            v[s] = vs
        if delta < theta:  # 查看是否满足迭代条件
            break
    return v


print('状态价值函数：')
v_random = evaluate_policy(env, random_policy)
print(v_random.reshape(4, 4))

print('动作价值函数：')
q_random = v2q(env, v_random)
print(q_random)


def improve_policy(env, v, policy, gamma=1.):
    optimal = True
    # 遍历所有的状态
    for s in range(env.nS):
        # 求当前状态的所有动作的价值
        q = v2q(env, v, s, gamma)
        # 选取价值对好的动作
        a = np.argmax(q)
        # 如果当前策略的当前动作的概率不是1
        if policy[s][a] != 1.:
            # 优化没有完成
            optimal = False
            policy[s] = 0.
            policy[s][a] = 1.
    return optimal


policy = random_policy.copy()
optimal = improve_policy(env, v_random, policy)
if optimal:
    print('无更新，最优策略为：')
else:
    print('有更新，更新后的策略为：')

print(np.argmax(policy, axis=1).reshape(4, 4))


def iterate_policy(env, gamma=1., tolerant=1e-6):
    # 初始化为任意一个策略
    policy = np.ones((env.unwrapped.nS, env.unwrapped.nA)) / env.unwrapped.nA
    while True:
        v = evaluate_policy(env, policy, gamma, tolerant)  # 策略评估
        if improve_policy(env, v, policy):  # 策略改进
            break
    return policy, v


policy_pi, v_pi = iterate_policy(env)
print('状态价值函数 =')
print(v_pi.reshape(4, 4))
print('最优策略 =')
print(np.argmax(policy_pi, axis=1).reshape(4, 4))

# 看看效果
episode_rewards = [play_policy(env, policy_pi) for _ in range(100)]
print("平均奖励：{}".format(np.mean(episode_rewards)))


def iterate_value(env, gamma=1, tolerant=1e-6, kmax=1000000):
    v = np.zeros(env.unwrapped.nS)  # 初始化
    for i in range(kmax):
        delta = 0
        for s in range(env.unwrapped.nS):
            vmax = max(v2q(env, v, s, gamma))  # 更新价值函数
            delta = max(delta, abs(v[s] - vmax))
            v[s] = vmax
        if delta < tolerant:  # 满足迭代需求
            break

    policy = np.zeros((env.unwrapped.nS, env.unwrapped.nA))  # 计算最优策略
    for s in range(env.unwrapped.nS):
        a = np.argmax(v2q(env, v, s, gamma))
        policy[s][a] = 1.
    return policy, v


policy_vi, v_vi = iterate_value(env)
print('状态价值函数 =')
print(v_vi.reshape(4, 4))
print('最优策略 =')
print(np.argmax(policy_vi, axis=1).reshape(4, 4))
