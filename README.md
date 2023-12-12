# How to use this implementation #

This is an implementation of a permutation invariant reinforcement learning based on the math of [this paper](https://attentionneuron.github.io/)
To facilitate ease of use the fuctionalities mirror the basic fuctionalities from  [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/).


Example code ([mirroring Stable Baselines Quickstart Guide](https://stable-baselines.readthedocs.io/en/master/guide/quickstart.html))
```
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    print("Init Model")
    if CURRENT_MODEL == "DQN":
        model = DQN(env, n_hid=[128, 128])
    elif CURRENT_MODEL == "PIDQN":
        model = PIDQN(env, n_hid=[128, 128])
    else:
        raise ValueError(CURRENT_MODEL + " is not a valid model!")
    model.learn(total_timesteps=1000)  # 60 or 600
    ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    filename = CURRENT_MODEL + "-" + ct
    model.save("log/" + filename)

    del model

    if CURRENT_MODEL == "DQN":
        model = DQN.load("log/" + filename)
    elif CURRENT_MODEL == "PIDQN":
        model = PIDQN.load("log/" + filename)
    else:
        raise ValueError("Not a valid model in CURRENT_MODEL variable!")

    print("Model complete")
    env = gym.make('CartPole-v1', render_mode="human")
    obs, info = env.reset()
    not_done = True
    while not_done:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        not_done = not (terminated or truncated)
    print(f"Score: {model.episode_durations[-1]}")}
```