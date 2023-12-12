import time


from pidqn import *
import gymnasium as gym
import datetime
import numpy as np

CURRENT_MODEL = "PIDQN"  # can be DQN or PIDQN
TAU = 0.005
LR = 1e-4
BS = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
MEM_SIZE = 10000
TRAIN_FREQ = 1
TRAIN_START = 0
N_HID = [128, 128]


def load_model(model_filename: str):
    print("Loading Model")
    if CURRENT_MODEL == "DQN":
        model = DQN.load("log/" + model_filename)
    elif CURRENT_MODEL == "PIDQN":
        model = PIDQN.load("log/" + model_filename)
    else:
        raise ValueError("Not a valid model in CURRENT_MODEL variable!")
    return model


def train_new_model(total_timesteps: int = 600, n_hid=None, tau: float = 0.005, lr: float = 0.0003, batch_size: int = 1024, gamma: float = 0.99, eps_start: float = 0.9, eps_end: float = 0.04, eps_decay: int = 1000, memory_size: int = 10000, train_freq: int = 1, train_start: int = 1000, persistence=0.2):
    env = gym.make('CartPole-v1')
    print("Init Model")
    if n_hid is None:
        n_hid = N_HID
    if CURRENT_MODEL == "DQN":
        model = DQN(env, n_hid=n_hid, tau=tau, lr=lr, batch_size=batch_size, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, memory_size=memory_size, train_freq=train_freq, train_start=train_start, persistence=persistence)
    elif CURRENT_MODEL == "PIDQN":
        model = PIDQN(env, n_hid=n_hid, tau=tau, lr=lr, batch_size=batch_size, gamma=gamma, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, memory_size=memory_size, train_freq=train_freq, train_start=train_start, persistence=persistence)
    else:
        raise ValueError(CURRENT_MODEL + " is not a valid model!")
    print("Start learning")
    model.learn(total_timesteps=total_timesteps)  # 60 or 600
    print("Finish learning")
    ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    filename = CURRENT_MODEL + "-" + ct
    model.save("log/" + filename)
    return model, filename


def shuffle_observations(num_obs, model_file_name, testing_time):
    log_filename = "log/shuffle/" + model_file_name + "-" + testing_time + "-shuffle_log.txt"
    rng = np.random.default_rng()
    new_obs = rng.permutation(num_obs)
    # with open(log_filename, "w") as f:
    #    f.write("Old Obs: " + str(obs) + " New Obs: " + str(new_obs) + "\n")
    return new_obs

def shuffle_observations_without_log(num_obs):
    rng = np.random.default_rng()
    new_obs = rng.permutation(num_obs)
    # with open(log_filename, "w") as f:
    #    f.write("Old Obs: " + str(obs) + " New Obs: " + str(new_obs) + "\n")
    return new_obs

def create_noise(num_extra_channels, noise_type):
    noise = np.zeros(num_extra_channels)
    if noise_type == "zeros":
        return noise
    #elif noise_type == "random":
    #elif noise_type == "gaussian":
    return noise


def shuffle_observations_add_noise(obs, num_extra_channels, model_file_name, testing_time):
    log_filename = "log/shuffle/" + model_file_name + "-" + testing_time + "-shuffle_log.txt"
    rng = np.random.default_rng()
    noise_channels = create_noise(num_extra_channels, "zeros")
    new_obs = obs + noise_channels
    shuffle_with_noise = rng.permutation(obs.shape[0]+num_extra_channels)
    new_obs = new_obs[shuffle_with_noise]
    return new_obs


def train_fnn(total_timesteps: int = 600):
    env = gym.make('CartPole-v1')
    print("Init Model")
    # model = FNN(env)
    model = DQN(env)
    print("Start learning")
    model.learn(total_timesteps=total_timesteps)
    print("Finish learning")
    ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    filename = "FNN" + "-" + ct
    model.save("log/" + filename)
    return model, filename

def testing_model_noise(filename: str = "test.txt", name: str = "testing"):
    model = PIDQN.load("log/"+filename)
    tests = []
    shuffles = []
    for t in range(1, 100):
        # test the model
        env = gym.make('CartPole-v1')
        obs, _ = env.reset()
        not_done = True
        duration = 0
        shuffle = shuffle_observations_without_log(obs.shape[0])
        while not_done:
            obs = obs[shuffle]
            obs += 0.05 * obs # noise
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            duration += 1
            not_done = not (terminated or truncated)

        tests += [model.episode_durations[-1]]
        shuffles += [shuffle]
    test_filename = filename + name + "-test.txt"
    with open(test_filename, "w") as f:
        f.write("tests: " + str(tests) + "\n")
        f.write("shuffles: " + str(shuffles) + "\n")

def testing_model_double_channels(filename: str = "test.txt", name: str = "testing"):
    model = PIDQN.load("log/"+filename)
    tests = []
    shuffles = []
    for t in range(1, 100):
        # test the model
        env = gym.make('CartPole-v1')
        obs, _ = env.reset()
        not_done = True
        duration = 0
        shuffle = shuffle_observations_without_log(obs.shape[0])
        while not_done:
            obs = obs[shuffle]
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            duration += 1
            not_done = not (terminated or truncated)

        tests += [model.episode_durations[-1]]
        shuffles += [shuffle]
    test_filename = filename + name + "-test.txt"
    with open(test_filename, "w") as f:
        f.write("tests: " + str(tests) + "\n")
        f.write("shuffles: " + str(shuffles) + "\n")


def testing_model(filename: str = "test.txt", name: str = "testing", mode: str = "shuffle"):
    model = PIDQN.load("log/"+filename)
    tests = []
    shuffles = []
    shuffle = shuffle_observations_without_log(4)
    for t in range(1, 100):
        # test the model
        env = gym.make('CartPole-v1')
        obs, _ = env.reset()
        not_done = True
        duration = 0
        #shuffle = shuffle_observations_without_log(obs.shape[0])
        while not_done:
            if mode == "normal":
                obs = obs
            elif mode == "shuffle":
                obs = obs[shuffle]
            elif mode == "noise":
                obs += 0.05 * obs  # noise
            elif mode == "noise2":
                obs += 0.10 * obs  # more static noise
            elif mode == "noise3":
                obs += 0.02 * obs  # less static noise
            elif mode == "noise4": # Impulse Noise
                noise_sample = np.random.default_rng().uniform(0.02*min(obs), 0.03*max(obs), int(0.03*len(obs)))
                zeros = np.zeros(len(obs) - len(noise_sample))
                noise = np.concatenate([noise_sample, zeros])
                np.random.shuffle(noise)
            elif mode == "noise5": # Impulse Noise
                noise_sample = np.random.default_rng().uniform(0.2*min(obs), 0.3*max(obs), int(0.03*len(obs)))
                zeros = np.zeros(len(obs) - len(noise_sample))
                noise = np.concatenate([noise_sample, zeros])
                np.random.shuffle(noise)
                obs = obs + noise
            elif mode == "noise6":
                noise = np.random.normal(0, 0.2, len(obs))
                obs = obs + noise  # gaussian noise 1
            elif mode == "noise7":
                noise = np.random.normal(0, 0.1, len(obs))
                obs = obs + noise  # gaussian noise 2
            elif mode == "noise8":
                noise = np.random.normal(0, 0.02, len(obs))
                obs = obs + noise  # gaussian noise 2
            elif mode == "noise9":
                noise = np.random.normal(0, 0.05, len(obs))
                obs = obs + noise  # gaussian noise 3
            elif mode == "noise10":  # Impulse Noise
                noise_sample = np.random.default_rng().uniform(0.2*min(obs), 0.4*max(obs), int(0.03*len(obs)))
                zeros = np.zeros(len(obs) - len(noise_sample))
                noise = np.concatenate([noise_sample, zeros])
                np.random.shuffle(noise)
                obs = obs + noise
            elif mode == "noise11":  # Impulse Noise
                noise_sample = np.random.default_rng().uniform(0.1*min(obs), 0.3*max(obs), int(0.03*len(obs)))
                zeros = np.zeros(len(obs) - len(noise_sample))
                noise = np.concatenate([noise_sample, zeros])
                np.random.shuffle(noise)
                obs = obs + noise
            elif mode == "noise12":  # Impulse Noise
                noise_sample = np.random.default_rng().uniform(0.1*min(obs), 0.4*max(obs), int(0.03*len(obs)))
                zeros = np.zeros(len(obs) - len(noise_sample))
                noise = np.concatenate([noise_sample, zeros])
                np.random.shuffle(noise)
                obs = obs + noise
            elif mode == "double13":
                obs[0] = obs[1]
            elif mode == "double2":
                obs[1] = obs[3]
            elif mode == "double3":
                obs[3] = obs[2]
            elif mode == "double4":
                obs[0] = obs[2]
            elif mode == "double5":
                obs[0] = obs[3]
            elif mode == "double6":
                obs[1] = obs[2]
            elif mode == "double7":
                obs[1] = obs[0]
            elif mode == "double8":
                obs[2] = obs[0]
            elif mode == "double9":
                obs[2] = obs[1]
            elif mode == "double10":
                obs[2] = obs[3]
            elif mode == "double11":
                obs[3] = obs[0]
            elif mode == "double12":
                obs[3] = obs[1]
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            duration += 1
            not_done = not (terminated or truncated)

        tests += [duration]
        shuffles += [shuffle]
    test_filename = "log/" + filename + name + "-test.txt"
    with open(test_filename, "w") as f:
        f.write("Testing: " + str(mode) + "\n")
        f.write("tests: " + str(tests) + "\n")
        f.write("shuffles: " + str(shuffle) + "\n")



if __name__ == '__main__':

    # model_file_name = "PIDQN-2023-07-30-19-02-21.535441"# "DQN-2023-07-31-13-03-27.107234"  # "PIDQN-2023-07-30-19-02-21.535441"
    # model = load_model(model_file_name)
    testing_models = []  # TAU,LR,BS,GAMMA,EPS_START,EPS_END,EPS_DECAY,MEM_SIZE,TRAIN_FREQ,TRAIN_START,N_HID, PERSISTENCE

    testing_models.append([0.005,  1e-4, 128, 0.99, 0.9, 0.05, 1000, 10000, 1, 0, [128, 128], 0.1])

    for m in testing_models:

        timesteps = 500
        model, model_file_name = train_new_model(timesteps, m[-2], m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], persistence=m[-1])
        # model = PIDQN.load('log/PIDQN-2023-08-03-12-37-12.221417')
        # ct = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
        # model_file_name = CURRENT_MODEL + "-" + ct
        testing_time = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")

        print(testing_time)
        print("Model complete")

        for tests in range(1, 20):
            print(f'saving after {timesteps*tests} episodes...')
            model.save('log/' + model_file_name)
            print('saved!')
            current_time = time.time()
            model.learn(timesteps)
            # test the model
            env = gym.make('CartPole-v1')
            obs, _ = env.reset()
            not_done = True
            duration = 0
            shuffle = shuffle_observations(obs.shape[0], model_file_name, testing_time)
            print(shuffle)
            while not_done:
                obs = obs[shuffle]
                action, _states = model.predict(obs)
                obs, rewards, terminated, truncated, info = env.step(action)
                duration += 1
                not_done = not (terminated or truncated)
            print('#########################################################')
            print(f'Test Score after {timesteps*(tests+1)} episodes: {duration}')
            print('#########################################################')
            print(f'took {time.time() - current_time} seconds')
