from pyvirtualdisplay import Display
display = Display(visible=0, size=(1000, 1000))
display.start()
import pdb; pdb.set_trace()
import os
import gym
import matplotlib
matplotlib.use('agg')
from envs.modified_mountain_car import ModifiedMountainCarEnv
import random
import numpy as np
import keras
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, Flatten, MaxPool2D
from keras.optimizers import Adam
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from openai.replay_buffer import ReplayBuffer
from openai.schedules import PiecewiseSchedule
from keras.models import load_model
np.random.seed(0)
# def model_data_preparation_old():
#     training_data = []
#     accepted_scores = []
#     for game_index in tqdm(range(intial_games)):
#         score = 0
#         game_memory = []
#         previous_observation = env.reset()


#         for step_index in range(goal_steps):
#             action = random.randrange(0, 3)
#             observation, reward, done, info = env.step(action)

#             if len(previous_observation) > 0:
#                 game_memory.append([previous_observation, action])
                
#             previous_observation = observation
#             if observation[0] > -0.2:
#                 reward = 1
            
#             score += reward
#             if done:
#                 break



#         if score >= score_requirement:
#             accepted_scores.append(score)
#             for data in game_memory:
#                 if data[1] == 1:
#                     output = [0, 1, 0]
#                 elif data[1] == 0:
#                     output = [1, 0, 0]
#                 elif data[1] == 2:
#                     output = [0, 0, 1]
#                 training_data.append([data[0], output])
        
#         env.reset()
    
#     print(accepted_scores)
    
#     return training_data


# def test_old(model, tests=100, render = False):

#     scores = []
#     choices = []
#     for each_game in tqdm(range(tests)):
#         score = 0
#         prev_obs = env.reset()
#         for step_index in range(goal_steps):
#             # Uncomment this line if you want to see how our bot playing

#             if render: 
#                 arr = env.render()
#                 plt.imshow(arr)
#                 plt.show(block=False)
#                 plt.pause(.001)
#                 plt.close()

#             action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            
#             choices.append(action)
#             new_observation, reward, done, info = env.step(action)
#             prev_obs = new_observation
#             score+=reward
#             if done:
#                 break

        
#         scores.append(score)

#     return scores, choices
#     # print(scores)
#     # print('Average Score:',sum(scores)/len(scores))
#     # print('choice 1:{}  choice 0:{} choice 2:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(2)/len(choices)))

FRAMESKIP = 2

class Monitor(object):
    def __init__(self, env, filepath):
        self.frame_num = 0
        self.vid_num = 0
        self.filepath = os.path.join(os.getcwd(), filepath)
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        self.image_name = "image%05d.png"
        self.env = env
        self.images = []

    def save(self):
        #import matplotlib.pyplot as plt
        full_path = os.path.join(self.filepath, self.image_name % self.frame_num)
        self.images.append(full_path)
        # plt.imsave(full_path, self.env.render('rgb_array'))
        im = self.env.render()
        plt.imshow(im, cmap='gray')
        #plt.show(block=False)
        #plt.pause(.001)
        #plt.close()
        plt.imsave(full_path, im)
        self.frame_num += 1

    def make_video(self):
        import subprocess
        current_dir = os.getcwd()
        os.chdir(self.filepath)
        # #'ffmpeg -framerate 8 -i image%05d.png -r 30 -pix_fmt yuv420p car_vid_0.mp4'
        subprocess.call([
            'ffmpeg', '-hide_banner', '-loglevel', 'panic', '-framerate', '8', '-i', self.image_name, '-r', '30', '-pix_fmt', 'yuv420p',
            'car_vid_%s.mp4' % self.vid_num
        ])

        self.vid_num += 1
        self.frame_num = 0
        os.chdir(current_dir)

    def delete(self):
        self.frame_num = 0
        current_dir = os.getcwd()
        os.chdir(self.filepath)
        
        for file_name in [f for f in os.listdir(os.getcwd()) if '.png' in f]:
             os.remove(file_name)

        os.chdir(current_dir)


#nitor(self.env, 'videos')


def to_gray(arr):
    if len(arr.shape) == 2:
        return arr
    else:
        return np.dot(arr[...,:3]/255. , [0.299, 0.587, 0.114])

def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in tqdm(range(intial_games)):
        score = 0
        game_memory = []
        previous_observation = env.reset()
        frames = [to_gray(env.render())]*FRAMESKIP

        for step_index in range(goal_steps):
            action = random.randrange(0, 3)
            
            observation, reward, done, info = env.step(action)

            frames.append(to_gray(env.render()))
            frames.pop(0)

            
            game_memory.append([frames[:2], frames[1:], action])
                
            previous_observation = observation
            if observation[0] > -0.2:
                reward = 1
            
            score += reward
            if done:
                break

        if score >= score_requirement:
            print(game_index, score)
            accepted_scores.append(score)
            for data in game_memory:
                if data[-1] == 1:
                    output = [0, 1, 0]
                elif data[-1] == 0:
                    output = [1, 0, 0]
                elif data[-1] == 2:
                    output = [0, 0, 1]

                training_data.append([data[0], output])
        
        env.reset()
    
    print(accepted_scores)
    
    return training_data

def build_model(input_size, output_size):
        # model = Sequential()
        # model.add(Dense(128, input_dim=input_size, activation='relu'))
        # model.add(Dense(52, activation='relu'))
        # model.add(Dense(output_size, activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer=Adam())


        inp = keras.layers.Input(input_size, name='frames')
        actions = keras.layers.Input((3,), name='mask')
  
        # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
        # conv_1 = keras.layers.convolutional.Convolution2D(
        #     8, 5 , 5, subsample=(4, 4), activation='relu'
        # )(normalized)

        conv1 = Conv2D(64, kernel_size=16, strides=2, activation='relu', data_format='channels_first')(inp)
        #pool1 = MaxPool2D(data_format='channels_first')(conv1)
        conv2 = Conv2D(64, kernel_size=8, strides=2, activation='relu', data_format='channels_first')(conv1)
        #pool2 = MaxPool2D(data_format='channels_first')(conv2)
        conv3 = Conv2D(64, kernel_size=4, strides=2, activation='relu', data_format='channels_first')(conv2)
        #pool3 = MaxPool2D(data_format='channels_first')(conv3)
        flat = Flatten()(conv3)
        dense1 = Dense(10, activation='relu')(flat)
        dense2 = Dense(30, activation='relu')(dense1)
        out = Dense(output_size, activation='linear', name='all_Q')(dense2)
        filtered_output = keras.layers.dot([out, actions], axes=1)

        model = keras.models.Model(input=[inp, actions], output=[filtered_output])

        all_Q = keras.models.Model(inputs=[inp],
                                 outputs=model.get_layer('all_Q').output)
        
        rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='mse', optimizer=rmsprop)

        # model = Sequential()
        # model.add( Conv2D(8, kernel_size=5, strides=5, activation='relu', data_format='channels_first', input_shape=input_size) )
        # model.add( Conv2D(8, kernel_size=3, strides=3, activation='relu') )
        # model.add( Flatten() )
        # model.add( Dense(10, activation='relu') )
        # model.add(Dense(10, activation='relu'))
        # model.add(Dense(output_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam())
        # model.summary()

        return model, all_Q

def train_model(epochs = 5):
    arr = env.render()
    experience_replay = ReplayBuffer(50000)
    greedy_schedule = PiecewiseSchedule([(0,.3), (500, .1), (750, .01)], outside_value = .01)
    model, all_Q = build_model(input_size=(2,) + arr.shape, output_size=3)
    model.summary()
    target_model, all_Q_target = build_model(input_size=(2,) + arr.shape, output_size=3)
    trained = False
    max_timesteps = 200
    last_200 = [0]*200
    total_t = 0
    number_of_episodes = 0
    frame_skip = FRAMESKIP
    while not trained:

        state = env.reset()

        frames = [to_gray(env.render())]*FRAMESKIP
        score = 0
        t = 0
        done = False
        while (not done) and (t < max_timesteps):
            
            eps = np.random.random()
            if eps < greedy_schedule.value(number_of_episodes):
                action = np.random.choice(range(3))
            else:
                action = np.argmax(all_Q.predict(np.array(frames)[np.newaxis, ...]))
            
            rew = 0
            for _  in range(frame_skip):
                if done: continue
                next_state, reward, done, info = env.step(action)

                if next_state[1] > state[1] and next_state[1]>0 and state[1]>0:
                    reward = -.5
                elif next_state[1] < state[1] and next_state[1]<=0 and state[1]<=0:
                    reward = -.5
            

                # give more reward if the cart reaches the flag in 200 steps
                if done:
                    reward = 1.
                rew += reward
            rew /= frame_skip
            #else:
            #    # put a penalty if the no of time steps is more
            #    reward = -1.

            frames.append(to_gray(env.render()))
            experience_replay.add(np.array(frames[:2]), action, rew, np.array(frames[1:]), done)
            frames.pop(0)

            if (total_t % 3000) == 0:
                target_model.set_weights(model.get_weights()) 

            if total_t >= 5000:
                if (total_t % 20) == 0:
                    s,a,r,s_,dones = experience_replay.sample(64)
                    Q_s_ = all_Q_target.predict(s_)
                    y = r + np.max(Q_s_,axis=1)*(1.-dones.astype(float))
                    # print(y)
                    # print(model.predict([s, np.eye(3)[a]]))
                    # import pdb; pdb.set_trace()
                    model.fit([s, np.eye(3)[a]], y, verbose=0, epochs= 1)
                    # print(model.predict([s, np.eye(3)[a]]))

            state = next_state
            score += rew
            t += 1
            total_t += 1
            if done:
                break

        number_of_episodes += 1
        last_200.append(done)
        last_200.pop(0)
        print(number_of_episodes, int(score), done, np.mean(last_200))
        if np.mean(last_200) > .95:
            import pdb ;pdb.set_trace()
            trained = True

    return model, all_Q

def test(model, tests=100, render = False):

    frame_skip = 2
    model, all_Q = model
    scores = []
    choices = []
    dones = []
    mon = Monitor(env, 'videos')
    for each_game in tqdm(range(tests)):
        score = 0
        prev_obs = env.reset()
        frames = [to_gray(env.render())]*FRAMESKIP
        done = False
        if render:
            mon.save() 
            #arr = frames[-1]
            #plt.imshow(arr)
            #plt.show(block=False)
            #plt.pause(.001)
            #plt.close()

        for step_index in range(goal_steps):
            
            action = np.argmax(all_Q.predict(np.array(frames)[np.newaxis, ...]))
            
            choices.append(action)
            
            rew = 0
            for _ in range(frame_skip):
                if done: continue
                new_observation, reward, done, info = env.step(action)
                rew += reward
                if render: mon.save()
            frames.append(to_gray(env.render()))
            frames.pop(0)
            rew /= frame_skip
            #if render: 
                #mon.save()
		#arr = frames[-1]
                #plt.imshow(arr, cmap='gray')
                #plt.show(block=False)
                #plt.pause(.000001)
                #plt.close()


            prev_obs = new_observation
            score+=rew
            if done:
                break
        mon.make_video()
        dones.append(done)
        scores.append(score)

    return scores, dones, choices
    # print(scores)
    # print('Average Score:',sum(scores)/len(scores))
    # print('choice 1:{}  choice 0:{} choice 2:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(2)/len(choices)))


env = ModifiedMountainCarEnv() #gym.make('MountainCar-v0')
env.reset()
goal_steps = 200
score_requirement = -198
intial_games = 1000

# training_data = model_data_preparation()
# trained_model = train_model(epochs = 5)
#untrained_model = train_model(epochs = 1)

# scores, choices = test(trained_model, tests=100, render = False)
# print(scores)
# print('Average Score:',sum(scores)/len(scores))
# print('choice 1:{}  choice 0:{} choice 2:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(2)/len(choices)))

trained_model_Q = load_model('trained_model_Q.h5')
trained_model = load_model('trained_model.h5')
_,_,_ = test([trained_model, trained_model_Q], tests=1, render = 1)#False)
#_,_ = test(untrained_model, tests=1, render = True)


#yaml_string = trained_model.to_yaml()


#trained_model.save('car_trained.h5')
import pdb; pdb.set_trace()
