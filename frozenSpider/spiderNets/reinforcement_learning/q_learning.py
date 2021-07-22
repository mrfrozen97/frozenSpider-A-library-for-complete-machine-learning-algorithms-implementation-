#import pygame
import os
import random
from collections import deque
#from enum import Enum
import torch
#import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
#from collections import namedtuple
import matplotlib.pyplot as plt



class Deep_q_learning:

    def __init__(self, game,
                 max_memory=50_000, epsilon=0, gamma=0.9, lr=0.001,
                 outputs=3, inputs=11, hidden_layers=[256],
                 batch_size=100,  average_last=10,
                 show_graph=False, plot_every=10, plot_display_time=2):
        self.plot_scores = []
        self.plot_scoresx = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.record = 0
        self.average_last = average_last
        self.show_graph = show_graph
        self.agent = Agent(max_memory=max_memory, gamma=gamma, epsilon=epsilon, lr=lr,
                           outputs=outputs, inputs=inputs, hidden_layers=hidden_layers,
                           batch_size=batch_size)
        self.game = game
        self.game.agent = self.agent
        self.valid_game = True
        self.plot_display_time=plot_display_time
        self.plot_every = plot_every
        self.left_score_ptr = 0
        if not(callable(getattr(self.game, 'reset', None))):
            self.valid_game = False
            print("Warning: Game object class does not contain method reset\n" +
                  "reset method should reset/restart the game\n")
        else:
            pass

        if not(callable(getattr(self.game, 'play_step', None))):
            self.valid_game = False
            print("Warning: Game object class does not contain method play_step\n" +
                  "play_step method takes in action and returns following values\n" +
                  "1)Reward\n2)game_over (boolean)\n3)score\n")
        else:
            pass

        if not(callable(getattr(self.game, 'get_state', None))):
            self.valid_game = False
            print("Warning: Game object class does not contain get_state\n" +
                  "play_step method should return an array of states that are given input to the neural nets\n")
        else:
            pass






    def plot_graph(self):

        plt.style.use("dark_background")
        plt.ion()
        plt.plot(self.plot_scoresx, self.plot_scores, label='Score')
        plt.plot(self.plot_scoresx, self.plot_mean_scores, label= 'Avg Score')
        plt.ylabel("Score", fontdict={'size':10, 'color':'blue'})
        plt.xlabel("Number of Games", fontdict={'size':10, 'color':'blue'})
        plt.legend()
        plt.show()
        plt.pause(self.plot_display_time)
        plt.close()








    def train_model(self):
        if self.valid_game:
            n_iters = 0
            while True:
                # get old state
                state_old = self.game.get_state()

                # get move
                final_move = self.agent.get_action(state_old)

                # perform move and get new state
                reward, done, score = self.game.play_step(final_move)
                state_new = self.game.get_state()  # /8

                self.agent.train_short_memory(state_old, final_move, reward, state_new, done)
                self.agent.remember(state_old, final_move, reward, state_new, done)

                if done:
                    n_iters+=1
                    # train long memory
                    self.game.reset()
                    self.agent.n_games += 1
                    self.agent.train_long_memory()
                    self.score = score
                    if self.score > self.record:
                        self.record = self.score
                        self.agent.model.save()
                    print('Game', self.agent.n_games, 'Score', self.score, 'Record:', self.record)
                    self.plot_scores.append(score)
                    self.plot_scoresx.append(n_iters)
                    self.total_score += score
                    if n_iters>self.average_last:
                        self.total_score-=self.plot_scores[self.left_score_ptr]
                        self.left_score_ptr += 1
                    self.plot_mean_scores.append(self.total_score / min(n_iters, self.average_last))

                    if n_iters%self.plot_every==0:
                        self.plot_graph()
        else:
            print("The game object class does not contain the required methods\nCheck if class contains reset, train_step, methods ")



class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_sizes=[256],
                 hidden_activation='relu', output_size=3,
                 dropout=None, input_activation='relu',
                 output_activation='linear'):
        super().__init__()
        temp_size = hidden_sizes[0]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, temp_size))
        if len(hidden_sizes)>1:
            for i in hidden_activation[1:]:
                self.layers.append(nn.Linear(temp_size, i))
                temp_size = i

        self.layers.append(nn.Linear(temp_size, output_size))





    def forward(self, x):
        for i in self.layers[:-1]:
            x = f.relu(i(x))

        x = self.layers[-1](x)
        return x



    def save(self, filename='model.pth'):
        model_folder_path = './mode'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        filename = os.path.join(model_folder_path, filename)
        #torch.save(self.state_dict(), filename)









class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.MINIBATCH_SIZE = 100


    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()









class Agent:

    def __init__(self, max_memory=50_000, epsilon=0, gamma=0.9, lr=0.001,
                 outputs=3, inputs=11, hidden_layers=[256],
                 batch_size=100):
        self.n_games = 0
        self.epsilon = epsilon  # Randomness to agent
        self.memory =deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.gamma = gamma

        #model
        self.model = Linear_QNet(inputs, hidden_layers, output_size=outputs)
        self.trainer = QTrainer(self.model, lr=lr, gamma=self.gamma)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):

        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)



    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move















########################################################################################################################




#Example of using this file is shown below




"""

pygame.init()
font = pygame.font.Font(pygame.font.get_default_font(), 25)


# font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 200


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.agent = None
        self.hscore = 0
        self.score = 0
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]


        if self.score > self.hscore:
            self.hscore = self.score

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0




    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y  # food down
        ]

        return np.array(state, dtype=int)




    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            #reward-=0.05
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(int(pt.x), int(pt.y), BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(int(pt.x + 4), int(pt.y) + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        text = font.render("Games: " + str(self.agent.n_games), True, WHITE)
        self.display.blit(text, [140, 0])
        text = font.render("Highest Score: " + str(self.hscore), True, WHITE)
        self.display.blit(text, [310, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)














def train():
    #agent = Agent()
    game = SnakeGameAI()
    ql_model = Deep_q_learning(game=game)
    ql_model.train_model()






if __name__=='__main__':
    train()










"""
















