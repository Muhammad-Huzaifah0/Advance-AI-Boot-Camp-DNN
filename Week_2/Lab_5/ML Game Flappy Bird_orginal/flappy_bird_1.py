import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sys
import pickle


# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Flappy Bird')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)                             

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Bird settings
#pygame.Rect(left, top, width, height)
bird = pygame.Rect(50, 300, 30, 30)
bird_velocity = 0
gravity = 0.5

# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -3

# Load assets
try:
    bird_image = pygame.image.load(r"E:\BootCamp (AI)\Labs\Week_2\Lab_5\bird.png")
    bird_image = pygame.transform.scale(bird_image, (30, 30))  # Resize bird to 30x30 pixels
    pipe_image = pygame.image.load(r"E:\BootCamp (AI)\Labs\Week_2\Lab_5\pipe.png")

    jump_sound = pygame.mixer.Sound(r"E:\BootCamp (AI)\Labs\Week_2\Lab_5\jump.mp3")
    collision_sound = pygame.mixer.Sound(r"E:\BootCamp (AI)\Labs\Week_2\Lab_5\collision.mp3")

except pygame.error as e:
    print(f"Error loading assets: {e}")
    pygame.quit()
    sys.exit()

# Create a neural network model
model = Sequential([
    Dense(24, activation='relu', input_shape=(4,)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

#bird_y: Vertical position of the bird (normalized).
#bird_velocity: Vertical velocity of the bird (normalized).
#pipe_x: Horizontal position of the nearest pipe (normalized).
#pipe_gap_y: Vertical position of the gap between the nearest pipes (normalized).


def get_state(bird, pipes, bird_velocity):
    bird_y = bird.y / HEIGHT  # Normalize
    bird_velocity /= 10  # Normalize
    pipe_x = pipes[0][0].x / WIDTH  # Normalize
    pipe_gap_y = pipes[0][0].height / HEIGHT  # Normalize
    return np.array([bird_y, bird_velocity, pipe_x, pipe_gap_y])

def get_reward(bird, pipes):
    if bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1]) or bird.y > HEIGHT:
        return -1  # Collision penalty
    return 0.1  # Reward for staying alive

def create_pipe():
    height = random.randint(100, 400)
    top_pipe = pygame.Rect(WIDTH, 0, pipe_width, height)
    bottom_pipe = pygame.Rect(WIDTH, height + pipe_gap, pipe_width, HEIGHT - height - pipe_gap)
    return top_pipe, bottom_pipe

def draw_bird_and_pipes(bird, pipes):
    screen.fill(WHITE)
    screen.blit(bird_image, (bird.x, bird.y))
    for pipe in pipes:
        top_pipe_image = pygame.transform.scale(pipe_image, (pipe_width, pipe[0].height))
        bottom_pipe_image = pygame.transform.scale(pipe_image, (pipe_width, HEIGHT - pipe[0].height - pipe_gap))
        screen.blit(top_pipe_image, (pipe[0].x, pipe[0].y))
        screen.blit(bottom_pipe_image, (pipe[1].x, pipe[1].y))
    pygame.display.flip()

def reset_game():
    global bird, bird_velocity, pipes
    bird = pygame.Rect(50, 300, 30, 30)
    bird_velocity = 0
    pipes = [create_pipe()]

def collect_training_data():
    global bird_velocity
    training_data = []
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_velocity = -8
                if hasattr(pygame, 'mixer'):
                    jump_sound.play()

        # Bird movement
        bird_velocity += gravity
        bird.y += bird_velocity

        # Pipe movement
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        # Remove pipes off the screen
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())

        # Collision detection and reward calculation
        reward = get_reward(bird, pipes)
        state = get_state(bird, pipes, bird_velocity)
        training_data.append((state, reward))

        if reward == -1:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes)
        clock.tick(30)

    # Save training data to a file
    with open('training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)

    return training_data
        
    # # Print some training data for debugging
    # for i in range(10):
    #     print(f"State: {training_data[i][0]}, Reward: {training_data[i][1]}")
    
    # return training_data

def train_model(model, training_data, epochs=10):
    states, rewards = zip(*training_data)
    states = np.array(states)
    rewards = np.array(rewards)
    model.fit(states, rewards, epochs=epochs)

def load_training_data():
    try:
        with open('training_data.pkl', 'rb') as f:
            training_data = []
            while True:
                try:
                    training_data.extend(pickle.load(f))
                except EOFError:
                    break
        print("Loaded existing training data.")
        return training_data
    except FileNotFoundError:
        print("No existing training data found.")
        return []

def automatic_play():
    global bird_velocity
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Get state and predict action
        state = get_state(bird, pipes, bird_velocity)
        action = model.predict(state.reshape(1, 4), verbose=0)[0][0]

        # Bird movement based on action
        if action > 0.5:
            bird_velocity = -8
            if hasattr(pygame, 'mixer'):
                jump_sound.play()

        bird_velocity += gravity
        bird.y += bird_velocity

        # Pipe movement
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        # Remove pipes off the screen
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())

        # Collision detection
        if bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1]) or bird.y > HEIGHT:
            if hasattr(pygame, 'mixer'):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes)
        clock.tick(30)

def main():

    # Let user play the game for 5 times to collect training data
    for i in range(5):
        print(f"Playing game {i+1}/5")
        collect_training_data()

    # Load all training data
    all_training_data = load_training_data()

    # Train the model with collected data
    if all_training_data:
        print("Training the model")
        train_model(model, all_training_data, epochs=10)  # Reduce epochs for initial testing

        # Automatic Play with Neural Network
        print("Starting automatic play")
        automatic_play()

if __name__ == "__main__":
    main()





#Pygame Coordinate System
#Origin (0, 0): The top-left corner of the screen.
#X-axis: Increases to the right.
#Y-axis: Increases downward.
#Specifics
#Top-left corner: (0, 0)
#Bottom-left corner: (0, HEIGHT)
#Top-right corner: (WIDTH, 0)
#Bottom-right corner: (WIDTH, HEIGHT)
#Bird's Movement
#Vertical Position (y-coordinate):

#Moving Up: Decrease in the y-coordinate.
#Moving Down: Increase in the y-coordinate.
#Horizontal Position (x-coordinate):

#Typically constant for the bird in Flappy Bird. The bird's horizontal position is usually fixed, and the pipes move horizontally.