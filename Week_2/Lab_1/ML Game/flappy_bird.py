import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
import os

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
bird = pygame.Rect(50, 300, 30, 30)
bird_velocity = 0
gravity = 0.5

# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -3
pipes = []

def create_pipe():
    height = random.randint(100, 400)
    top_pipe = pygame.Rect(WIDTH, 0, pipe_width, height)
    bottom_pipe = pygame.Rect(WIDTH, height + pipe_gap, pipe_width, HEIGHT - height - pipe_gap)
    return top_pipe, bottom_pipe

pipes.append(create_pipe())

# Create a neural network model
model_save_path = 'flappy_bird_model.keras'
if os.path.exists(model_save_path):
    model = load_model(model_save_path)
    print("Loaded existing model.")
else:
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='sigmoid')
])


# Add output layer
    model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))

# Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

def get_state(bird, pipes):
    bird_y = bird.y
    pipe_x = pipes[0][0].x
    top_pipe_height = pipes[0][0].height
    return np.array([bird_y, bird_velocity, pipe_x, top_pipe_height])

def get_reward(bird, pipes):
    if bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1]) or bird.y < 0 or bird.y > HEIGHT:
        return -1  # Collision penalty
    return 0.1  # Small reward for staying alive

def save_training_data(training_data, filename='training_data.npz'):
    if training_data:
        states, rewards = zip(*training_data)
        np.savez(filename, states=states, rewards=rewards)
    else:
        print("No training data to save.")

def load_training_data(filename='training_data.npz'):
    try:
        training_data = np.load(filename, allow_pickle=True)
        states = training_data['states']
        rewards = training_data['rewards']
        return list(zip(states, rewards))
    except FileNotFoundError:
        return []

def train_model(model, epochs=10, training_data_filename='training_data.npz'):
    training_data = load_training_data(training_data_filename)
    if not training_data:
        print("No training data found.")
        return

    states, rewards = zip(*training_data)
    states = np.array(states)
    rewards = np.array(rewards, dtype=np.float32)
    
    model.fit(states, rewards, epochs=50, batch_size=32)
    
    # Save the model
    model.save(model_save_path)
    
    # Save the training data
    save_training_data(training_data, training_data_filename)

# Game loop
running = True
training_data = []
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            bird_velocity = -8

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
    state = get_state(bird, pipes)
    training_data.append((state, reward))

    if reward == -1:
        running = False

    # Clear the screen
    screen.fill(WHITE)

    # Draw the bird
    pygame.draw.rect(screen, BLACK, bird)

    # Draw the pipes
    for pipe in pipes:
        pygame.draw.rect(screen, BLACK, pipe[0])
        pygame.draw.rect(screen, BLACK, pipe[1])

    pygame.display.flip()
    clock.tick(30)

# Save new training data before exiting the game
existing_data = load_training_data()
training_data = existing_data + training_data
save_training_data(training_data)

# Train the model with collected data and save it
train_model(model)

pygame.quit()