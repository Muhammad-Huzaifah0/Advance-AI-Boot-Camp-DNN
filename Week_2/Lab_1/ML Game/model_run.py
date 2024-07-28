import pygame
import random
import numpy as np
from tensorflow.keras.models import load_model

avg_action=[]
# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Flappy Bird AI')

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

# Load the trained model
model = load_model('flappy_bird_model.keras')

def get_state(bird, pipes):
    bird_y = bird.y
    pipe_x = pipes[0][0].x
    top_pipe_height = pipes[0][0].height
    return np.array([bird_y, bird_velocity, pipe_x, top_pipe_height])

# Prediction interval (frames)
prediction_interval = 5
frame_count = 0

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Bird movement
    bird_velocity += gravity
    bird.y += bird_velocity

    # Get the state and predict action
    if frame_count % prediction_interval == 0 or True:
        state = get_state(bird, pipes)
        state = state.reshape((1, 4))
        action = model.predict(state)
        avg_action.append(action)
        print(f"State: {state}, Action: {action}")  # Debug print to check the state and action
        
        if action > 0.08:
            bird_velocity = -8

    # Pipe movement
    for pipe in pipes:
        pipe[0].x += pipe_velocity
        pipe[1].x += pipe_velocity

    # Remove pipes off the screen
    if pipes[0][0].x < -pipe_width:
        pipes.pop(0)
        pipes.append(create_pipe())

    # Collision detection
    if bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1]) or bird.y < 0 or bird.y > HEIGHT:
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

    frame_count += 1

pygame.quit()
ans=np.mean(avg_action)
print(ans)


