import pygame
import flappy_bird_gymnasium
import gymnasium as gym

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

state, info = env.reset()
done = False


pygame.init()
screen = pygame.display.get_surface()

while not done:
    action = 0
    # (feed the observation to your agent here)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        elif event.type ==pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1

    state , reward , done , truncated , info = env.step(action)
    env.render()

env.close()
pygame.quit()