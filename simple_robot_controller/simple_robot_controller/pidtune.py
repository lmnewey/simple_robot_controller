import pygame
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class PIDSettingsNode(Node):
    def __init__(self):
        super().__init__('pid_settings_node')
        self.publisher_linear = self.create_publisher(Float32, '/pid_settings/linear', 10)
        self.publisher_angular = self.create_publisher(Float32, '/pid_settings/angular', 10)

        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption('PID Settings')
        self.clock = pygame.time.Clock()

        self.linear_gain = 1.0
        self.angular_gain = 1.0

    def run(self):
        running = True
        while running:
            self.clock.tick(30)
            self.screen.fill((255, 255, 255))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.linear_gain += 0.1
                    elif event.key == pygame.K_DOWN:
                        self.linear_gain -= 0.1
                    elif event.key == pygame.K_LEFT:
                        self.angular_gain -= 0.1
                    elif event.key == pygame.K_RIGHT:
                        self.angular_gain += 0.1

            self.draw_text(f"Linear Gain: {self.linear_gain}", (50, 50))
            self.draw_text(f"Angular Gain: {self.angular_gain}", (50, 100))

            pygame.display.flip()

            self.publish_settings()

        pygame.quit()

    def draw_text(self, text, pos):
        font = pygame.font.SysFont(None, 36)
        text_surface = font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, pos)

    def publish_settings(self):
        linear_msg = Float32()
        linear_msg.data = self.linear_gain
        angular_msg = Float32()
        angular_msg.data = self.angular_gain
        self.publisher_linear.publish(linear_msg)
        self.publisher_angular.publish(angular_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PIDSettingsNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
