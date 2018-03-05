# Self Driving Car

'''In this class, we have the map in which the car will be navigating from the upper
left corner of the map (initial state) to the lower right corner (goal state) and vice versa.
We will draw sands on the map that are going to give the agent a negative reward each time
it steps on them. Our reward system will be between 1 and -1. 1 will be given to the agent when
it goes from the initial state to the goal state, -0.1 will be given to the agent when it steps
on the sand, the car will be slowed down, and will get a negative reward of -1. The car will also
be penalized if it goes to the edges of the map, unless it reaches the goal or the initial state
while driving, the car will be getting a -0.2 reward, so we can get the minimum time to get to
our goal state. If the orientation of the car is toward the goal, then the car will be getting a
positive reward of 0.1. When the sensors sense the sand and sends the signal of sand, the car will rotate
by 20 degrees to the opposite direction of the sensor '''

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from the agent in agent.py
from agent import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our Agent, which is called "brain", and that contains the neural network that represents the Q-function
brain = Dqn(7,3,0.9)
action2rotation = [0,15,-15]
last_reward = 0
scores = []

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initializing the last distance
last_distance = 0

# Creating the car class
class Car(Widget):

    # Initialinzing angle: angle between the x axis and the axis of the direction of the car
    angle = NumericProperty(0)
    # Initialinzing rotation: Rotation is the last rotation of the car which can be seen in the action2rotation variable 0, 20, -20 degrees.
    rotation = NumericProperty(0)
    # Initialinzing velocity_x: x coordinate of the velocity vector
    velocity_x = NumericProperty(0)
    # Initialinzing velocity_y: y coordinate of the velocity vector
    velocity_y = NumericProperty(0)
    # Initialinzing velocity: vector of the coordinates, velocity_x and velocity_y
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    # Initializing the x-coordinate of the 1st sensor (front sensor)
    sensor1_x = NumericProperty(0)
    # Initializing the y-coordinate of the 1st sensor (front sensor)
    sensor1_y = NumericProperty(0)
    # 1st sensor vector
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    # Initializing the x-coordinate of the 2nd sensor (left sensor)
    sensor2_x = NumericProperty(0)
    # Initializing the y-coordinate of the 2nd sensor (left sensor)
    sensor2_y = NumericProperty(0)
    # 2nd sensor vector
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    # Initializing the x-coordinate of the 3rd sensor (right sensor)
    sensor3_x = NumericProperty(0)
    # Initializing the y-coordinate of the 3rd sensor (right sensor)
    sensor3_y = NumericProperty(0)
    # 3rd sensor vector
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    # Initializing the x-coordinate of the 4th sensor (right sensor)
    sensor4_x = NumericProperty(0)
    # Initializing the y-coordinate of the 4th sensor (left sensor)
    sensor4_y = NumericProperty(0)
    # 4th sensor vector
    sensor4 = ReferenceListProperty(sensor4_x, sensor4_y)
    # Initializing the x-coordinate of the 5th sensor (right sensor)
    sensor5_x = NumericProperty(0)
    # Initializing the y-coordinate of the 5th sensor (right sensor)
    sensor5_y = NumericProperty(0)
    # 5th sensor vector
    sensor5 = ReferenceListProperty(sensor5_x, sensor5_y)
    # Signal1: signal received by the 1st sensor
    signal1 = NumericProperty(0)
    # Signal2: signal received by the 2nd sensor
    signal2 = NumericProperty(0)
    # Signal3: signal received by the 3rd sensor
    signal3 = NumericProperty(0)
    # Signal3: signal received by the 3rd sensor
    signal4 = NumericProperty(0)
    # Signal3: signal received by the 3rd sensor
    signal5 = NumericProperty(0)

    def move(self, rotation):
        ''' This function will allow the car to go to the left, right, or straight.
            it will be updated in the direction of the velocity vector.'''
        # Update of the position of the car with it's last position
        self.pos = Vector(*self.velocity) + self.pos
        # The rotation that we will get from rotation = action2rotation(action).
        self.rotation = rotation
        # update the angle between the x axis and the axis of the direction of the car
        self.angle = self.angle + self.rotation
        ''' When the car moves, we need to update the sensors and the signals
            30 is the distance between the car and the sensor (between the car and what the car detects)'''
        # Position of sensor 1, which is going to rotate when the car rotates, so we need to update its position.
        self.sensor1 = Vector(25, 0).rotate(self.angle) + self.pos
        # Position of sensor 2, which is going to rotate when the car rotates, so we need to update its position.
        self.sensor2 = Vector(20, 0).rotate((self.angle+30)%360) + self.pos
        # Position of sensor 3, which is going to rotate when the car rotates, so we need to update its position.
        self.sensor3 = Vector(20, 0).rotate((self.angle-30)%360) + self.pos
        # Position of sensor 3, which is going to rotate when the car rotates, so we need to update its position.
        self.sensor4 = Vector(12, 0).rotate((self.angle+100)%360) + self.pos
        # Position of sensor 3, which is going to rotate when the car rotates, so we need to update its position.
        self.sensor5 = Vector(12, 0).rotate((self.angle-100)%360) + self.pos
        '''We get the x coordinates of the sensor, then we take all the cells from -10 to +10 and we do the same for the y coordinate,
            we take the sum of 1s and divide them by 400. This way we will be getting the signals from all the sensors'''
        # Getting the signal received by sensor 1, which is the density of sand around it
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        # Getting the signal received by sensor 2, which is the density of sand around it
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        # Getting the signal received by sensor 3, which is the density of sand around it
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
         # Getting the signal received by sensor 3, which is the density of sand around it
        self.signal4 = int(np.sum(sand[int(self.sensor4_x)-10:int(self.sensor4_x)+10, int(self.sensor4_y)-10:int(self.sensor4_y)+10]))/400.
         # Getting the signal received by sensor 3, which is the density of sand around it
        self.signal5 = int(np.sum(sand[int(self.sensor5_x)-10:int(self.sensor5_x)+10, int(self.sensor5_y)-10:int(self.sensor5_y)+10]))/400.

        # Giving the agent a bad reward when reaching one of the edges of the map
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.
        if self.sensor4_x>longueur-10 or self.sensor4_x<10 or self.sensor4_y>largeur-10 or self.sensor4_y<10:
            self.signal4 = 1.
        if self.sensor5_x>longueur-10 or self.sensor5_x<10 or self.sensor5_y>largeur-10 or self.sensor5_y<10:
            self.signal5 = 1.
# Ball = sensor used by the car (their shape is like a ball). this is taken from kivy
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Ball4(Widget):
    pass
class Ball5(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    ball4 = ObjectProperty(None)
    ball5 = ObjectProperty(None)
    # Car object
    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(9, 0)
    # Update function
    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        # Orientation of the car w.r.t the goal
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        # Last signal of the 3 sensors
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, self.car.signal4, self.car.signal5, orientation, -orientation]
        # The action that the car will have to take at each time to accomplish the goal, which is the output of the neural network
        action = brain.update(last_reward, last_signal)
        # Appending the score (mean of the last 100 rewards to the reward window)
        scores.append(brain.score())
        # Converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        rotation = action2rotation[action]
        # Moving the car according to this last rotation angle
        self.car.move(rotation)
        # Getting the new distance between the car and the goal right after the car moved
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        # Updating the position of the first sensor (ball1) right after the car moved
        self.ball1.pos = self.car.sensor1
        # Updating the position of the second sensor (ball2) right after the car moved
        self.ball2.pos = self.car.sensor2
        # Updating the position of the third sensor (ball3) right after the car moved
        self.ball3.pos = self.car.sensor3
        # Updating the position of the third sensor (ball3) right after the car moved
        self.ball4.pos = self.car.sensor4
        # Updating the position of the third sensor (ball3) right after the car moved
        self.ball5.pos = self.car.sensor5

        # If the car is on the sand
        if sand[int(self.car.x),int(self.car.y)] > 0:
            # It is slowed down (speed = 1), and Reward = -1
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            # It goes to a normal speed = 6, and it gets bad reward = -0.2
            self.car.velocity = Vector(9, 0).rotate(self.car.angle)
            last_reward = -0.3
            # If it is getting close to the goal, it still gets slightly positive reward 0.1
            if distance < last_distance:
                last_reward = 0.2

        # If the car is in the left edge of the frame
        if self.car.x < 10:
            # It is not slowed down
            self.car.x = 10
            # but it gets bad reward -1
            last_reward = -1
        # If the car is in the right edge of the frame
        if self.car.x > self.width-10:
            # It is not slowed down
            self.car.x = self.width-10
            # but it gets bad reward -1
            last_reward = -1
        # If the car is in the bottom edge of the frame
        if self.car.y < 10:
            # It is not slowed down
            self.car.y = 10
            # but it gets bad reward -1
            last_reward = -1
        # If the car is in the upper edge of the frame
        if self.car.y > self.height-10:
            # It is not slowed down
            self.car.y = self.height-10
            # but it gets bad reward -1
            last_reward = -1
        # When the car reaches its goal
        if distance < 100:
            # the goal becomes the bottom right corner of the map, and vice versa (updating of the x coordinate and y coordinate of the goal)
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
        # Updating the last distance from the car to the goal
        last_distance = distance

# Adding the painting tools taken from kivy
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
