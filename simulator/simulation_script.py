import numpy as np


# Constants
room_temp = 23
watts = 225
light = [3, 3]

# Actions and sub actions
actions = [(0,0), (-1, 0), (1, 0), (0, -1), (0, 1)] # Halt, Forward, Back, Left, Right 
sub_actions = [(-1, -1), (1, -1), (-1, 1), (1, 1)]

# Environment
env = [
    [2,2,2,2,2,2,2],
    [2,1,1,1,1,1,2],
    [1,1,0,0,0,1,1],
    [1,0,0,0,0,0,1],
    [1,1,0,0,0,1,1],
    [2,1,1,1,1,1,2],
    [2,2,2,2,2,2,2],
]

def magnitude(vector) -> float:
    """
    args: array
    return: magnitude of array (not length)
    """
    # Total for sum
    total = 0 

    # Run summation of squared values
    for i in range(len(vector)):
        total += vector[i]**2

    return total **.5

def collision(state_val) -> bool:
    """
    args: state value

    return: bool, collided with obstacle
    """
    return state_val == 1

def limit_exceeded(temp) -> float:
    """
    """
    return not (temp >= 25.5 and temp <= 27.5)

def lamp_distance(state) -> float:
    """
    args: State agent is in
    return: float, distance of lamp
    """
    return ((state[0] - light[0])**2 + (state[1] - light[1])**2)**.5

def obstacle_distance(state, walls) -> tuple: 
    """
    args: State agent is in and obstacles nearby
    return: tuple, tuple of arrays - distance, angles, bearings
    """
    # Store length for looping
    length = len(walls)

    # Ignore warning
    np.seterr(invalid='ignore')

    # Distances 
    distance = [(((state[0] - walls[i][0])**2 + (state[1] - walls[i][1])**2)**.5 )/5*100  for i in range(length)]

    # Angles in radians
    angles = [np.arccos(np.dot(state, walls[i])/(magnitude(state)*magnitude(walls[i]))) for i in range(length)]

    # Bearings in degrees
    bearings = [np.arccos(np.dot(state, walls[i])/(magnitude(state)*magnitude(walls[i])))*180/np.pi for i in range(length)]

    return distance, angles, bearings

def create_points(env) -> tuple:
    """
    args: Environment - Hexagon over grid, can be infinite
    
    States: Are one hot encoded tuples/arrays of coordinates, representing state multi-dimensional state.

    return: tuple, dictionary<states, values> and array of obstacles
    """
    # Value function
    v = {}

    # Obstacles reachable by agent
    walls = []

    # size of 2D environment
    length = len(env)

    # Build world
    for i in range(length):
        for j in range(length):
            
            # Accessible by agent
            if env[i][j] == 0:
                v[(i, j)] = 0

            # Obstacle blocking agent
            elif env[i][j] == 1:
                walls.append((i, j))

    return v, walls

def light__temp_intensity(state):
    r = lamp_distance(state)
    r = 1 if r == 0 else r
    temp = room_temp + ((watts * 3.4121 * 1055.06)/1899.1005)/r**2
    temp = 43.0 if temp >= 43.0 else temp
    temp = 23.0 if temp < 23.0 else temp
    return watts/r**2, temp

def reward(temp, collided):

    # Exceed temp limit
    r = 0.0 if limit_exceeded(temp) else -1.0
    
    # Crashed!
    r = r - 1.0 if collided else r
    
    return r

def get_roc(_data, _record, _type):
    delta = 0.0
    for i in range(_record.size):
        delta += (_data - _record[i])
    delta /= float(_record.size)
    if _type == 0:  # photo data
        delta = (delta + 511.50) / 1023.0
    else:  # temp data
        delta = (delta + 10.0) / 20.0
    return delta > 0.5

def simulate(state, action, walls):
    
    next_state = (action[0] + state[0], action[1] + state[1])
    next_state = state if collision(env[next_state[0]][next_state[1]]) else next_state
    intensity = light__temp_intensity(next_state)
    photo = intensity[0]
    temp = intensity[1]
    uss = obstacle_distance(next_state, walls)
    r = reward(env[next_state[0]][next_state[1]], temp)
    return next_state, r

def policy_evaluation(points, delta = 1, theta = 0.001, gamma = 0.9) -> dict:
    """
    args: parameters for policy evaluation for state, encoded state representation of points and walls
    returns: dictionary, value function
    """

    # Value function
    V = points[0]

    # Obstacles
    walls = points[1]

    # Run policy evaluation until delta < theta
    while delta >= theta:

        # Delta <-- 0
        delta = 0

        # Loop for each s in S
        for s in V:

            # v <-- V(s)
            v = V[s]

            # Check each actions for state value 
            value = 0
            for a in actions:
                
                # Simulate a run through the environment
                s_prime, r = simulate(s, a, walls)

                # Sum action values for state value
                value += 1/5*(r + gamma*V[s_prime])

            # Update value function    
            V[s] = value

            # Delta <-- max(Delta, |v - V(s)|)
            delta = max(delta, abs(v - V[s]))

    return V

if __name__ == "__main__":
    ##############################
    # Run Agent Experiments Here #
    ##############################

    points = create_points(env)
    
    print(policy_evaluation(points))
    