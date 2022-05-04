from vpython import *
#GlowScript 3.2 VPython

import random

# Change background
scene.background = color.white

# Center scene at
scene.center = vec(2,2,1)

# Environment
floor = box(pos=vector(2,-.2,2), size=vector(5,.2,5), texture=textures.wood)
room_temp = 23.0
room_temp_energy = room_temp * 1899.1005 # Convert Celsius->Joules

# Robot Car Agent
body = box(pos=vector(1,.01,2), size = vec(.1, .025, .05), color=color.cyan)
right_wheel = cylinder(pos=vector(.98,0,2.01), axis=vec(0,0,.01), radius=.02)
left_wheel = cylinder(pos=vector(.98,0,1.98), axis=vec(0,0,.01), radius=.02)
wheels = compound([left_wheel, right_wheel])
agent = compound([wheels, body])
agent.pos = vec(1, -.08 ,2)
agent.velocity = vector(0,0,0)
agent.mass = 0.069
agent.p = agent.mass*agent.velocity
agent.delta_signal = 0
agent.delta_heat_signal = 0
agent.I = vec(0,0,0)
agent.sensor_area = vec(0,.0009,0)
agent.heat_sensor_area = vec(0,.001,0)
agent.signal = vec(0,0,0)
agent.heat_signal = vec(0,0,0)
agent.temp = room_temp
agent.spec_heat = 949.55 # .5 Celsius = 949.55 Joules

# Lamp
lamp = sphere(pos=vector(2,.4,2), radius=0.0616, color=vec(1,0.75,0), emissive=True)
lamp.light = local_light(pos=vector(2,.4,2), radius=0.062, color=color.orange)
lamp.watts = 225 # 10 percent reduction of efficacy
lamp.heat = lamp.watts * 3.4121 * 1055.06 / 3600 # Convert Watts->BTU->Joules

# Intitial conditions and graphics
grey = color.gray(0.8)
black = color.black
w = 2.3
R = 2
d = .3
h = .5
n_slabs = 6
slabs = [None] * n_slabs
deltat = .001
t = 0
oofpez = 9e9

# RL Parameters
epsilon = 0.01
lambda_decay = 0.8
alpha = 0.025
gamma = 0.9

# units per metric: width / tile_num
photo_unit = 2.575     # range of 511.50, from 0 - 511.50
temp_unit = 0.25      # range of 20, from 23 - 43 C
uss_unit = 0.5       # range of 60, from 2 - 62 cm

# Build walls
for i in range(n_slabs):
    theta = i*2*pi/n_slabs
    c = cos(theta)
    s = sin(theta)
    xc = R*c
    zc = R*s
    slab = box(pos=vec(xc+2, -.02, zc+2),axis=vec(c,0,s),size=vec(d,h,w),color=black)
    slabs[i] = slab

# Create side lengths
side = w - d

# Create hexagon from wall slabs
hexagon = compound(slabs)

# Reward Function
def get_reward(collision, limit_exceeded):

    # Exceed temp limit
    r = 0.0 if limit_exceeded else -1.0

    # Crashed!
    r = r - 1.0 if collision else r

    return r


scene.pause()
while True:
    rate(100)

    # Relative position vector
    r = agent.pos - lamp.pos

    # Direction vector
    r_hat = hat(r)

    # Distance of displacement
    r_mag = mag(r)

    # Calculate light intensity at agent’s location
    agent.I = (lamp.watts / r_mag**2 ) * r_hat

    # light signal from environment received by agent
    agent.signal = dot(-agent.I, agent.sensor_area)

    # Calculate change in light intensity
    agent.delta_signal = agent.delta_signal - agent.signal

    # Calculate thermal energy at agent’s location
    state_temp = (room_temp_energy + (lamp.heat / r_mag**2 )) * r_hat

    # Heat signal from environment received by agent
    agent.heat_signal = dot(-state_temp, agent.heat_sensor_area)

    # Calculate change in thermal energy
    agent.delta_heat_signal = agent.delta_heat_signal - agent.heat_signal

    # Check if change in temp (in joules) is in range
    if agent.spec_heat < abs(agent.delta_heat_signal):

        # phi = acos(dot(agent.pos, lamp.pos)/(mag(agent.pos)*mag(lamp.pos)))
        phi = random.choices([-pi/3, pi/3, 0], weights=[2, 2, 80], k=1)[0]
        dx = sin(phi)
        dz = dx
        new_unit_vec = vec(r_hat.x + dx,0,r_hat.z + dz)

        # The Momentum Principle
        agent.p = agent.p + agent.heat_signal/r_mag * -new_unit_vec * deltat

        agent.rotate(phi, axis = vec(0,1,0))


        # Position Update
        agent.pos = agent.pos + (agent.p/agent.mass)*deltat

        # Reset change in signals
        agent.delta_signal = 0
        agent.delta_heat_signal %= 60

    if not (-side+w < agent.pos.x < side+R-d):
        agent.p.x = -agent.p.x
        print("Crashed! on x")

    if not (-side+w < agent.pos.z < side+R-d):
        agent.p.z = -agent.p.z
        print("Crashed! on z")

    t = t + deltat
