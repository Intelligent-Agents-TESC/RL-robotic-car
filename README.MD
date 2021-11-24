# Model-free robotic agent

Reinforcement Learning (RL) is a class of machine learning technique that draws inspiration from psychology. A RL-based autonomous agent discovers an optimal behavior through trial-and-error interactions with its environment. With the right reward shaping, the agent can learn a wide range of behavior, and change its behavior to adapt to a dynamic environment. This makes RL a great solution for behavior-based control problems where the environment might be stochastic, or when the optimal solution is unknown. In this research, we applied RL to solve a robotic control problem in a classical conditioning setting.

## Experiment overview

The goal of this research is to create a robot exhibiting behavior that **maintains its bodily temperature to within a half-degree-Celsius window while avoiding collisions**, without programming the behavior explicitly.

This experiment is set up in a way that forces the agent to use an initially neutral stimulus (i.e. light intensity of a heat lamp) to predict an natural stimulus (i.e. temperature increase from a heat lamp) which triggers a non-negative numerical reward when temperature increases or decreases to a desirable range.

There are 2 key challenges in this learning task:

1. **Learning by association**
    - While light detection by a sensor is immediate, temperature measurement depends on the rate of heat transfer, which takes more time to detect. The agent has to learn to associate increasing light intensity with the likelihood of an increase in temperature.

2. **Curse of dimensionality**
    - To solve complex real-life problems, RL agents often have to operate within state-spaces that are high-dimensional. For example, in our research, our agent has an 8-dimensional state space that is based on sensory inputs and their derivatives. Computing optimal solutions in these high-dimensional spaces often demands enormous memory, computation, and time. 

## Robotic Car Setup

![robot-setup-diagram](/sarsa-lambda/img/robot-set-up-diagram)
<br>
![Robotic agent top view](/sarsa-lambda/img/robot-top.jpeg)
![Robotic agent side view](/sarsa-lambda/img/robot-side.jpeg) 

The robot is a portable computer (Pi 3 Model B) that communicates with a microcontroller (Arduino Uno). The microcontroller collects data from an array of sensors to send to the computer. The computer then computes the current state and action values, and instructs the microcontroller to take an action (by driving the motors) on each time step. 

## Environment Setup

![Arena for experiment](/sarsa-lambda/img/arena.jpeg)

An hexagonal, walled enclosure was built to contain the robotic agent to an area, the walls ensure that the switches at the front of the robot is triggered whenever there is a collision. The enclosure is about 8 feet wide corner to corner. A heat lamp radiates light and heat towards the center of the enclosure.

## Methodology
This code base shows a solution to the problem using a value-based, custom method that is built on Sarsa(λ). Pseudocode for the computer and the microcontroler below gives an overview of the learning process. 

Pseudocode for the Computer:
```
Input: a function F(s, a) returning the indices of active features for s, a Input: a epsilon-greedy policy π
Algorithm parameters: step size α > 0, trace decay rate λ ∈ [ 0, 1] Initialize:
w = ( w1, ..., wd )T ∈ Rd ( e.g. w = 0 ) z = ( z1, ..., zd )T ∈ Rd ( e.g. z = 0 )
Initialize S
z = 0 or load from memory w = 0 or load from memory
choose A ~ max ˆq( S, • , w ) t =0
Loop forever:
    Loop for each step:
        Instruct microcontroller to take action A
        while no feedback from microcontroller: 
            wait
        Observe R, S’ from feedback δ= R
        Loop for i in F(S, A):
            δ = δ - wi
            zi = zi + 1
        Choose A’ ~ max ˆq( S’, • , w )
        Loop for i in F(s’, a’):
            δ = δ + ( γ * wi )
        w = w + ( α * δ * z ) 
        If t % 500 == 0:
            Store w and z into memory z=γ*λ*z
        S = S’, A = A’ t += 1
```
Pseudocode for the Microcontroller:
```
Gather sensory data
Transmit sensory data to the Computer 
While no instruction from the Computer:
    wait
Take action A as instructed
```

The code examples below are taken from the small-feature-space version for simplicity. We will highlight the different techniques used to construct this solution and how they help solve the problem.

- Function approximation
    - Provides a statistical framework of representation for the sample data collected during a trial. We can use function approximation algorithms such as Sarsa(λ) to generalize neighboring states and actions to predict the value of actions in an unfamiliar state. 
    - A condensed overview on how this approach works can be found [here](https://towardsdatascience.com/function-approximation-in-reinforcement-learning-85a4864d566), or chapter 9 of Reinforcement Learning: An Introduction by Sutton & Barto.
- Coarse coding / tile coding
    - Once the states are aggregated, we can further partition the entire state space into different layers, each responsible for part of the value estimate. In a 2D state space, our states can be visualized as rectangular tiles, and each segment of the state space would be layered on top of one another with each layer shifted. We would then make updates to states included in the active tiles on the different layers.
    - For example, the code below shows the shifting behavior across different tilings. Varible `tile_num` declears how many "layers", or tilings, we want to work with. For each feature in our current state, we shift it randomly by -1, 0, or 1 unit. If w denotes the tile width and n the number of tilings, then (w / n) is a fundamental unit for a particular feature.

        ```
         for i in range(tile_num):
            rand = np.random.randint(-1, 2)
            gen_x_i[i, 0] = get_photo_i(bound_photo(avg_photo + (photo_unit * rand)))
            rand = np.random.randint(-1, 2)
            gen_x_i[i, 1] = get_temp_i(bound_temp(temp + (temp_unit * rand)))
            rand = np.random.randint(-1, 2)
            gen_x_i[i, 2] = get_echo_i(bound_uss(uss + (uss_unit * rand)))
            gen_x_i[i, 3] = xp[3]
        ```
    - In the above code block, the get_<feature-name>_i functions return the index of which the shifted data lays in their respective feature dimension. 
-  Eligibility Traces
    - After several time steps of high light intensity, the agent will experience a rise in temperature, as the temperature reaches 25.5°C, it receives a reward. Incorporating an eligibility trace vector (denoted with variable `z` in code) helps us keep track of values that have contributed to our decision making process recently. These values are then eligible to be updated with the newly collected sample data. 
    - Continuing from the last code block, we get our first glimpse at `z`. Here, `z` are getting updated at the feature indices that are part of the current state, this includes all the features on all tilings (some of which we arrived at by randomly shifting by a unit at the previous step).
        ```
        for i in range(tile_num):
                rand = np.random.randint(-1, 2)
                ...
                z[i, gen_x_i[i, 0], gen_x_i[i, 1], gen_x_i[i, 2], gen_x_i[i, 3], a] += 1
        ```
//// TO BE CONTINUED ///
