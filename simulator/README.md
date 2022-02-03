# Simulator

For practical tasks with high dimensional state spaces in the context of reinforcement learning, employing the help of a simulator can offer a way to offset the cost of real-world interaction. Since our goal is to compare the real life and simulated performances of learning algorithms, a simulator that models the actual environment is needed. Our simulator is powered by VPython’s 3D modeling engine and modeled after the hexagonal arena that exists in the physical world. 

The simulator provides a way for the agent to interact with the environment backed by sensory and coordinate data collected in the physical environment. We divided the area with a 5x5 grid and collected data on a per coordinate basis. The data include ambient temperature, change in temperature, and light intensity. These data, used alongside the energy principle help inform a model for the heat lamp which is situated at the center of the arena.

<img src="https://github.com/Intelligent-Agents-TESC/RL-robotic-car/blob/main/simulator/initial%20data/data.png" width="900" height="auto">
