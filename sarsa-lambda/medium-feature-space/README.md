# Feature Dimensions

We are considering 7 variables in the temperature-regulation and obstacle avoidance. 

| Order | Description                         | Range             | State aggregation |
| ----  | ----------------------------------- | ----------------- | ----------------- |
| 1     | phototransistor reading             | 0 - 511.50        | 6                 |
| 2     | rate of change for phototransistor  | -511.50 - 511.50  | 2                 |
| 3     | temperature                         | 23 - 43 °C        | 6                 |
| 4     | rate of change for temperature      | -20 - 20 °C       | 2                 |
| 5     | ultrasonic reading left             | 2 - 62 cm         | 4                 |
| 6     | ultrasonic reading right            | 2 - 62 cm         | 4                 |
| 7     | bumper                              | 0 - 1             | 2                 |


There is a total of 5 availalbe actions in every state.

1. forward
2. backward
3. turn left
4. turn right
5. idle

We implemetned tile coding to generalize across feature space using 4 tilings. If we initialize a multi-dimensional array to store weight vector data with numpy, it might look something like this:

`w = numpy.zeros(shape=(4,6,2,6,2,4,4,2,5))` 
