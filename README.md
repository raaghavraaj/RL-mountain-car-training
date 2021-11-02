# RL-mountain-car-training
The repository contains the code to train the mountain car using SARSA method.

# Problem Statement
Please head over to the [assignment webpage](https://www.cse.iitb.ac.in/~shivaram/teaching/cs747-a2021/pa-3/programming-assignment-3.html) on the instructor's website.

The state of the car is described using two variables `(x, v)` where `x` is the position of the car in the one-dimensional track and `v` is the velocity.

The script `mountain_car.py` has the following two arguments:
1. `--task`: the options are `T1/T2`
2. `--train`: the options are 0/1 where 1 is for training the weights, 0 is for using the trained weights

Please use a command similar to the following in the `submission` directory to run the script:
```
RL-mountain-car-training/submission $ python3 mountain_car.py --task T1 --train 1
```