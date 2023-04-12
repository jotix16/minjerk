# minjerk
A simple python package for calculating minimum jerk trajectories. It provides 4 different functions:

### Multi-Interval Multi-Dim minjerk 
The first two functions compute multi-interval minjerk trajectories for multi-dimensional data. 
They are provided with a list of timepoints, a list of positions and a list of velocities (no accelerations). 
It then computes minjerk trajectories between the given (time, position, velocitiy) truples and returns either the samples or the lambda functions for the trajectories.
No acceleration constraints can be provided, but instead this functions allow you to ask for `smooth_start` (a0=0) and `smooth_traj`(acceleration does not jump) trajectories,.
``` python
get_multi_interval_multi_dim_minjerk_samples(..)
get_multi_interval_multi_dim_minjerk_lamdas(..)
```
### Basic Multi-Dim Minjerk
The second two functions compute minjerk trajectories for multi-dimensional data with a single interval.
In addition to before, providing the start and end accelerations is possible but optional.
``` python
get_min_jerk_trajectory_samples(..)
get_min_jerk_trajectory_lambdas(..)
```

## Installation
Aftere creating and avctivating your virtual environment, install the package with pip:
``` bash
cd .../minjerk
pip install .
```