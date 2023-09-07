# Dataset Management

This folder holdes the data from the experiment oneLH_2d_scene experiment, as well as any script needed to parse it.

This experiment consisted of a camera and a lighthouse tracking a rotbot as it moved through the floor.

## Getting Started

There is nothing to do here, all the data has been parsed.

## Understanding the Data

A total of 6 individual experiments were run with trajectories of different shapes

### Ground Truth

The video from the camera was manually tracked in Blender, the results are in the `ground_truth/*_blender.csv`.
The files contain the following columns:
- ``timestamp`: Real timestamp of the datapoint, synchronized with the lighthouse data
- `frame`: frame of the video where that data point was recorded.
- `x`: X - position of the robot in the video frame, in pixels.
- `y`: Y - position of the robot in the video frame, in pixels.


### lighthouse data

The lighthouse data was captured with the [PyDotBot](https://github.com/DotBots/PyDotBot) software. The logs from this application were parsed using the `lighthouse/parse_log.py` to create the files `lighthouse/*_lh2.csv`

The files contain the following columns:
- ``timestamp`: Real timestamp of the datapoint, synchronized with the ground truth data
- `x`: X - position of the robot in the LH frame, unitless [0,1].
- `y`: Y - position of the robot in the LH frame, unitless [0,1].
- `c0`: raw bit count of the first sweep of the lighthouse
- `c1`: raw bit count of the second sweep of the lighthouse





