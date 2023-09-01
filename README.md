# lighthouse-localization-of-miniature-wireless-robots
This repository holds the code and dataset used to replicate the plots fomr the 2023 paper "Lighthouse Localization of Miniature Wireless Robots"


## Getting Started

### Requirements
To install the required modules, run the following command from the root folder.

`$ pip install -r requirements.txt`

### Run the code

The paper presents 3 different algorithms for 3 different scenarios. The relevant code for each case can be found in the following directories:
- One lighthouse, 2D scene: `oneLH-2D_scene`
- Two lighthouse, 2D scene: `twoLH-2D_scene`
- Two lighthouse, 3D scene: `twoLH-3D_scene`

### Run the tests
From the repository's root folder run the command.

- `$ pytest tests`

