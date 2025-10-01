# lighthouse-localization-of-miniature-wireless-robots
This repository holds the code and dataset used to replicate the plots from the 2023 paper "Lighthouse Localization of Miniature Wireless Robots"

## Lighthouse v2 Decoding Software

An open-source implementation of the decoding algorithm presented in the paper can be found at the following links. The firmware was written for the nRF52 and nRF53 families of Nordic Semiconductor microcontrollers; it was tested on the nRF52840, nRF52833, and nRF5340 microcontrollers.
- [Header file](https://github.com/DotBots/DotBot-firmware/blob/f3a46c34e9ee9bda8e34b37e943eea6c8cd8e2da/bsp/lh2.h)
- [Implementation](https://github.com/DotBots/DotBot-firmware/blob/f3a46c34e9ee9bda8e34b37e943eea6c8cd8e2da/bsp/nrf/lh2.c)
- [Usage example](https://github.com/DotBots/DotBot-firmware/blob/f3a46c34e9ee9bda8e34b37e943eea6c8cd8e2da/projects/01bsp_lighthouse/01bsp_lighthouse.c)

## Getting Started

### Requirements
To install the required modules, run the following command from the root folder.

`$ pip install -r requirements.txt`

### Run the code

The paper presents 3 different algorithms for 3 different scenarios. The relevant code for each case can be found in the following directories:
- One lighthouse, 2D scene: `oneLH-2D_scene`
- Two lighthouse, 2D scene: `twoLH-2D_scene`
- Two lighthouse, 3D scene: `twoLH-3D_scene`




# Acknowledgement

Part of the source code in this repository is developed within the frame and for the purpose of the OpenSwarm project. This project has received funding from the European Unioan's Horizon Europe Framework Programme under Grant Agreement No. 101093046.

![OpenSwarm - Funded by the European Union](logos/ack.png)
