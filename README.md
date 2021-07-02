# IASSEN_Sector_Design
This code is implementing the simulation used in the following paper that will be presented in ICCCN 2021:

["Guillaume Gaillard and Gentian Jakllari. 2021. Interference-Aware Sector Design in IEEE 802.11ad Networks. In ICCCN21 - 30th International Conference on Computer Communications and Networks."](https://hal.inria.fr/hal-03276187)


## Purpose
Phased-array antennas are configurable to focus their radiation beams toward target destinations.
But in a network, throughput is impaired if one does not also avoid the direction of neighbooring devices, hence trading off interference with strength of received signal at target peers.
The dimension of this problem grows with the number of antenna elements and devices, making exhaustive search too long.

In that context, IASSEN designs network-wise configurations resorting to an heuristic strategy using Simulated Annealing and Gibbs Sampling mechanisms and Python multiprocessing feature.
The code here implements IASSEN's strategy and evaluates its performance over different simulated scenarios, involving TP-Link Talon AD7200 devices, compared with two other strategies in the litterature.

This code also includes various schemes to explore the space of possible configurations. And many possible output representations of resulting radiation patterns, execution process of the strategies, and performance indicators.


## Installation and Requirements
These are all described in [IASSEN-INSTALL.md](IASSEN-INSTALL.md). 

I used Python on Linux.
### Dependencies
I used previous works from Daniel Steinmetzer and Joan Palacios:
* [talon-sector-patterns](https://github.com/seemoo-lab/talon-sector-patterns) from **[Talon Tools: The Framework for Practical IEEE 802.11ad Research](https://seemoo.de/talon-tools/)**, 2017
* [Adaptive-Codebook-Optimization](https://github.com/Joanguitar/Adaptive-Codebook-Optimization) for Mobicom 2018.

## Python scripts
The main folder contains the following files:
 * *param_plot.py*: A library to create graphical representations of radiation patterns for a set of devices' sectors.
 * *sector_explorer.py*: Implementation of various strategies to find better interference-aware network configurations
 * *simu_IASSEN.py*: Simulation engine including comparisons and iterations/repetitions scenarios

Many comments inline should help understanding the code and its options. 
## Contact
Questions and suggestions are welcome :).

[Guillaume Gaillard](https://hal.inria.fr/search/index/?q=%2A&authIdHal_s=guillaumegaillard) [guillaume.gaillard "at" toulouse-inp.fr]
