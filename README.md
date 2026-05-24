# Basic Stereo Algorithms Evolution

The basic stereoscopic algorithms have many similarities to each other and can be considered, in a way, that each algorithm is an evolution of another. In this project we created simplified forms of some basic stereoscopic algorithms in MATLAB. The code has been adapted to show the improvement and evolution of an algorithm from the previous one.

## Features

Stereo matching algorithms:

1. **Block Matching**
2. **Dynamic Programming**
3. **Semi-Global Matching**
4. **Belief Propagation (Sequential)**
5. **Belief Propagation (Synchronous)**

The algorithms are implemented in MATLAB.

The algorithms are optimized for performance using matrix operations and other techniques.

## Algorithms

| Number | Name | Implementation |
| --- | --- | --- |
| 1 | Block Matching | **[`stereo1_BM.m`](./stereo1_BM.m)** |
| 2 | Dynamic Programming | **[`stereo2_DP.m`](./stereo2_DP.m)** |
| 3 | Semi-Global Matching | **[`stereo3_SGM.m`](./stereo3_SGM.m)** |
| 4 | Belief Propagation (Sequential) | **[`stereo4_BP1.m`](./stereo4_BP1.m)** |
| 5 | Belief Propagation (Synchronous) | **[`stereo5_BP2.m`](./stereo5_BP2.m)** |

## Installation

Download the project as ZIP file, unzip it, and run the scripts.

## Usage

A stereo matching algorithm works with stereo image pairs to produce disparity maps.
This project contains 5 MATLAB scripts, each implementing a stereo matching algorithm. The files `left.png` and `right.png` contain the stereo image pair used as input.
To use a different stereo pair, replace these two images with your own. In this case, you must also adjust the **disparity levels** parameter in the script you are running.
You may optionally modify other parameters as needed. If the input images contain little or no noise, it is recommended not to use the Gaussian filter.

## Results

Below are the disparity maps produced from the **Tsukuba stereo pair**.

![Tsukuba Left](left.png) ![Tsukuba Right](right.png)

### Block Matching

![Block Matching (SAD) Disparity Map](results/disparity1_BM.png)

### Dynamic Programming

![Dynamic Programming (Left-Right) Disparity Map](results/disparity2_DP.png)

### Semi-Global Matching

![Semi-Global Matching Disparity Map](results/disparity3_SGM.png)

### Belief Propagation (Sequential)

![Belief Propagation (Sequential) Disparity Map](results/disparity4_BP1.png)

### Belief Propagation (Synchronous)

![Belief Propagation (Synchronous) Disparity Map](results/disparity5_BP2.png)

## Links

### Project Repository
- https://github.com/aposb/stereo-algorithms-evolution

### Related Projects
- [Stereo Matching Algorithms in MATLAB and Python](https://github.com/aposb/stereo-matching-algorithms)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
