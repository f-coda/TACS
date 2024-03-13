# TACS: Trajectory Analysis via Compression and Similarity

## Basic Info

This repository provides a wide range of several well-known **trajectory compression algorithms** and evaluate their performance on data originating from **vessel trajectories**. Trajectory compression algorithms included in this research are suitable for either historical data (**offline** compression) or real-time data streams (**online** compression). The performance evaluation is three-fold and each algorithm is evaluated in terms of compression ratio, execution speed and information loss.

Τhe quality of the compression is evaluated by employing **similarity measures** over both the original and the compressed trajectory data, in order to evaluate their accuracy. The purpose of a similarity measure is to obtain a quantitative measure between any two trajectories, thus to identify to what extent two objects are similar.

### Offline Compression Algorithms

 - Douglas-Peucker (DP)
 - Top-down Time-Ratio (TR)
 - Speed-based (SP)
 - Heading-based (HD)
 - Time-Speed-Heading-based (TSH)

### Online Compression Algorithms

 -  Spatiotemporal Trace – STTrace (STT)
 - Dead-Reckoning (DR)

### Trajectory Similarity Measures

 - Dynamic Time Warping (DTW)
 - Edit distance with Real Penalty (ERP)
 - Fréchet distance
 -  Discrete Fréchet
 - Hausdorff
 - Symmetrized Segment-Path Distance (SSPD)

## Dataset Format Dependency

One significant drawback of the code implementation is its strict dependency on the format
of the dataset (data_test.csv). The code is designed to work with this specific dataset format (columns),
and any deviation from this format may result in errors or unexpected behavior. 
Therefore, it is crucial to ensure that the dataset conforms precisely to the expected 
format for the code to execute correctly.

## Threshold definition

One of the challenges is to define the required thresholds to be employed by the compression algorithms. Setting the proper
threshold can significantly affect the compression results in terms of compression ratio and achieved quality.

Each trajectory is different, hence it is beneficial to select the appropriate threshold for each trajectory automatically.

In general, each of the examined algorithms follows the steps below:

- group the points and create a trajectory of each object
based on an identifier. This practically means that the
number of trajectories in a dataset is as large as the
number of objects
- compress the whole trajectory of each identifier
- write the points remaining after compression to a file
grouped by identifiers

In order to determine the threshold that each algorithm will use, **a dynamic process** is proposed: for each
individual trajectory a different threshold is automatically
defined based on an **average**. Specifically, the average value refers to the discarding criterion of each algorithm. 

For example, in case of DP the average epsilon (ε) is calculated
while for STTrace two thresholds are calculated, the average speed and the average orientation. 
Then we use this average value as a reference point to define a common
rule for the threshold calculation. 

This practically means
that in every trajectory of each dataset a different threshold
is applied in the corresponding algorithm, which depends
on the actual features and peculiarities of this trajectory.
Thus, we have eliminated the need of arbitrary user-defined
thresholds.


## Usage  
  
### Compression
 
 1. Download and unzip project  
 2. ```cd $PROJECTPATH```   
 3. ```chmod u+x compression_job.sh``` to make the script executable   
 4. ```sh compression_job.sh```  and follow the instructions (```sh compression_job.sh -h``` for more information)
    
   
### Similarity  
  
```python3 similarity.py path/to/the/original/dataset path/to/the/compressed/dataset algorithm_name```   

###### <em>Note: The algorithm_name parameter is arbitrary and is exclusively used for generating a distinct result file containing the aggregated similarity results of each algorithm.</em>


## Cite Us

If you use the above code for your research, please cite 
our paper entitled ["A Comparison of Trajectory Compression Algorithms Over AIS Data"](https://ieeexplore.ieee.org/document/9466112):

    @article{makris2021comparison,  
      title={A comparison of trajectory compression algorithms over AIS data},  
      author={Makris, Antonios and Kontopoulos, Ioannis and Alimisis, Panagiotis and Tserpes, Konstantinos},  
      journal={IEEE Access},  
      volume={9},  
      pages={92516--92530},  
      year={2021},  
      publisher={IEEE}  
    }

Related articles:

- [Evaluating the effect of compressing algorithms for trajectory similarity and classification problems](https://link.springer.com/article/10.1007/s10707-021-00434-1)