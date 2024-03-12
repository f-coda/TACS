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

If you use the above code for your research please cite our paper

    @article{makris2021comparison,  
      title={A comparison of trajectory compression algorithms over AIS data},  
      author={Makris, Antonios and Kontopoulos, Ioannis and Alimisis, Panagiotis and Tserpes, Konstantinos},  
      journal={IEEE Access},  
      volume={9},  
      pages={92516--92530},  
      year={2021},  
      publisher={IEEE}  
    }