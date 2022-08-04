# Efficient Optimization of Energy Recovery from Geothermal Reservoirs with Recurrent Neural Network Predictive Models

This work presents an optimization framework for net power generation in geothermal reservoirs using a variant of the recurrent neural network (RNN) as a data-driven predictive model. The RNN architecture is developed and trained to replace the simulation model for computationally efficient prediction of the objective function and its gradients with respect to the well control variables. The net power generation performance of the field is optimized by automatically adjusting the mass flow rate of production and injection wells over 12 years, using a gradient-based local search algorithm. Two field-scale examples are presented to investigate the performance of the developed data-driven prediction and optimization framework. The prediction and optimization results from the RNN model are evaluated through comparison with the results obtained by using a numerical simulation model of a real geothermal reservoir.


<p align="center">
<img src="https://github.com/ZhenQin-USC/ProxyBasedOptimization/blob/main/Image/Workflow.png" width="750" height="650">
</p>



## Prerequisite
Python 3.7.6

TensorFlow 2.1.0

PyDOE2 1.3.0

scipy 1.6.2

## Data
Due to the data confidentiality, we are not allowed to share the field examples or simulation model that were used in our paper. 



## Acknowledgments
This material is based upon work supported by the U.S. Department of Energy's Office of Energy Efficiency and Renewable Energy (EERE) under the Geothermal Technologies Office award number DE-EE0008765. The authors acknowledge the Energi Simulation support of the Center for Reservoir Characterization and Forecasting at USC.

