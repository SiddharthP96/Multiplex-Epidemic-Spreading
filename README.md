# Epidemic Spreading in Group-Structured Populations

## Overview

Our research project explores the dynamics of epidemic spreading within group-structured populations, emphasizing the critical role of group organization in determining the severity of epidemics within common social settings. By focusing on individuals who share physical proximity, such as college students in the same class or dormitory, we've uncovered insights that can significantly impact epidemic control strategies.

## Description

The research findings suggest that reshaping the organization of groups within a population can lead to remarkable results in mitigating the impact of epidemics. Specifically, when group structures exhibit a significant correlation, outbreaks tend to be longer but considerably milder compared to situations with uncorrelated group structures. Additionally, as the correlation among group structures increases, interventions for disease containment become increasingly effective.

## Installation

To get started, simply clone the repository:

```bash
git clone https://github.com/SiddharthP96/Multiplex-Epidemic-Spreading.git
```

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Multiplex stochastic block model function

The `MSBM_COMM` function implements a Multi-layer Stochastic Block Model (MSBM) to simulate epidemic spreading in group-structured populations. This function generates synthetic networks representing social structures, considering two layers with distinct community structures. The layers model families and classes, to understand how the organization of groups within a population affects the severity of epidemics.

#### Parameters:

- `N`: Total number of nodes in the synthetic network.
- `C`: Number of communities in the first layer.
- `div`: Division factor determining the number of communities in the second layer.
- `mu1`, `mu2`: Mixing parameters controlling the probability of intra-community and inter-community connections in each layer.
- `k1`, `k2`: Average degrees for the first and second layers.
- `p_s`: Probability of shuffling nodes between layers, promoting randomness.
- `p_o`: Probability of overlap between layers (default is 1, meaning no overlap).

#### Usage:

```python
result = MSBM_COMM(N, C, div, mu1, mu2, k1, k2, p_s, p_o)
```

#### Output:

The function returns a list containing:

1. A list of two dictionaries representing the adjacency lists of the generated networks for the first and second layers.
2. Normalized Mutual Information (NMI) score, measuring the agreement between the true community assignments in the two layers.
3. Lists specifying the community assignments for nodes in each layer.


### Gillespie multiplex SIR simulation

The `gillespie_sir_sim` function performs a Gillespie simulation of an SIR (Susceptible-Infectious-Recovered) model on a two-layer network. This simulation models the spread of infection in a population with distinct community structures represented by two layers.

#### Parameters:

- `g1`, `g2`: Dictionaries representing the adjacency lists of the first and second layers, respectively. Keys are node-ids, and values are sets of neighbors.
- `b1`, `b2`: Float values representing the infection spread parameters for the first and second layers.
- `mu`: Float value representing the recovery rate.
- `ft`: Full time of the model run, specifying the duration of the simulation.
- `seed`: Node ID for the starting seed value, initiating the infection.
- `p_i`: Decimal value from [0-1] representing the proportion of nodes that start in the recovered state.

#### Usage:

```python
time, susceptible, infectious, recovered = gillespie_sir_sim(g1, g2, b1, b2, mu, ft, seed, p_i)
```

#### Output:

The function returns time-dependent arrays representing the percentage of the population in each compartment (Susceptible, Infectious, Recovered) over the course of the simulation.

- `time`: Array containing the time points at which simulation values are recorded.
- `susceptible`: Array representing the percentage of susceptible individuals in the population at each time point.
- `infectious`: Array representing the percentage of infectious individuals in the population at each time point.
- `recovered`: Array representing the percentage of recovered individuals in the population at each time point.


## Data

Due to confidentiality reasons, the data related to the addresses and majors of the 10,000+ IUB students cannot be shared. If you want to collaborate or replicate the study, please contact us.

## Citation

to add

## Contact

For any inquiries or further assistance, please contact us at siddharthpatwardhan1@gmail.com or vakrao@iu.edu.


## Contributors

- Siddharth Patwardhan
- Varun Rao
- Santo Fortunato
- Filippo Radicchi
