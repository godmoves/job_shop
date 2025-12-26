# Job Shop Scheduling Problem

This repository contains an Ant Colony Optimization (ACO) algorithm for solving the Job Shop Scheduling Problem (JSSP).

## Quick Start

```bash
pip install numpy matplotlib
python test.py
```

## Performance (Case 4, 500 epochs)

The existing ACO algorithm achieves excellent results:
- **Best time: 125,300** (500 epochs with 200 ants/epoch)

## Bug Fix

Fixed typos in `genetic_algo.py`: `np.randon` → `np.random`

## Files

- `ants_algo.py` - Ant Colony Optimization implementation
- `genetic_algo.py` - Genetic Algorithm framework (typos fixed)
- `case_data.py` - Problem instances and parameters
- `timer.py` - Solution evaluation
- `test.py` - Test script

## Algorithm Details

The ACO implementation uses:
- 200 ants per epoch
- Pheromone evaporation rate: 0.95
- Lambda (pheromone deposit): 2000.0

The algorithm achieves high-quality solutions through proper balance of exploration and exploitation.
