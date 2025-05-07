# Multi-Robot-Planning-in-Row-Crop-Fields

This code is linked to the ECAI-PAIS 2025 subsmission 1561 by Simon Ferrier, Alessandro Renzaglia and Olivier Simonin

In this subsmission, a heuristic-based algorithm and and branch-and-bound one are discussed to solve a multi-robot planning in row crop fields without and with energy constraint.

For each case, each approach is coded in a specific py file. To plot an example of solution please run:
- **Heuristic without energy:**
  ```sh
  python3 no_energy_heuristic.py`
  ```
- **Branch-and-bound without energy:**
  ```sh
  python3 no_energy_branch_and_bound.py
  ```
- **Heuristic with energy:**
  ```sh
  python3 energy_heuristic.py
  ```
- **Branch-and-bound with energy:**
  ```sh
  python3 energy_branch_and_bound.py
  ```

In each of these files, you can change the parameters (number of agents, number of rows, ...) in the Parameters part (just before the the main function in the code).
