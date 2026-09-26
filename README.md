
# Graph Data Structures for Local Search

Experimental evaluation of graph data structures for GPU-based local search in combinatorial optimization.

## Optimization Problems

- Maximum Diversity Problem (MDP)
- Maximum Bisection

## Graph Data Structures

- CSR
- Hash
- WSwap
- CSR with Bitset
- CSR with Binary Search
- CSR with Bitset and Binary Search

## Project Structure

- `src/mdp/`: CUDA implementations of the MDP algorithms.
- `src/maxbisection/`: CUDA implementations of Max Bisection.
- `src/convert_mwis.cpp`: Graph instance converter.
- `src/run_experimentos.py`: Experiment automation.
- `Makefile`: Project compilation and execution.

## Datasets

The experiments use graph instances from the paper:

Dong et al. (2021), *New Instances for Maximum Weight Independent Set From a Vehicle Routing Application*.

Original datasets:
https://registry.opendata.aws/mwis-vr-instances/

The converter generates undirected, edge-weighted graphs from the original conflict graphs and node weights.

## Running Experiments

Run the complete pipeline:

```bash
make tudo
```

Run a specific experiment:

```bash
make experimentos ARGS="--problema mdp --alg csr --instancia MR-D-01 --repeticoes 10"
```

Requirements include Linux, Python 3, GNU Make, GCC/G++, NVIDIA CUDA Toolkit and AWS CLI.
