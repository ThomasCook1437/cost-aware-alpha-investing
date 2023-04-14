# cost-aware-alpha-investing
Supporting code for Cost-aware Generalized α-investing for Multiple Hypothesis Testing (https://arxiv.org/abs/2210.17514)

To replicate results for Table 2:
    - cd 'Table 2'
    - run first cell in make_db.ipynb to create a db file to store results
    - run single_iter.py for k in [1,...,10000].
        - run_sim.sh can distribute this across a cluster using SLURM
    - Run cells in process_results.ipynb to generate table.
    
The file results_in_paper.db is the db file used from our run of this simulation. You can edit the path of the file being loaded in process_results.ipynb to process these results without running the simulation.