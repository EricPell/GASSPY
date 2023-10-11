
# Database building
## Running:
Template script can be found at GASSPY/gasspy/scripts/database_building/gasspy_build_database.py.
The overall class that is used is the DatabaseCreator class. This needs to be coupled to a ModelRunner class that deals with running the actuall models. currently only one ModelRunner is available (gasspy.physics.sourcefunction_database.cloudy.CloudyModelRunner).
Use the gasspy_build_database.py with the approapriate parameters or write your own as

    import gasspy
    from gasspy.physics.sourcefunction_database.cloudy import CloudyModelRunner

    """
        Initialize simulation readers here.
    """

    # initialize the model_runner
    model_runner = CloudyModelRunner(gasspy_config, args.rundir, fluxdef)

    # initialize database creator
    database_creator = gasspy.DatabaseCreator(gasspy_config, model_runner)

    # Add snapshots
    for sim_reader in sim_readers:
        database_creator.add_snapshot(sim_reader)

    # Run models
    database_creator.run_models() 

where gasspy_config is the gasspy configuration, either as a dictionary or as a path to a yaml file, the same with the radiation field definition file fluxdef.  

gasspy_build_database.py is written to allow for mpi4py usage. To parallelize the running of the cloudy models execute using the command 

    mpirun -n $number_of_cores python3 /path/to/gasspy_build_database.py --args 

## Parameters
The parameters taken from the gasspy_config yaml file are given below. For details of syntax and options see GASSPY/gasspy/templates/gasspy_config_all.yaml

    - database_name         : name of the database

    - gasspy_modeldir       : directory to save the database

    - database_fields       : the fields of the simulation to use in the database

    - refinement_fields     : the fields we refine on. Must be a subset of database_fields

    - log10_field_limits    : limits of the fields in log space

    - fields_lrefine        : mininum and maximum refinement levels of the fields

    - convergence_criterions: Limits of the maximal difference between
                              neighboring models, that if exceeded causes refinement.

    - line_labels           : specific Cloudy lines we want to save along with 
                              the emission/absorption spectra. 
                              Mainly used in convergence_criterions.

    - populator_dump_time   : time intervalls at which the code saves the
                              cloudy models to the database. 

    - est_model_time        : estimated model time in seconds. 
                              Used for memory management.

    - max_walltime          : maximum wall time the code runs, 
                              after which it safely exits and allows for restarts.

    - cloudy_ini            : path to the cloudy ini configuration file that
                              sets all the physical parameters common to all 
                              cells (included chemical models, dust physics etc.)

    - cloudy_path : path to the cloudy main directory

