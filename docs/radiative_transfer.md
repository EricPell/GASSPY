# Radiative transfer
After generating a database of cloudy models that covers all cells of a given snapshot, one can perform a radiative transfer step. There are two modes that can be used, fast or full spectra.

## Fast radiative transfer
Here the radiative transfer is calculated at the time of ray tracing using a ray_processor. A basic script for this can be found at gasspy/scripts/radiative_transfer/gasspy_radtrans_one_bin.py.
The ray_processor in question is the Single_band_radiative_transfer. It takes as argument a simulation reader to get the mapping from cells to the database, and looks up in the gasspy_config what database to use and which energy ranges to use to construct bins of photons. General structure of script can be seen in the documentation for raytracing.

## Full spectra radiative transfer
Here the radiative transfer is done in a seperate step. First the raytracing is done as per ususal, but rather than computing the ray-cell intersections, it saves all of them into a single hdf5 file using the Raytrace_saver class. After which another class, Trace_processor can be used to take that trace and compute the total flux of all the rays, and include physics such as doppler shifting. 

The Raytrace_saver class works in two steps. Every time a ray has to "dump" its buffer (containing pathlengths,cell index, solid angle, area), the "ray dump" is saved into a set of rotating secondary buffers. Once a secondary buffer has been filled up it is piped to system memory and saved along with all ray-dumps. We use multiple of these secondary buffers (called in the code pipelines) such that the offloading to system memory can happen independently without the code having to stop.

Once the raytracing is done, we have a set of "ray dumps" and their assosiated rays, that we call a "trace". This trace can then be loaded by another radiative transfer class, Trace_processor, which will perform a full spectrum radiative transfer. Here we can perform radiative transfer on thousands of bins, while keeping within memory constraints.

An expample of how to run this can be found in GASSPY/gasspy/scripts/radiative_transfer/gasspy_raytrace_radtrans.py, with the general script structure being

    from gasspy.raytracing.raytracers import Raytracer_AMR_neighbor
    from gasspy.raytracing.ray_processors import Raytrace_saver
    from gasspy.raytracing.observers import observer_plane_class, observer_healpix_class
    from gasspy.radtransfer.rt_trace import Trace_processor
    """
        Load simulation reader
    """
    # initalize observer and raytracer
    observer = observer_plane_class(gasspy_config)
    raytracer = Raytracer_AMR_neighbor(gasspy_config, sim_reader)

    # initalize ray_processor
    ray_processor = Raytrace_saver(gasspy_config, raytracer)

    # set processor and observer
    raytracer.set_ray_processor(ray_processor)
    raytracer.update_observer(observer = observer)

    # Run and save the trace
    raytracer.raytrace_run()
    raytracer.save_trace(trace_file)

    """
        Do some cleanup here... 
    """
    ## Memory management
    cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
    mempool = cupy.get_default_memory_pool()
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    cuda_device = torch.device('cuda:0')

    # initialize trace processor
    mytree = Trace_processor(
        gasspy_config,
        traced_rays=trace_file,
        sim_reader = sim_reader, 
        accel="torch",
        spec_save_name=spec_file,
        cuda_device=cuda_device,
    )
    # load necessary data
    mytree.load_all()

    # process trace
    mytree.process_trace

Note that you need to make sure that all memory is released for the raytracer and ray_processor. 

For now the trace is saved to a file and then loaded. In the future this could possibly be avoided and just kept in memory.

The trace_processor class utilizes PyTorch for many of its functions. In the future we should allow this to work only with CuPy, but for now torch is needed.

## Parameters

    ##################
    # Ray tracing variables
    ##################
    maxMemoryGPU_GB : Target memory utilization of the GPU in GigaBytes. Note: that the code will use more memory 
                      than this, so be conservative.
    maxMemoryCPU_GB : Target memory utilization of the CPU. Note: same caveat as maxMemoryGPU_GB
    
    liteVRAM        : If set to False, will try to put most things on the GPU, only viable for large memory GPU's and small simulations
   
    NcellBuff       : Number of ray-cell intersections to buffer before they are pushed to the ray_processor

    ##################
    # Radiative trasfer parameters
    ##################

    radiative_transfer_precision : precision used in raidative transfer ("double"/"single"), default "double"


    target_segments_per_ray : Rough number of ray segments we expect per ray. If not set will use exact average
                              Can be used to optimize radiative transfer work flow

    doppler_shift : Turn on doppler shifting (default False)

    doppler_shift_est_bin_ratio : Approximate size ratio between neighbouring photon bins. 
                                  Used to estimate memory usage when doppler shifting (default 10) 

    spec_prefix : Optional prefix for the output spectra file

    energy_limits : Specifies bands used for the fast radiative transfer schemer OR the ranges used for the full spectra scheme