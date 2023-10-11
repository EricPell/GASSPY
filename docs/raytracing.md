# Usage
To use the raytracing of GASSPY there are three classes:
- raytracer: The main class that advances the rays through the cells of the simulation
- observer: A class that defines the source of the rays (eg. an observer)
- ray_processor: A class that determines how each ray-cell intersection is handled.

There are multiple example scripts of how this can be used, mainly GASSPY/gasspy/scripts/radiative_transfer/gasspy_raytrace_radtrans.py and GASSPY/gasspy/scripts/radiative_transfer/gasspy_radtrans_one_bin.py. 

The general structure for a script would be

    """
        Initialize simulation reader here
    """

    # Initialize observer
    observer = observer_plane_class(gasspy_config)

    # Initialize raytraces
    raytracer = Raytracer_AMR_neighbor(gasspy_config, sim_reader)

    # Initialize ray_processor
    ray_processor = Single_band_radiative_transfer(gasspy_config, raytracer, sim_reader)

    # set ray_processor and observer in raytracer class
    raytracer.set_ray_processor(ray_processor)
    raytracer.set_observer(observer)

    # run ray trace
    raytracer.raytracer_run()

After which one can acces the output of the raytrace from the ray_processor and raytracer.
# raytracer class:
The raytracer is an adaptive raytracer that works assuming and AMR mesh of cells. It works by first determening the starting cell of the rays (if the observer is outside of the simulation domain, the rays are traced to the edge of the domain). Within that cell, the ray is traced to the approapriate face of the cell. Given the face, and where on the face the ray leaves, the next cell that the ray enters can be determined from the list of neighboring cells. The cells are then traced untill the ray hits one of its termination criterion, which are that either that the ray leaves the simulation domain, or that the area that the ray covers becomes to large compared to the surface area of the cell, at which the ray splits. 

After a ray terminates, or a user-defined number of cells have been traced, it is passed of to the ray_processor.

Currently only one raytracer is supported, Raytracer_AMR_neighbor. 

# observer class:
There are currently two observer available. A plane observer (parallel rays shot out from a plane of pixels) and a healpix observer (rays shot out from a point according to a healpix decomposition). The observers are generally defined from a position (observer_center) and a target direction (pov_center). The observers coordinate frame is then centered on observer_center with its "z" axis being parallel to the vector from observer_center to pov_center.

## Plane observer
The plane observer takes as arguments the extent of the plane within the simulation domain in units of the box size (eg. between 0 and 1) via the parameters detector_size_x and detector_size_y. x and y here is not in the coordinate frame of the simulation but of the observer (eg, rotated). The inital size of the rays are given such that 
$$
\Delta x_\mathrm{ray} = \mathrm{detector\_size\_x}\times2^{-\mathrm{ray\_lrefine\_min}}\\
\Delta y_\mathrm{ray} = \mathrm{detector\_size\_y}\times2^{-\mathrm{ray\_lrefine\_min}}.
$$ 
The rays are then uniformly distributed across the plane. Note that if detector_size_x and detector_size_y differ, the resulting rays will cover rectangular areas with $\Delta x\neq\Delta y$.

## Healpix observer
The healpix observer does not need any extra arguments. The initial set of rays are taken from the healpix level of ray_lrefine_min. The healpix decomposition assumes an nested ordering. As an optional argument one can pass min_pix_lmin and max_pix_lim to specify a range of pixels to use as the inital rays (See healpix documentation).

When refining we use the NUNIQ packaging scheme found in https://ivoa.net/documents/MOC/. Each pixed (or ray) is then described by a unique number defined as 
$$
uniq = 4\times nside^2 + ipix.
$$
which is saved along the ray for future reconstruction. We recommend using the [mhealpy](https://mhealpy.readthedocs.io) package to visualize the rays.

# Parameters
    
    ##################
    # Ray tracing variables
    ##################
    maxMemoryGPU_GB : Target memory utilization of the GPU in GigaBytes. Note: that the code will use more memory 
                      than this, so be conservative.
    maxMemoryCPU_GB : Target memory utilization of the CPU. Note: same caveat as maxMemoryGPU_GB
    
    liteVRAM        : If set to False, will try to put most things on the GPU, only viable for large memory GPU's and small simulations
   
    NcellBuff       : Number of ray-cell intersections to buffer before they are pushed to the ray_processor

    no_ray_spitting : If set to True (default False) stops the rays from refining. THIS WILL NOT BE ACCURATE.

    ray_max_area_frac : Maximum fraction of a cells face area that a given ray can cover. If exceeded the ray refines. 

    ray_lrefine_min : Minimum refinement level of the rays, definition of which depends on the observer
    
    ray_lrefine_max : Maximum refinement level of the rays.

    ##################
    # Observer parameters
    ##################

    observer_type : used in scripts to determine which observer to use
   
    detector_size_x : size of detector plane in its "x" direction (according to its coordinate frame)

    detector_size_y : size of detector plane in its "y" direction (according to its coordinate frame)

    observer_center : Position of the observer in the coordinate frame of the simulation

    pov_center : position of the observer target in the coordinate frame of the simulation

    # specific to plane observer

    external_distance : assumed distance of the plane from the observer_center to the "real observer". Used for calculating solid angles.

    # specific to healpix observer
    min_pix_lmin : starting ipix to use to generate initial rays (at level ray_lrefine_min). Default 0
 
    max_pix_lmin : ending ipix to use to generate initial rays (at level ray_lrefine_min). 
                   Default to number of pixels at the starting refinement level