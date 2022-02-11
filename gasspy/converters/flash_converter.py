import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml




class flash_converter:
    def __init__(self, 
            root_dir = "./",
            gasspy_subdir = "GASSPY",
            fields = None
            ):
        self.root_dir = root_dir
        
        assert type(gasspy_subdir) == str, "gasspy_subdir not a string"
        if not gasspy_subdir.endswith("/"):
            gasspy_subdir = gasspy_subdir + "/"
        if not gasspy_subdir[0] == "/":
            gasspy_subdir = "/" + gasspy_subdir

        self.gasspy_subdir = gasspy_subdir
        self.fields = fields


        self.nxb = None
        self.nyb = None
        self.nzb = None

    def read_config(self, flash_in, write_yaml = True, config_yaml_path = None):
        h5file = h5py.File(flash_in, "r")
        config_dict = {}

        # From integer scalars we need nxb, nyb and nzb
        integer_scalars = h5file["integer scalars"][:]
        for i in range(len(integer_scalars)):
            if integer_scalars[i][0][:3] == b"nxb":
                self.nxb = int(integer_scalars[i][1])
                config_dict["nxb"] = self.nxb
            if integer_scalars[i][0][:3] == b"nyb":
                self.nyb = int(integer_scalars[i][1])
                config_dict["nyb"] = self.nyb
            if integer_scalars[i][0][:3] == b"nzb":
                self.nzb = int(integer_scalars[i][1])
                config_dict["nzb"] = self.nzb

        #TODO: CHANGE CODE TO ALLOW FOR BLOCKS
         
        assert np.log2(self.nxb) % 1 == 0 and np.log2(self.nyb) % 1 == 0 and np.log2(self.nzb) % 1 == 0 and self.nxb == self.nyb and self.nxb == self.nzb, "For now only nxb = nyb = nzb = 2**int is allowed."
        self.nib = np.array([self.nxb, self.nyb, self.nzb])
        
        self.block_map = np.zeros((self.nzb, self.nyb, self.nzb, 3))
        self.block_map[:,:,:,0] = np.arange(self.nxb)[np.newaxis, np.newaxis,:]
        self.block_map[:,:,:,1] = np.arange(self.nyb)[np.newaxis, :, np.newaxis]
        self.block_map[:,:,:,2] = np.arange(self.nzb)[:, np.newaxis, np.newaxis]
        
        # From integer runtime parameters we need lrefine_min, lrefine_max
        integer_rtp = h5file["integer runtime parameters"][:]
        for i in range(len(integer_rtp)):
            if integer_rtp[i][0][:11] == b"lrefine_min":
                config_dict["amr_lrefine_min"] = int(integer_rtp[i][1])
            if integer_rtp[i][0][:11] == b"lrefine_max":
                config_dict["amr_lrefine_max"] = int(integer_rtp[i][1])

        # From real runtime parameters we need xmin, xmax, ymin, ymax, zmin, zmax
        # From integer runtime parameters we need lrefine_min, lrefine_max
        real_rtp = h5file["real runtime parameters"][:]
        for i in range(len(real_rtp)):
            if real_rtp[i][0][:4] == b"xmin":
                self.xmin = float(real_rtp[i][1])
            if real_rtp[i][0][:4] == b"xmax":
                self.xmax = float(real_rtp[i][1])

            if real_rtp[i][0][:4] == b"ymin":
                self.ymin = float(real_rtp[i][1])
            if real_rtp[i][0][:4] == b"ymax":
                self.ymax = float(real_rtp[i][1])

            if real_rtp[i][0][:4] == b"zmin":
                self.zmin = float(real_rtp[i][1])
            if real_rtp[i][0][:4] == b"zmax":
                self.zmax = float(real_rtp[i][1])

        # Close the file
        h5file.close()
            
        size_x = self.xmax - self.xmin
        size_y = self.ymax - self.ymin
        size_z = self.zmax - self.zmin

        self.scale_l = min(min(size_x, size_y),size_z)

        # Normalize
        config_dict["sim_size_x"] = size_x/self.scale_l
        config_dict["sim_size_y"] = size_y/self.scale_l
        config_dict["sim_size_z"] = size_z/self.scale_l
        config_dict["sim_unit_length"] = self.scale_l

        if write_yaml:
            if config_yaml_path is None:
                config_yaml_path = self.root_dir+self.gasspy_subdir+"gasspy_config.yaml"
            with open(config_yaml_path, "w") as yaml_file:
                yaml.dump(config_dict, yaml_file, default_flow_style=False)


        return

       
    def convert_snapshot(self, flash_in, gasspy_out, write_yaml = False):
        # If the nxb's havent been set, we havent read the parameters
        if self.nxb is None or self.nyb is None or self.nzb is None:
            self.read_config(flash_in, write_yaml=write_yaml) 

        #Open the snapshot and the file to store the reformatted data
        h5old = h5py.File(flash_in, "r")
        h5new = h5py.File(gasspy_out,"w")

         # Determine which blocks are leafs
        leafBlocks = np.where(h5old["gid"][:,7] == -1)[0]
        #
        # Positional data: We need to change from an 
        #
        bbox = h5old["bounding box"][leafBlocks,:]
        # size of the cells
        dx = (bbox[:,:,1] - bbox[:,:,0])/self.nib[np.newaxis,:]

        posx = (bbox[:,0,0][:, np.newaxis, np.newaxis, np.newaxis] + dx[:,0][:, np.newaxis, np.newaxis, np.newaxis] * self.block_map[np.newaxis,:,:,:,0]).ravel()
        posy = (bbox[:,1,0][:, np.newaxis, np.newaxis, np.newaxis] + dx[:,1][:, np.newaxis, np.newaxis, np.newaxis] * self.block_map[np.newaxis,:,:,:,1]).ravel()
        posz = (bbox[:,2,0][:, np.newaxis, np.newaxis, np.newaxis] + dx[:,2][:, np.newaxis, np.newaxis, np.newaxis] * self.block_map[np.newaxis,:,:,:,2]).ravel()

        h5new.create_dataset("x", data = (posx - self.xmin)/self.scale_l)
        h5new.create_dataset("y", data = (posy - self.ymin)/self.scale_l)
        h5new.create_dataset("z", data = (posz - self.zmin)/self.scale_l)
        
        # create the index1D's
        # create a dx and lrefine per cell
        dxs = np.repeat(dx[:,0], self.nxb*self.nyb*self.nzb).ravel()
        dys = np.repeat(dx[:,1], self.nxb*self.nyb*self.nzb).ravel()
        dzs = np.repeat(dx[:,2], self.nxb*self.nyb*self.nzb).ravel()

        lrefine = np.repeat(h5old["refine level"][leafBlocks], self.nxb*self.nyb*self.nzb)

        Nmax_y = (2**lrefine).astype(int) * self.nyb
        Nmax_z = (2**lrefine).astype(int) * self.nzb
        
        print(Nmax_y.shape, dxs.shape, posx.shape)
        index1D = ((posx - self.xmin)/dxs).astype(int)*Nmax_y*Nmax_z + ((posy-self.ymin)/dys).astype(int) * Nmax_z + ((posz-self.zmin)/dzs).astype(int)
        
        h5new.create_dataset("index1D", data = index1D)
        # TODO: When blocks are supported remove the addion of log2(nxb)
        h5new.create_dataset("amr_lrefine", data = lrefine + int(np.log2(self.nxb)))

        del(posx)
        del(posy)
        del(posz)
        del(dxs)
        del(dys)
        del(dzs)
        del(dx)
        del(index1D)
        del(lrefine)


        #
        # Now start with all the required fields
        #

        # if None are specified grab the entire snapshot
        if self.fields is None:
            fields = h5old["unknown names"][:,0]
        else:
            fields = self.fields

        for field in fields:
            data = h5old[field][leafBlocks,:,:,:].ravel()
            h5new.create_dataset(str(field), data = data)
            del(data)

        h5old.close()
        h5new.close()

"""
To use a flash output in GASSPY we need to change a few things.
GASSPY assumes that the amrgrid follows a simple 2**amr_lrefine grid
whereas flash uses a block structure.

We need to convert these over 
"""

ap=argparse.ArgumentParser()

#---------------outputs-----------------------------
ap.add_argument('f', nargs='+')
ap.add_argument("--root_dir", default = "./")
args=ap.parse_args()

# We wont be needing all of the fields, so only store the duplicates
fields_to_use = ["dens", "temp", "iha ", "ihp ", "ih2 ", "ico ", "icp ", "flge", "fluv", "flih", "fli2"] #velx, vely, velz]

root_dir = args.root_dir


converter = flash_converter(root_dir = root_dir, fields=fields_to_use)
for flash_in in args.f:

    #determine the name of the new file
    iout = int(flash_in[-4:])
    gasspy_out = root_dir+"snapshot_%04d.hdf5"%iout

    converter.convert_snapshot(flash_in, gasspy_out, write_yaml=True)



