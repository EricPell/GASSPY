from pathlib import Path
import h5py
import cupy
import torch

class HDF5_SAVE():
    def __init__(self):
        pass

    def open_spec_save_hdf5(self, init_size=0):
        assert isinstance(self.spec_save_name,
                          str), "hdf5 spec save name is not a string...exiting"
        if not self.spec_save_name.endswith(".hdf5"):
            self.spec_save_name += ".hdf5"

        if Path(self.gasspy_spec_subdir).is_dir():
            self.spec_outpath = self.gasspy_spec_subdir+self.spec_save_name
        else:
            self.spec_outpath = self.root_dir+self.gasspy_subdir + \
                self.gasspy_spec_subdir+self.spec_save_name

        self.spechdf5_out = h5py.File(self.spec_outpath, "w")
        self.N_spec_written = 0

        if init_size >= 0:
            init_size = int(init_size)
        else:
            init_size = self.numlib.int(
                self.new_global_rays.cevid[self.new_global_rays.cevid == -1].shape[0])

        self.spechdf5_out.create_dataset(
            "flux", (init_size, len(self.energy)), maxshape=(None, len(self.energy)))
        self.spechdf5_out.create_dataset("x", (init_size,), maxshape=(None,))
        self.spechdf5_out.create_dataset("y", (init_size,), maxshape=(None,))
        self.spechdf5_out.create_dataset(
            "ray_lrefine", (init_size,), dtype="int8", maxshape=(None,))
        if isinstance(self.energy, cupy._core.core.ndarray):
            self.spechdf5_out.create_dataset(
                "E", data=cupy.asnumpy(self.energy))
        elif isinstance(self.energy, torch.Tensor):
            self.spechdf5_out.create_dataset(
                "E", data=self.energy.cpu().numpy())

    def write_spec_save_hdf5(self, new_data, grow=True):
        n_E, n_spec = new_data['flux'].shape

        for key in new_data.keys():
            new_data_shape = new_data[key].shape

            if not grow:
                if len(new_data_shape) == 1:
                    self.spechdf5_out[key][self.N_spec_written:
                                           self.N_spec_written+n_spec] = new_data[key][:]

                elif len(new_data_shape) == 2:
                    self.spechdf5_out[key][self.N_spec_written:self.N_spec_written +
                                           n_spec, :] = new_data[key].T[:]

            else:
                if len(new_data_shape) == 1:
                    self.spechdf5_out[key].resize(
                        (self.spechdf5_out[key].shape[0] + n_spec), axis=0)
                    self.spechdf5_out[key][-n_spec:] = new_data[key][:]

                elif len(new_data_shape) == 2:
                    self.spechdf5_out[key].resize(
                        (self.spechdf5_out[key].shape[0] + n_spec), axis=0)
                    self.spechdf5_out[key][-n_spec:, :] = new_data[key].T[:]

        self.N_spec_written += n_spec

    def close_spec_save_hdf5(self):
        if self.spec_save_type == 'hdf5':
            self.spechdf5_out.close()
        else:
            print(
                "wARNING: You tried to close an output hdf5 file, but are not using hdf5 output.")