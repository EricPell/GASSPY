
class gasspy_rt_datum():
    """
    This class unifies api differences between 
    numpy, cupy and torch/tensor.
    check all database and simulation objects.
    Each call should have an equivalent here.
    """
    def __init__(self, lib, data_array, device, take_axis=None, map=None, pinned=False):
        self._lib = lib
        self._device = device
        self.take_axis = take_axis
        self._map = map

        self.data_array = lib.asarray(data_array)

        if self.map != None:
            def take(self, targets):
                return(self.data_array.take(targets, axis=self.take_axis))
        else:
            def take(self,targets):
                return(self.data_array.take(self._map[targets], axis=self.take_axis))

        self.shape = self.data_array.shape

