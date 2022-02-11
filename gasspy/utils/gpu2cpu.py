import cupy
import cupyx
import numpy as np

class pipeline(object):
    def __init__(self, buff_NraySegs, buff_type, NcellPerRaySeg, varname, traced_rays, Nswap_buffers = 3, outdiskref=None):
        # Number of ray segments stored in a buffer
        self.buff_NraySegs = buff_NraySegs
        
        # Number of cells one ray segment consists of
        self.NcellPerRaySeg = NcellPerRaySeg

        # Data type of buffer
        self.buff_type = buff_type
        
        # Number of buffers we swap between
        self.Nswap_buffers = Nswap_buffers

        # Allocate swap buffers
        self.__alloc_in__()

        # Initialize the active_buffer pointer to the first swap buffer
        self.active_buffer = self.__dict__["buffer_%i"%(self.stream_labels[0])]
        # Intialize the index in the active swap buffer
        self.current_buffer_index = 0

        # save pointer to the traced rays object
        self.traced_rays = traced_rays

        # variable name associated with the pipe
        self.varname = varname

        # Counter of how many raysegments have been pushed to the traced rays output object
        self.current_output_index = 0
        pass

    def push(self, incoming_data, target="cpu", *args, **kwargs):
        # Get incoming data length to buffer
        N_incoming = len(incoming_data)

        # Check that the size of incoming data is smaller than total allocated buffer, or throw an exception
        assert N_incoming <= self.buff_NraySegs, "Get fucked, you overflowed the intermediate buffer: Increase the buffer size or send less data"
        #TODO: write a while loop instead of assert failure

        # If incombing data is greater than available buffer capacity, dump the buffer data
        if N_incoming > self.buffer_capcity_avail:
            self.__switchbuffer__()

            if "cpu" in target:
                self.__push2cpu__()
            
            if "directstorage" in target:
                self.__push2directstorage__()

        
        # Insert incoming data into the buffer
        self.active_buffer[self.current_buffer_index : self.current_buffer_index + N_incoming, :] = incoming_data[:,:]
        # reduce available buffer capcity
        self.buffer_capcity_avail -= N_incoming
        # Set next available buffer index for next push
        self.current_buffer_index += N_incoming
        pass

    def __push2cpu__(self):
        # figure out end point of current dump in the output_array
        self.next_output_index = self.current_output_index + self.previous_buffer_size
        # move the data within the corresponding stream of the previous buffer
        with self.__dict__["stream_%i"%(self.previous_buffer_index)]:
            # Here in the swap buffers dedicated stream we call the transfer of the data to the traced_rays cpu memory.
            # In the same stream/queue we also reinitalize the buffer, with an order such that
            # the copy will finish, then the reinitialization will occur, making the buffer read.
            # TODO: what do we do if self.next_output_index is greater than the output_array? Append?
            self.traced_rays.data_from_cupy(
                                            self.varname,
                                            self.__dict__["buffer_%i"%(self.previous_buffer_index)][:self.previous_buffer_size],
                                            self.current_output_index,
                                            self.previous_buffer_size,
                                            stream = self.__dict__["stream_%i"%(self.previous_buffer_index)]
                                            )
            self.__dict__["buffer_%i"%(self.previous_buffer_index)][:,:] = 0
        # set the next index
        self.current_output_index = self.next_output_index


    def __push2directstorage__(self):
        pass

    def __switchbuffer__(self):
        # LOKE DEBUG: save the size of the last buffer (discuss ordering of push2cpu and switch buffer....)
        self.previous_buffer_size = self.current_buffer_index
        
        # Determine the next swap buffer to use
        self.next_swap_buffer_index = (self.active_swap_buffer_index + 1)%self.Nswap_buffers

        # make sure the next stream is synchronized to ensure that it is 
        # done copying its content to system memory and reinitialized on the GPU 
        # See: __push2cpu__
        self.__dict__["stream_%i"%(self.next_swap_buffer_index)].synchronize()

        # change the refernce of the active-buffer to the next swap-buffer-pointer
        self.active_buffer = self.__dict__["buffer_%i"%(self.next_swap_buffer_index)]

        # reset the "active-swap-buffer" element/index/position and capacity.
        self.buffer_capcity_avail = self.buff_NraySegs
        self.current_buffer_index = 0

        # roll over swap_buffer indices
        self.previous_buffer_index = self.active_swap_buffer_index
        self.active_swap_buffer_index = self.next_swap_buffer_index


    def __alloc_in__(self):
        # (re)set the capacity and current index in the active buffer
        self.buffer_capcity_avail = self.buff_NraySegs
        self.active_swap_buffer_index = 0

        # Create stream labels 
        self.stream_labels = np.arange(0, self.Nswap_buffers)

        # Create streams and initialize swap buffer
        for i in self.stream_labels:
            self.__dict__["stream_%i"%(i)] = cupy.cuda.stream.Stream()
            self.__dict__["buffer_%i"%(i)] = cupy.zeros((self.buff_NraySegs, self.NcellPerRaySeg), dtype = self.buff_type)
        
        pass

    def pushBuffer(self, target = "cpu"):
         # If there are any slots occupied, send these to the cpu
        if self.buffer_capcity_avail < self.buff_NraySegs:
            self.__switchbuffer__()
        
            if "cpu" in target:
                self.__push2cpu__()
            
            if "directstorage" in target:
                self.__push2directstorage__()       

    def finalize(self, target = "cpu"):
        self.pushBuffer(target=target)
        for i in self.stream_labels:
            self.__dict__["stream_%i"%(i)].synchronize()
                    

'''
    NO LONGER USED HERE
    def __alloc_out__(self):
        # Create a numpy array on pinned system memorry that the GPU will write to directly
        self.output_array = cupyx.zeros_pinned(self.buffer_dump_size, dtype = self.buff_type)
        # Current element/index/position of data in the pinned memory array.
        self.current_output_index = 0
        pass

    def get_output_array(self):
        # Make sure that the active swap buffer pushes its data to the cpu
        # LOKE DEBUG: need to tell the last buffer to start swapping
        self.__switchbuffer__()
        self.__push2cpu__()
        # Before we can return the output array, make sure that all streams are done writing to it.
        for i in self.stream_labels:
            self.__dict__["stream_%i"%(i)].synchronize()
        return self.output_array[:self.current_output_index]
'''
