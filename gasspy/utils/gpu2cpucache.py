import cupy
import cupyx
import numpy as np

class store(object):
    def __init__(self, buff_size, buff_type, buffer_dump_size, Nswap_buffers = 2, outdiskref=None):
        # Size of buffer in elements
        self.buff_size = buff_size
        # Data type of buffer
        self.buff_type = buff_type

        # size of output array buffer
        self.buffer_dump_size = buffer_dump_size

        # Number of buffers we swap between
        self.Nswap_buffers = Nswap_buffers

        # Allocate swap buffers and output array 
        self.__alloc_in__()
        self.__alloc_out__()

        pass

    def push(self, incoming_data, target="cpu", *args, **kwargs):
        N_incoming = len(incoming_data)
        assert N_incoming <= self.buff_size, "Get fucked, you overflowed the intermediate buffer: Increase the buffer size or send less data"
        #TODO: write a while loop instead of assert failure

        if N_incoming > self.buffer_capcity_avail:
            self.__switchbuffer__()

            if "cpu" in target:
                self.__push2cpu__()
            
            if "directstorage" in target:
                self.__push2directstorage__()

        
        # add to buffer
        self.active_buffer[self.active_buffer_label_index : self.active_buffer_label_index + N_incoming] = incoming_data
        self.buffer_capcity_avail -= N_incoming
        self.current_buffer_index += N_incoming
        pass

    def __push2cpu__(self):
        # figure out end point of current dump in the output_array
        self.next_output_index = self.current_output_index + self.buff_size
        # move the data within the corresponding stream of the previous buffer
        with self.__dict__["stream_%i"%(self.previous_buffer_index)]:
            self.output_array[self.current_output_index: self.next_output_index] = self.__dict__["buffer_%i"%(self.previous_buffer_index)]
            self.__dict__["buffer_%i"%(self.previous_buffer_index)][:] = 0
        # set the next index
        self.current_output_index = self.next_output_index


    def __push2directstorage__(self):
        pass

    def __switchbuffer__(self):
        # Determine the next swap buffer to use
        self.next_swap_buffer_index = (self.active_swap_buffer_index + 1)%self.Nswap_buffers

        # make sure the next stream is synchronized to ensure that it is 
        # done copying its content to system memory before we start reinitialize its values on the GPU 
        self.__dict__["stream_%i"%(self.next_swap_buffer_index)].synchronize()

        # clear the next buffer
        # point active buffer to the new swap_buffer
        self.active_buffer = self.__dict__["buffer_%i"%(self.next_swap_buffer_index)]

        # reset counter
        self.buffer_capcity_avail = self.buff_size
        self.current_buffer_index = 0

        # roll over swap_buffer indices
        self.previous_buffer_index = self.active_swap_buffer_index
        self.active_swap_buffer_index = self.next_swap_buffer_index


    def __alloc_in__(self):
        # (re)set the capacity and current index in the active buffer
        self.buffer_capcity_avail = self.buff_size
        self.active_swap_buffer_index = 0

        # set ut labels for 
        self.stream_labels = np.arange(0, self.Nswap_buffers)
        for i in self.stream_labels:
            self.__dict__["stream_%i"%(i)] = cupy.cuda.stream.Stream()
            self.__dict__["buffer_%i"%(i)] = cupy.zeros(self.buff_size, dtype = self.buff_type)
        
        pass

    def __alloc_out__(self):
        self.output_array = cupyx.zeros_pinned(self.buffer_dump_size, dtype = self.buff_type)
        self.current_output_index = 0
        pass

    def get_output_array(self):
        for i in self.stream_labels:
            self.__dict__["stream_%i"%(i)].synchronize()
        return self.output_array