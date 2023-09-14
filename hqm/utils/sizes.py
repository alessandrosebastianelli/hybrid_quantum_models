def size_conv_layer(s : int, kernel_size : int, padding : int, stride : int) -> int:
        '''
            Calculate the size of the image after convolution layer

            Parameters:  
            -----------  
            - s : int  
                integer represeting the size of one axis of the image  
            - kernel_size : int  
                integer represeting the size of the convolutional kernel  
            - padding : int  
                integer represeting the padding size  
            - stride : int  
                integer representing the stride size  
  
            Returns:  
            --------  
            - size : int  
                size after conv2D and Maxpool  
        '''

        size = int(((s - kernel_size + 2 * padding)/stride) + 1)
        return size