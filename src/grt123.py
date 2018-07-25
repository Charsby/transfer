from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv3D
from keras.layers import AveragePooling3D
from keras.layers import MaxPooling3D
from keras.layers import Conv3DTranspose
from keras.layers import Dropout
from keras.layers import Permute
from keras.layers import Concatenate
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras import layers

  
class GRT123Net():
    def __init__(self):
        """Loss functions.
        # Arguments
            num_blocks_forw: Number of forward res blocks
            num_blocks_back: Number of back res blocks
            
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        """
        self.num_blocks_forw = [2,2,3,3]
        self.num_blocks_back = [3,3]
        self.num_channel_forw = [24,32,64,64,64]
        self.num_channel_back =  [128,64,64]

    def pre_block(self, input_tensor, input_channel, output_channel, strides=(1, 1, 1)):
        """
        # Arguments
            input_tensor: input tensor
            input_channel: input channel
            output_channel: output channel
            strides: stride
        # Returns
            Output tensor for the block.
        """
        x = input_tensor
        x = Conv3D(filters = output_channel, kernel_size = (3, 3, 3), 
            strides = strides, padding="same", kernel_initializer = "random_normal", name = 'preBlock.0', data_format='channels_first')(x)
        
        x = BatchNormalization(axis = 1, momentum=0.1, name = 'preBlock.1')(x)
        out = Activation('relu', name = 'preBlock.2')(x)
        x = Conv3D(filters = output_channel, kernel_size = (3, 3, 3), 
            strides = strides, padding="same", kernel_initializer = "random_normal", name = 'preBlock.3', data_format='channels_first')(out)
        x = BatchNormalization(axis = 1,momentum=0.1, name = 'preBlock.4')(x)
        x = Activation('relu', name = 'preBlock.5')(x)

        return x  
    
    def res_unit(self, input_tensor, input_channel, output_channel, o_name, strides=(1, 1, 1)):
        """
        # Arguments
            input_tensor: input tensor
            input_channel: input channel
            output_channel: output channel
            strides: stride
        # Returns
            Output tensor for the block.
        """

        if strides != (1, 1, 1) or output_channel != input_channel:
            shortcut = True
        else:
            shortcut = False

        if shortcut:
            residual = Conv3D(filters = output_channel, kernel_size = (1, 1, 1),
            strides = strides, kernel_initializer = "random_normal", name = o_name+'.shortcut.0',data_format='channels_first')(input_tensor)
            residual = BatchNormalization(axis = 1, momentum=0.1, name = o_name+'.shortcut.1')(residual)

        else:
            residual = input_tensor

        x = input_tensor
        x = Conv3D(filters = output_channel, kernel_size = (3, 3, 3), 
            strides = strides, padding="same", kernel_initializer = "random_normal", name = o_name+'.conv1',data_format='channels_first')(x)
        x = BatchNormalization(axis = 1, momentum=0.1, name = o_name+'.bn1')(x)
        x = Activation('relu')(x)
        x = Conv3D(filters = output_channel, kernel_size = (3, 3, 3), 
            strides = strides, padding="same", kernel_initializer = "random_normal", name = o_name+'.conv2',data_format='channels_first')(x)
        x = BatchNormalization(axis = 1, momentum=0.1, name = o_name+'.bn2')(x)
        x = layers.add([x, residual])
        x = Activation('relu')(x)

        return x
    
    
    def res_block(self, num_units, input_tensor, input_channel, output_channel, name, strides=(1, 1, 1)):
        """
        # Arguments
            num_units: num of unit in the block
            input_tensor: input tensor
            input_channel: input channel
            output_channel: output channel
            strides: stride
        # Returns
            Output tensor for the block.
        """
        for i in range(num_units):
            if i == 0:
                x = self.res_unit(input_tensor, input_channel, output_channel, name+str(i))
            else:
                x = self.res_unit(x, output_channel, output_channel, name+str(i))
        return x
    
    def deconv_block(self, input_tensor, input_channel, output_channel, o_name, strides = (2, 2, 2)):
        """
        # Arguments
            input_tensor: input tensor
            input_channel: input channel
            output_channel: output channel
            strides: stride
        # Returns
            Output tensor for the block.
        """
        x = Conv3DTranspose(output_channel, kernel_size = (2, 2, 2), strides = strides, kernel_initializer = "random_normal", name = o_name+'.0',data_format='channels_first')(input_tensor)
        x = BatchNormalization(axis = 1, momentum=0.1,  name = o_name+'.1')(x)
        x = Activation('relu')(x)
        return x
    
    
    def output_block(self, input_tensor, input_channel, output_channel):
        """
        # Arguments
            input_tensor: input tensor
            input_channel: input channel
            output_channel: output channel
        # Returns
            Output tensor for the block.
        """
        x = Conv3D(filters = 64, kernel_size = (1, 1, 1), kernel_initializer = "random_normal", name = 'output.0',data_format='channels_first')(input_tensor)
        x = Activation('relu')(x)
        x = Conv3D(filters = output_channel, kernel_size = (1, 1, 1), kernel_initializer = "random_normal", name = 'output.2',data_format='channels_first')(x)
        return x
    
    def get_model(self, input_tensor, input_shape, input_channel, output_channel, coord = None, coord_shape = (3,32,32,32)):
        """
        # Arguments
            input_tensor: input tensor
            input_shape: input shape of the tensor
            input_channel: input channel
            coord: coord tensor
            coord_shape: coord tensor shape
        # Returns
            Output tensor for the block.
        """
        if input_tensor is None:
            input_tensor = Input(shape = input_shape)
            
        if coord is None:
            coord = Input(shape = coord_shape)
        
        out_0 = self.pre_block(input_tensor, input_channel, self.num_channel_forw[0])
        pool_0 = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2),data_format='channels_first')(out_0)
        
        out_1 = self.res_block(self.num_blocks_forw[0], pool_0, self.num_channel_forw[0], self.num_channel_forw[1], 'forw1.')
        pool_1 = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2),data_format='channels_first')(out_1)
        
        out_2 = self.res_block(self.num_blocks_forw[1], pool_1, self.num_channel_forw[1], self.num_channel_forw[2], 'forw2.')
        pool_2 = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2),data_format='channels_first')(out_2)
        
        out_3 = self.res_block(self.num_blocks_forw[2], pool_2, self.num_channel_forw[2], self.num_channel_forw[3], 'forw3.')
        pool_3 = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2),data_format='channels_first')(out_3)
        
        out_4 = self.res_block(self.num_blocks_forw[3], pool_3, self.num_channel_forw[3], self.num_channel_forw[4], 'forw4.')
                
        rev_3 = self.deconv_block(out_4, 64, 64, 'path1')
        comb_3 = concatenate([rev_3, out_3], axis = 1)
        comb_3 = self.res_block(self.num_blocks_back[0], comb_3, self.num_channel_back[0], self.num_channel_back[1], 'back3.')
        
        rev_2 = self.deconv_block(comb_3, 64, 64, 'path2')
        comb_2 = concatenate([rev_2, out_2, coord], axis = 1)
        comb_2 = self.res_block(self.num_blocks_back[1], comb_2, self.num_channel_back[0] + 3, self.num_channel_back[0], 'back2.')

        comb_2 = Dropout(rate = 0.5)(comb_2)
        
        out = self.output_block(comb_2, self.num_channel_back[0], 5 * 3)
        
        out = Reshape((3, 5, 32, 32, 32))(out)
        
        out = Permute((3, 4, 5, 1, 2))(out)
        
        model = Model([input_tensor, coord], out, name='grt123')
        
        return model
        # n,z,y,x,3(anchors),5(o,x,y,z,r)
                                

net = GRT123Net()
model = net.get_model(input_tensor = None, input_shape = (1, 128, 128, 128), input_channel = 1, output_channel = 3, coord = None)