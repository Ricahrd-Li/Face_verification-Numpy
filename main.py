import numpy as np
import pandas as pd
from skimage import io, transform, measure
import glob
import re
import pickle
import time
import logging

image_width = 128
image_height = 128

def traindata_loader(path):
    # paths = ['LFW_dataset/match_pairs', 'LFW_dataset/mismatch_pairs']
    # for path in paths:
    #     dirs = os.listdir(path)
    #     for dir in dirs:
    #         images = os.listdir(path+'/'+dir)
    #         for image in images:
    #             print(image)
    # imlist = io.ImageCollection(path)
    # print(imlist[0].shape)
    imlist = glob.glob(path)
    rawImageArray = np.zeros((len(imlist), image_height, image_width), dtype=np.double)
    people_names = []
    for i in range(len(imlist)):
        image = imlist[i]
        raw_image = io.imread(image, as_gray=True)
        rawImageArray[i] = transform.resize(raw_image, (image_width, image_height))
        people_name = re.search(r'(?<=\\)[a-zA-Z_-]+', image[28:]) 
        people_names.append(people_name.group(0)[:-1])  
    data_label = pd.get_dummies(people_names).to_numpy()    
    return rawImageArray, data_label

def conv(data_in, filter, filter_bias):
    '''
      ## convolve with stride=1 and padding=0
      (l, w, h), (fl, fw, h, n), (n) -> (l-fl+1, w-fw+1, n)
    '''
    assert len(data_in.shape) == 3 and len(filter.shape) == 4 and len(filter_bias.shape) == 1
    assert data_in.shape[2] == filter.shape[2]    # must share the same height
    assert filter.shape[3] == filter_bias.shape[0]  # each conv kernel has a bias
    input_len, input_width, input_height = data_in.shape
    filter_len, filter_width, filter_height, filter_num = filter.shape
    feature_len = input_len - filter_len + 1
    feature_width = input_width - filter_width + 1
    feature_height = filter_num
    output = np.zeros((feature_len, feature_width, feature_height), dtype=np.double)
    img2col = np.zeros((feature_len, feature_width, filter_len*filter_width*filter_height), dtype=np.double)
    for i in range(feature_len):
        for j in range(feature_width):
            img2col[i,j,:] = data_in[i:i+filter_len,j:j+filter_len,:].flatten()
    for filt_index in range(feature_height):
        filt = filter[:,:,:,filt_index].flatten()
        output[:,:,filt_index] = np.matmul(img2col, filt) + filter_bias[filt_index]
    # for filt_index in range(feature_height):
    #     for i in range(feature_len):
    #         for j in range(feature_width):
    #             output[i][j][filt_index] = (data_in[i:i+filter_len,j:j+filter_len,:] * filter[:,:,:,filt_index]).sum()
    #     output[:,:,filt_index] += filter_bias[filt_index]
    return output

def mfm(data_in):
    '''
    ## max-feature-map 2/1
    (l, w, h) -> (l, w, h/2)  
    '''
    assert len(data_in.shape) == 3
    assert data_in.shape[2]%2 == 0
    input_len, input_width, input_height = data_in.shape
    split = np.zeros((2,input_len, input_width, input_height//2), dtype=np.double)
    split[0] = data_in[:,:,:input_height//2]
    split[1] = data_in[:,:,input_height//2:]
    output = np.amax(split, axis=0)
    repmax = np.zeros(data_in.shape, dtype=np.double)
    repmax[:,:,:input_height//2] = repmax[:,:,input_height//2:] = output
    location = (repmax == data_in)
    assert output.shape == (input_len, input_width, input_height//2)
    assert location.shape == data_in.shape
    return output, location

def pool(data_in):
    '''
    ## max-pooling with 2*2 filter size and stride=2
    (l, w, h) -> (l/2, w/2, h)
    '''

    assert len(data_in.shape) == 3
    assert data_in.shape[0]%2 == 0 and data_in.shape[1]%2 == 0 
    output = measure.block_reduce(data_in, (2,2,1), func=np.max)
    repmax = np.repeat(np.repeat(output, 2, axis=0), 2, axis=1)
    location = (repmax == data_in)
    # location[location==0] = 1
    # input_len, input_width, input_height = data_in.shape
    # pool_len = input_len//2
    # pool_width = input_width//2
    # data_reshaped = data_in.reshape(pool_len, 2, pool_width, 2, input_height)
    # # feature_reshaped = feature.reshape(pool_h, feature_h//pool_h, pool_w, feature_w//pool_w, feature_ch)
    # out = data_reshaped.max(axis=1).max(axis=2)
    # out_location_c = data_reshaped.max(axis=1).argmax(axis=2)
    # out_location_r = data_reshaped.max(axis=3).argmax(axis=1)
    assert output.shape == (data_in.shape[0]//2, data_in.shape[1]//2, data_in.shape[2])
    assert location.shape == data_in.shape
    return output, location

def padding(data_in, pad_size):
    '''
    ## pad the first 2 dimension of data_in
     (l, w, h), n -> (n+l+n, n+w+n, h)
    '''
    assert len(data_in.shape) == 2 or len(data_in.shape) == 3
    assert pad_size > 0
    if len(data_in.shape) == 2:
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size] = data_in
    else:
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2, data_in.shape[2]), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size,:] = data_in
    return output

def fc(data_in, weights, bias):
    '''
    ## fully-connected layer
    ndarray, (w, node_num), (node_num) -> (node_num)
    '''
    data = data_in.flatten()
    assert data.shape[0] == weights.shape[0]
    assert weights.shape[1] == bias.shape[0]
    weight_num, node_num = weights.shape
    # output = np.zeros((node_num), dtype=np.double)
    # for i in range(node_num):
    #     output[i] = np.matmul(data, weights) + bias[i]
    output = np.matmul(data, weights) + bias
    assert output.shape == (node_num,)
    return output

def mfm_fc(data_in):
    '''
    ## max-feature-map for fully-connected layer, 2/1
    (node_num) -> (node_num/2)
    '''
    assert len(data_in.shape) == 1
    assert data_in.shape[0]%2 == 0
    node_num = data_in.shape[0]
    split = np.zeros((2, node_num//2), dtype=np.double)
    split[0] = data_in[:node_num//2]
    split[1] = data_in[node_num//2:]
    output = np.amax(split, axis=0)
    repmax = np.zeros(data_in.shape, dtype=np.double)
    repmax[:node_num//2] = repmax[node_num//2:] = output
    location = (repmax == data_in)
    assert output.shape == (node_num//2,)
    assert location.shape == data_in.shape
    return output, location

def softmax(data_in):
    '''
    ## softmax layer
    (3095) -> (3095)
    '''
    m = np.amax(data_in)
    data_in -=m
    # print(data_in)
    # return data_in - np.log()
    print(data_in)
    e = np.exp(data_in)
    s = np.sum(e)
    # print(e)
    # print(s)
    output = e/s
    return output 

def cross_entropy(data_in,label_vec):
    '''
    ## cross entropy as loss function. 
    (3095) -> 1
    '''
    l = np.log(data_in)
    return -np.dot(l, label_vec)

class LightCNN_9(object):
    def __init__(self, path=None):
        if path != None:    
            file = open(path, 'rb')
            data = file.read()
            file.close()
            self.__dict__ = pickle.loads(data)
        else:
            self.conv1_kernel = np.random.randn(5, 5, 1, 96)*np.sqrt(1/(5*5*1))
            self.conv1_bias = np.zeros((96), dtype=np.double)
            self.conv2a_kernel = np.random.randn(1, 1, 48, 96)*np.sqrt(1/(1*1*48))
            self.conv2a_bias = np.zeros((96), dtype=np.double)
            self.conv2_kernel = np.random.randn(3, 3, 48, 192)*np.sqrt(1/(3*3*48))
            self.conv2_bias = np.zeros((192), dtype=np.double)
            self.conv3a_kernel = np.random.randn(1, 1, 96, 192)*np.sqrt(1/(1*1*96))
            self.conv3a_bias = np.zeros((192), dtype=np.double)
            self.conv3_kernel = np.random.randn(3, 3, 96, 384)*np.sqrt(1/(3*3*96))
            self.conv3_bias = np.zeros((384), dtype=np.double)
            self.conv4a_kernel = np.random.randn(1, 1, 192, 384)*np.sqrt(1/(1*1*192))
            self.conv4a_bias = np.zeros((384), dtype=np.double)
            self.conv4_kernel = np.random.randn(3, 3, 192, 256)*np.sqrt(1/(3*3*192))
            self.conv4_bias = np.zeros((256), dtype=np.double)
            self.conv5a_kernel = np.random.randn(1, 1, 128, 256)*np.sqrt(1/(1*1*128))
            self.conv5a_bias = np.zeros((256), dtype=np.double)
            self.conv5_kernel = np.random.randn(3, 3, 128, 256)*np.sqrt(1/(3*3*128))
            self.conv5_bias =np.zeros((256), dtype=np.double)
            self.fc_weights = np.random.randn(8*8*128, 3095)*np.sqrt(2/(8*8*128+3095))
            self.fc_bias = np.zeros((3095), dtype=np.double)
            self.fcout_weights = np.random.randn(256, 3095)*np.sqrt(2/(256+3095))
            self.fcout_bias = np.zeros((3095), dtype=np.double)
            self.conv_kernel = [self.conv1_kernel,self.conv2_kernel,self.conv3_kernel,self.conv4_kernel,self.conv5_kernel]  
            self.conva_kernel = [self.conv2a_kernel,self.conv3a_kernel,self.conv4a_kernel,self.conv5a_kernel]  
            self.conv_bias = [self.conv1_bias,self.conv2_bias,self.conv3_bias,self.conv4_bias,self.conv5_bias]
            self.conva_bias = [self.conv2a_bias,self.conv3a_bias,self.conv4a_bias,self.conv5a_bias]
            self.fc_w = [self.fc_weights,self.fcout_weights]
            self.fc_b = [self.fc_bias,self.fcout_bias]

        return
    
    def forward(self, data):
        time1 = time.time()
        pad1 = padding(data, 2)
        conv_input = np.zeros((pad1.shape[0], pad1.shape[1], 1), dtype=np.double)
        conv_input[:,:,0] = pad1

        conv1 = conv(conv_input, self.conv1_kernel, self.conv1_bias)
        mfm1, mfm1_location = mfm(conv1)

        pool1, pool1_location = pool(mfm1)

        conv2a = conv(pool1, self.conv2a_kernel, self.conv2a_bias)
        mfm2a, mfm2a_location = mfm(conv2a)
        conv2 = conv(padding(mfm2a,1), self.conv2_kernel, self.conv2_bias)
        mfm2, mfm2_location = mfm(conv2)

        pool2, pool2_location = pool(mfm2)

        conv3a = conv(pool2, self.conv3a_kernel, self.conv3a_bias)
        mfm3a, mfm3a_location = mfm(conv3a)
        conv3 = conv(padding(mfm3a, 1), self.conv3_kernel, self.conv3_bias)
        mfm3, mfm3_location = mfm(conv3)

        pool3, pool3_location = pool(mfm3)

        conv4a = conv(pool3, self.conv4a_kernel, self.conv4a_bias)
        mfm4a, mfm4a_location = mfm(conv4a)
        conv4 = conv(padding(mfm4a, 1), self.conv4_kernel, self.conv4_bias)
        mfm4, mfm4_location = mfm(conv4)

        conv5a = conv(mfm4, self.conv5a_kernel, self.conv5a_bias)
        mfm5a, mfm5a_location = mfm(conv5a)
        conv5 = conv(padding(mfm5a,1), self.conv5_kernel, self.conv5_bias)
        mfm5, mfm5_location = mfm(conv5)
        
        pool4, pool4_location = pool(mfm5)

        fc1 = fc(pool4, self.fc_weights, self.fc_bias)
        mfm_fc1, mfm_fc1_location = mfm_fc(fc1)

        fc2 = fc(mfm_fc1, self.fcout_weights, self.fcout_bias)
        time2 = time.time()
        print(time2-time1)
        return fc2.shape, fc2, 

    def train(self, data, label, epoch, min_batch_size, eta):
        def SGD():

            pass
        def update_batch(data, label,min_batch_size, eta):

            # for i in min_batch_size:
            g_conv_w, g_conv_b, g_conva_w, g_conva_b, g_fc_w, g_fc_b = backprob(data,label)

            
            for w, g_w in zip(self.conv_kernel,g_conv_w):
                # print(g_w)
                w -= eta* g_w
            for w, g_w in zip(self.conva_kernel,g_conva_w):
                w -= eta* g_w

            for b, g_b in zip(self.conv_bias,g_conv_b):
                b -= eta* g_b
            for b, g_b in zip(self.conva_bias,g_conva_b):
                b -= eta* g_b

            for w, g_w in zip(self.fc_w,g_fc_w):
                w - eta* g_w
            for b, g_b in zip(self.fc_b,g_fc_b):
                b - eta* g_b[:,0]
            
            return

        def get_derivative_softmax(fc2_output, label_vec):
            '''
            fc2_output: (3095,1)
            return (3095,1)
            '''
            return fc2_output - label_vec

        def get_derivative_fcout(input_vec, fc_weights, fc_bias, bp_gradient):
            '''
            y = Wx + b
            bp_gradient: 从后续layer传来的梯度 (n, 1)
            fc2: input_vec: (256,) bp_gradient: (3095,), fc_weights:(256,3095)
            '''
            dw = np.matmul(input_vec[:,None],bp_gradient[:,None].transpose())
            # dw = np.clip(dw,-20,20)
            # print(dw.shape)
            assert dw.shape == fc_weights.shape
            db = bp_gradient
            db = db[:,None]
            assert db.shape == fc_bias[:,None].shape
            dx = np.matmul(fc_weights, bp_gradient[:,None])
            # assert dx.shape == (256,1)
            return dw,db,dx
        
        def get_derivate_mfm_fc1(location, bp_gradient):
            '''
            input_vec: (512,)
            bp_gradient: (256,1)
            '''
            tmp = np.vstack((bp_gradient,bp_gradient))
            # print(tmp.shape)
            assert tmp.shape[0] == 512
            # print(location.shape)
            output = tmp * location[:,None].astype(int) 
            assert output.shape == (512,1)
            return output

        def get_derivative_fc(input_vec, fc_weights, fc_bias, bp_gradient):
            '''
            y = Wx + b
            bp_gradient: 从后续layer传来的梯度 (n, 1)
            fc1: input_vec: (8*8*128,) bp_gradient: (512,1), fc_weights:(8*8*128,512)
            '''
            # print(bp_gradient.shape)
            dw = np.matmul(input_vec[:,None],bp_gradient.transpose())
            # dw = np.clip(dw,-20,20)
            # print(dw.shape)
            # print(fc_weights.shape)
            assert dw.shape == fc_weights.shape
            db = bp_gradient
            assert db.shape == fc_bias[:,None].shape
            dx = np.matmul(fc_weights, bp_gradient)
            assert dx.shape == (8*8*128,1)
            return dw,db,dx
            
        def rot180(conv_filters):
            '''
            conv_filters: (fl, fw, h, n) -> (fl, fw, h, n)
            '''
            rot180_filters = np.zeros((conv_filters.shape))
            for filter_num in range(conv_filters.shape[-1]):
                for img_channal in range(conv_filters.shape[-2]):
                    rot180_filters[:,:,img_channal,filter_num] = np.flipud(np.fliplr(conv_filters[:,:,img_channal,filter_num]))
            return rot180_filters
        
        def get_derivative_conv(input_img, filter, filter_bias, bp_gradient, conv_output):
            # dw = np.zeros(filter.shape)
            # tmp_bias = np.zeros(1)
            # for i in range(bp_gradient.shape[-1]):
            #     tmp = bp_gradient[:,:,i]
            #     tmp_filter = np.stack([tmp] * input_img.shape[-1],axis=-1)[:,:,:,None]
            #     dw[:,:,:,i] = conv(input_img, tmp_filter, tmp_bias)
            time1 =time.time()
            dw = np.zeros(filter.shape)
            tmp_bias = np.zeros(1)
            for i in range(filter.shape[-1]): # 256
                tmp = bp_gradient[:,:,i] # kernel
                for j in range(filter.shape[-2]): # 128个channal
                    tmp_filter = tmp[:,:,None,None] # 扩展成 (16,16,1,1)
                    dw[:,:,j,i] = conv(input_img[:,:,j][:,:,None], tmp_filter,tmp_bias)[:,:,0]
            time2 =time.time()
            print("conv_time:", time2-time1)


            # conv_result = np.zeros(()) 

            # dw = np.clip(dw, -20, 20)
            assert dw.shape == filter.shape
            tmp_bias = np.zeros(input_img.shape[-1])
            rot_filter = rot180(filter).swapaxes(-2,-1)
            if filter.shape[0]!= 1:
                dx = conv(padding(conv_output,filter.shape[0]-1), rot_filter, tmp_bias)[1:-1,1:-1,:]
            else:
                dx = conv(conv_output, rot_filter, tmp_bias)
            db = np.sum(np.sum(bp_gradient,axis = 1),axis=0)[:None]
            assert db.shape == filter_bias.shape
            return dw,db,dx      

        def get_derivative_conv1(input_img, filter, filter_bias, bp_gradient, conv_output):
            '''
            第一层卷积计算反向传播梯度时所用的函数
            '''
            time1 = time.time()
            dw = np.zeros(filter.shape)
            input_img = input_img[:,:,None] # (128,128) -> (128,128,1)
            tmp_bias = np.zeros(1)
            for i in range(filter.shape[-1]): # 256
                tmp = bp_gradient[:,:,i] # kernel
                for j in range(filter.shape[-2]): # 128个channal
                    tmp_filter = tmp[:,:,None,None] # 扩展成 (16,16,1,1)
                    dw[:,:,j,i] = conv(input_img[:,:,j][:,:,None], tmp_filter,tmp_bias)[:,:,0]  
                # dw[:,:,:,i] = conv(input_img, tmp_filter, tmp_bias)
            # dw = np.clip(dw, -20, 20)
            time2 = time.time()
            print("conv1_time:",time2-time1)
            assert dw.shape == filter.shape
            db = np.sum(np.sum(bp_gradient,axis = 1),axis=0)[:None]
            assert db.shape == filter_bias.shape
            return dw,db  

        def get_derivative_pool(location, bp_gradient,pool_output):
            '''
            bp_gradient: (8*8*128,1)
            '''
            # print(bp_gradient.shape)
            bp_gradient = bp_gradient.reshape(pool_output.shape)
            bp_gradient = bp_gradient.repeat(2,axis=0).repeat(2,axis=1)
            # print(location.shape)
            output = bp_gradient * location
            # assert output.shape == 
            # print(output)
            return output
            # output = bp_gradient

        def get_derivative_mfm(location, bp_gradient):
            '''
            location
            '''
            # print(location.shape)
            tmp = np.concatenate((bp_gradient,bp_gradient),axis=-1)
            # print(tmp.shape)
            output = tmp * location.astype(int) 
            # print(output.shape)
            return output

        def backprob(data, label):

            # forward
            pad1 = padding(data, 2)
            conv_input = np.zeros((pad1.shape[0], pad1.shape[1], 1), dtype=np.double)
            conv_input[:,:,0] = pad1

            conv1 = conv(conv_input, self.conv1_kernel, self.conv1_bias)
            mfm1, mfm1_location = mfm(conv1)

            pool1, pool1_location = pool(mfm1)

            conv2a = conv(pool1, self.conv2a_kernel, self.conv2a_bias)
            mfm2a, mfm2a_location = mfm(conv2a)
            conv2 = conv(padding(mfm2a,1), self.conv2_kernel, self.conv2_bias)
            mfm2, mfm2_location = mfm(conv2)

            pool2, pool2_location = pool(mfm2)

            conv3a = conv(pool2, self.conv3a_kernel, self.conv3a_bias)
            mfm3a, mfm3a_location = mfm(conv3a)
            conv3 = conv(padding(mfm3a, 1), self.conv3_kernel, self.conv3_bias)
            mfm3, mfm3_location = mfm(conv3)

            pool3, pool3_location = pool(mfm3)

            conv4a = conv(pool3, self.conv4a_kernel, self.conv4a_bias)
            mfm4a, mfm4a_location = mfm(conv4a)
            conv4 = conv(padding(mfm4a, 1), self.conv4_kernel, self.conv4_bias)
            mfm4, mfm4_location = mfm(conv4)

            conv5a = conv(mfm4, self.conv5a_kernel, self.conv5a_bias)
            mfm5a, mfm5a_location = mfm(conv5a)
            conv5 = conv(padding(mfm5a,1), self.conv5_kernel, self.conv5_bias)
            mfm5, mfm5_location = mfm(conv5)
            
            pool4, pool4_location = pool(mfm5)

            fc1 = fc(pool4, self.fc_weights, self.fc_bias)
            # mfm_fc1, mfm_fc1_location = mfm_fc(fc1)
# 
            # fc2 = fc(mfm_fc1, self.fcout_weights, self.fcout_bias)
            # fc2 = fc(fc1, self.fcout_weights, self.fcout_bias)
            # print(fc2)

            softmax_output = softmax(fc1)
            loss = cross_entropy(softmax_output,label)
            print("loss:", loss)

            time1 = time.time()
            g_softmax = get_derivative_softmax(fc1,label)
            # print(gradient_softmax)
            # print(fc2.shape)
            # print(mfm_fc1.shape)
            # print(self.fcout_weights.shape)

            # g_fc2_w, g_fc2_b, g_fc2_x = get_derivative_fcout(mfm_fc1,self.fcout_weights,self.fcout_bias,g_softmax)
            # print(g_fc2_w)/
            # print("===========g_fc2_w=============")
            # print(g_fc2_w)
            # self.fcout_weights -= 0.00001*g_fc2_w
            # self.fcout_bias -= 0.00001*g_fc2_b[:,0]
            # # # print(g_fc2_x.shape)x
            # print(g_fc2_x)
            # g_mfm_fc1 = get_derivate_mfm_fc1(mfm_fc1_location, g_fc2_x)
        
            
            g_fc1_w, g_fc1_b, g_fc1_x = get_derivative_fcout(pool4.flatten(), self.fc_weights, self.fc_bias, g_softmax)
            # print("===========g_fc1_w=============")
            # print(g_fc1_w)
            # self.fc_weights -= 0.00001*g_fc1_w
            # self.fc_bias -= 0.00001*g_fc1_b[:,0]

            g_pool4 = get_derivative_pool(pool4_location, g_fc1_x, pool4)
            
            g_mfm5 = get_derivative_mfm( mfm5_location, g_pool4)
            # # g_mfm5: (16,16,128) padding(mfm5a,1): (17,17,256)
            time3 = time.time()
            g_conv5_w, g_conv5_b, g_conv5_x = get_derivative_conv(padding(mfm5a,1),self.conv5_kernel, self.conv5_bias, g_mfm5,conv5)
            time4 = time.time()
            # print("=======g_conv5_w=")

            # self.conv5_kernel -= 0.00001*g_conv5_w
            # self.conv5_bias -= 0.00001*g_conv5_b

            
            g_mfm5a = get_derivative_mfm( mfm5a_location, g_conv5_x)
            g_conv5a_w, g_conv5a_b, g_conv5a_x = get_derivative_conv(mfm4,self.conv5a_kernel, self.conv5a_bias, g_mfm5a,conv5a)
            # self.conv5a_kernel -= 0.00001*g_conv5a_w
            # self.conv5a_bias -= 0.00001*g_conv5a_b
            # # #
            g_mfm4 = get_derivative_mfm( mfm4_location, g_conv5a_x)
            g_conv4_w, g_conv4_b, g_conv4_x = get_derivative_conv(padding(mfm4a,1),self.conv4_kernel, self.conv4_bias, g_mfm4,conv4)
            # self.conv4_kernel -= 0.00001*g_conv4_w
            # self.conv4_bias -= 0.00001*g_conv4_b
            g_mfm4a = get_derivative_mfm( mfm4a_location, g_conv4_x)
            g_conv4a_w, g_conv4a_b, g_conv4a_x = get_derivative_conv(pool3,self.conv4a_kernel, self.conv4a_bias, g_mfm4a,conv4a)
        
            # self.conv4a_kernel -= 0.00001*g_conv4a_w
            # self.conv4a_bias -= 0.00001*g_conv4a_b
            g_pool3 = get_derivative_pool(pool3_location, g_conv4a_x,pool3)
            
            # #
            g_mfm3 = get_derivative_mfm( mfm3_location, g_pool3)
            g_conv3_w, g_conv3_b, g_conv3_x = get_derivative_conv(padding(mfm3a,1),self.conv3_kernel, self.conv3_bias, g_mfm3,conv3)
            # self.conv3_kernel -= 0.00001*g_conv3_w
            # self.conv3_bias -= 0.00001*g_conv3_b
            g_mfm3a = get_derivative_mfm( mfm3a_location, g_conv3_x)
            g_conv3a_w, g_conv3a_b, g_conv3a_x = get_derivative_conv(pool2,self.conv3a_kernel, self.conv3a_bias, g_mfm3a,conv3a)
            # self.conv3a_kernel -= 0.00001*g_conv3a_w
            # self.conv3a_bias -= 0.00001*g_conv3a_b
            g_pool2 = get_derivative_pool(pool2_location, g_conv3a_x,pool2)

            #
            g_mfm2 = get_derivative_mfm( mfm2_location, g_pool2)
            g_conv2_w, g_conv2_b, g_conv2_x = get_derivative_conv(padding(mfm2a,1),self.conv2_kernel, self.conv2_bias, g_mfm2,conv2)
            g_mfm2a = get_derivative_mfm( mfm2a_location, g_conv2_x)
            g_conv2a_w, g_conv2a_b, g_conv2a_x = get_derivative_conv(pool1,self.conv2a_kernel, self.conv2a_bias, g_mfm2a,conv2a)
        
            g_pool1 = get_derivative_pool(pool1_location, g_conv2a_x,pool1)

            g_mfm1 = get_derivative_mfm( mfm1_location, g_pool1)
            g_conv1_w, g_conv1_b= get_derivative_conv1(data,self.conv1_kernel, self.conv1_bias, g_mfm1,conv1)
            time2 = time.time()
            print("bw_time:", time2 - time1)
            # print("conv_time:", time4- time3)
            # # gradient_fc2 = get_derivative_fc
            
            # g_conv_w = [ g_conv1_w,g_conv2_w, g_conv3_w, g_conv4_w, g_conv5_w ]
            # g_conv_b = [ g_conv1_b,g_conv2_b, g_conv3_b, g_conv4_b, g_conv5_b ]

            # g_conva_w = [  g_conv2a_w, g_conv3a_w, g_conv4a_w, g_conv5a_w ]
            # g_conva_b = [  g_conv2a_b, g_conv3a_b, g_conv4a_b, g_conv5a_b ]
            
            # g_fc_w = [g_fc1_w, g_fc2_w]
            # g_fc_b = [g_fc1_b, g_fc2_b]

            # return g_conv_w, g_conv_b, g_conva_w, g_conva_b, g_fc_w, g_fc_b
        
        for i in range(50):
            backprob(data,label)
            # update_batch(data,label,1,0.0001)

    def test(self, data, label):
        return
    def save(self):
        file = open('LightCNN9_model.bin', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return



if __name__ == "__main__":
    # path = './LFW_dataset/*/*/*.jpg'
    path = './test_image/*/*/*.jpg'
    train_data, train_label = traindata_loader(path)
    print("Data loading finished.")
    model = LightCNN_9()

    # print(train_label)
    # print(train_label.shape)
    a = np.zeros((8,3095),dtype=int)
    for i in range(train_label.shape[0]):
        a[i,:train_label[i].shape[0]] = train_label[i]
    # print(a
        
    model.train(train_data[0],a[0],1,1,1)
    # print(model.forward(train_data[0]))
    