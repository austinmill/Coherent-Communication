# -*- coding: UTF-8 -*-
#####################################################################
#Author: Austin Zhang
#Time: 2020/12/01
#Email: austinmill2010@gmail.com
#Website: https://github.com/austinmill/Coherent-Communication
#####################################################################
#####################################################################
if(1):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sympy as sy
    import scipy.constants as cons
    from scipy.interpolate import interp1d
    from scipy import signal
    from scipy.constants import pi
    from scipy.fftpack import fft,ifft
    from numpy import sqrt
    from numpy import exp
    import time as tm
    cd_coef = 17e3	#光纤色散系数，单位ns/m^2
#####################################################################
#####################################################################
#####################################################################
#定义<任意比例插值>函数
#可选'linear'，'quadratic'模式
def Interpolate(data, ratio, kind):
    ratio = abs(float(ratio))
    input_length = len(data)
    input_index  = np.arange(0,input_length)
    output_index = np.arange(0,input_length,1/ratio)
    output_index = np.array([x for x in output_index if x<=input_length-1])
    fl = interp1d(input_index, data, kind)
    return(fl(output_index))
#####################################################################
#####################################################################
#####################################################################
#定义<简单线性插值>函数
def simple_interpld(X,Y,x):
    x1 = X[0]
    x2 = X[1]
    y1 = Y[0]
    y2 = Y[1]
    return((y2-y1)/(x2-x1)*(x-x1)+y1)
#simple_interpld([0,2],[2,3],1)
#####################################################################
#####################################################################
#####################################################################
#定义<IIR>函数
#二阶IIR需要反转输入序列，物理上无法实现。优点是无时间平移，无需补偿skew
def IIR(data, weight, order=1):
    input_length = len(data)
    output = np.array([0.0]*input_length)
    if(order==1):
        output[0] = data[0] *(1-weight)
        for i in range(1,input_length):
            output[i] = data[i] *(1-weight) + output[i-1] *weight
        return(output)
    if(order==2):
        data = IIR(data, weight, 1)
        data = IIR(data[::-1], weight, 1)
        return(data[::-1])
    if(order==4):
        data = IIR(data, weight, 2)
        data = IIR(data[::-1], weight, 2)
        return(data[::-1])
#a_ff = IIR_2nd(a, order=7/32)
#####################################################################
#####################################################################
#####################################################################
#定义<N bit量化-整数>函数
#输入-0.5~0.5
#N=8bit；integ = True 时，输出-127~128
#N=8bit；integ = False时，输出-0.49803922~0.50196078
def Digitalized(data, N, integ=True):
    data = np.array(data)
    input_length = len(data)
    data = data*(2**N-1)
    for i in range(0,input_length):
        data[i] = max(data[i],1-2**(N-1))
        data[i] = min(data[i],2**(N-1))
        data[i] = round(data[i],0)
    if(not integ):
        data = data/(2**N-1)
    return(data)
#a.astype('int8')
#Digitalized(kk[0],8,True)
#####################################################################
#####################################################################
#####################################################################
def IIR_2nda(data, a_0, a_1, b_1):
    input_length = len(data)
    output = np.array([0.0]*input_length)
    temp_delay = 0
    for i in range(1,input_length):
        temp = data[i] + temp_delay *b_1
        output[i] = temp *a_0 + temp_delay *a_1
        temp_delay = temp
        i = i+1
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<硬判决>函数
def hard_dec(input_data,constellation,simple=True):
    e_sum = 0
    if(not simple):
        error = np.array([])
        det   = np.array([])
    for symbol in input_data:
        error_vector = abs(constellation -symbol)**2
        p     = np.argmin(error_vector)
        e_sum = e_sum + error_vector[p]
        if(not simple):        
            error = np.append(error,error_vector [p])
            det   = np.append(det  ,constellation[p])
    if(not simple):
        output = {'Decision':det,'Error':error,'Error_sum':e_sum}
        return(output)
    else:
        return(e_sum)
#hard_dec(input_data,constellation,simple=True)
#####################################################################
#####################################################################
#####################################################################
#定义<搜索最小码间距>函数
def d_free(input):
    input_length = len(input)
    d_free = []
    for i in range(0,input_length):
        d_free.append(min(abs(input[i] -input[i+1:input_length])))
        d_free = [min(d_free)]
    return(d_free)
#####################################################################
#####################################################################
#####################################################################
#定义<标准星座图生成>函数
def std_constellation(mod,plot=True):
    if  (mod == 'QPSK' or mod == '16QAM' or mod == '32QAM' or mod == '64QAM' or mod == '256QAM'):
        if  (mod == 'QPSK'):
            qam_order = 2
        elif(mod == '16QAM'):
            qam_order = 4
        elif(mod == '32QAM'):
            qam_order = 6
        elif(mod == '64QAM'):
            qam_order = 8
        elif(mod == '256QAM'):
            qam_order = 16
	#———————————————————————#
        a = [x -(qam_order-1)/2 for x in range(0,qam_order)]
        constellation = [1.0]*(qam_order**2)
        for i in range(0,qam_order):
            for k in range(0,qam_order):
                constellation[i*qam_order+k] = complex(a[i],a[k])
    elif(mod == 'BPSK'):
        constellation = [1+0j, -1-0j]
    elif(mod == '8QAM'):
        a = (sqrt(3)-1)/2
        constellation = [1, -1, 0+1j, 0-1j, a*(1+1j), a*(1-1j), a*(-1+1j), a*(-1-1j)]
    elif(mod == 'SP-16QAM'):
        constellation = [-0.5+0.5j,-0.5-0.5j/3,-0.5/3+0.5j/3,-0.5/3-0.5j,0.5/3+0.5j,0.5/3-0.5j/3,0.5+0.5j/3,0.5-0.5j]
    #———————————————————————————#
    if  (mod == '32QAM'):
        constellation = [x for x in constellation if abs(x)!=max([abs(x) for x in constellation])]
    sq_mod = np.mean([abs(x)**2 for x in constellation])
    constellation = np.array(constellation)/sqrt(sq_mod)
    #———————————————————————————#
    if(plot):
        plt.figure(figsize=(5,5))
        plt.scatter(constellation.real,constellation.imag,color='red',s=30,label='$'+str(mod)+'\ '+'Constellation$')
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.ylabel('Imaginary') 
        plt.xlabel('Real')
        plt.grid(color='k',linestyle=':')
        #plt.title('A simple plot')
        plt.legend()
        plt.show()
    return(constellation)
#std_constellation(mod='QPSK')
#####################################################################
#####################################################################
#####################################################################
#定义<QAM编码>函数
def encod_qam(data1, data2=0, data3=0, data4=0, mod='BPSK', plot=True):
    factor1 = 0.93    #用于 8QAM补偿调制器非线性
    factor2 = 3/3.2   #用于16QAM补偿调制器非线性
    #———————————————————————————#    
    input_length = len(data1)
    I = np.array([0.0]*input_length)
    Q = np.array([0.0]*input_length)
    #———————————————————————————#
    if(mod == 'BPSK'):
        for i in range(0,input_length):
            if  (data1[i]==0):  I[i] = -0.5
            else:               I[i] =  0.5
    #———————————————————————————#
    if(mod == 'QPSK'):
        if  (input_length!= len(data2)):
            print('Input data are not synchronized!!')
            return()
        for i in range(0,input_length):
            if  (data1[i]==0):  I[i] = -0.5
            else:               I[i] =  0.5
            if  (data2[i]==0):  Q[i] = -0.5
            else:               Q[i] =  0.5
    #———————————————————————————#
    if(mod == '8QAM'):
        if  (input_length!= len(data2) or input_length!= len(data3)):
            print('Input data are not synchronized!!')
            return()
        a = (sqrt(3)-1)/4 *factor1
        for i in range(0,input_length):
            if(data1[i]==0 and data2[i]==0 and data3[i]==0):	I[i] = -0.5;	Q[i] = 0
            if(data1[i]==1 and data2[i]==1 and data3[i]==0):	I[i] = +0.5;	Q[i] = 0
            if(data1[i]==0 and data2[i]==1 and data3[i]==0):	I[i] = 0;	Q[i] = 0.5
            if(data1[i]==1 and data2[i]==0 and data3[i]==0):	I[i] = 0;	Q[i] = -0.5
            #———————————————————#
            if(data1[i]==0 and data2[i]==0 and data3[i]==1):	I[i] = -a;	Q[i] = +a
            if(data1[i]==0 and data2[i]==1 and data3[i]==1):	I[i] = +a;	Q[i] = +a
            if(data1[i]==1 and data2[i]==1 and data3[i]==1):	I[i] = +a;	Q[i] = -a
            if(data1[i]==1 and data2[i]==0 and data3[i]==1):	I[i] = -a;	Q[i] = -a
    #———————————————————————————#
    if(mod == '16QAM'):
        if(input_length!= len(data2) or input_length!= len(data3) or input_length!= len(data4)):
            print('Input data are not synchronized!!')
            return()
        a = 0.5/3 *factor2
        for i in range(0,input_length):
            if(data1[i]==1 and data2[i]==0):	I[i] = +0.5
            if(data1[i]==1 and data2[i]==1):	I[i] = +a
            if(data1[i]==0 and data2[i]==1):	I[i] = -a
            if(data1[i]==0 and data2[i]==0):	I[i] = -0.5
            #———————————————————#
            if(data3[i]==0 and data4[i]==0):	Q[i] = +0.5
            if(data3[i]==0 and data4[i]==1):	Q[i] = +a
            if(data3[i]==1 and data4[i]==1):	Q[i] = -a
            if(data3[i]==1 and data4[i]==0):	Q[i] = -0.5
    #———————————————————————————#
    if(plot):
        #plt.figure(figsize=(5,5))
        plt.scatter(I,Q,color='blue',s=30,label='$'+str(mod)+'\ '+'Constellation$')
        #plt.plot   (I,Q,color='blue',linestyle=':')
        plt.xlim(-1.0,1.0)
        plt.ylim(-1.0,1.0)
        plt.ylabel('Imaginary') 
        plt.xlabel('Real')
        plt.grid(color='k',linestyle=':')
        plt.legend()
        #plt.show()
        #plt.savefig('line.jpg')
    return([I,Q])
'''
if(1):
    size = 1024
    a = np.random.randint(0,2,size)
    b = np.random.randint(0,2,size)
    c = np.random.randint(0,2,size)
    d = np.random.randint(0,2,size)
    e = np.random.randint(0,2,size)
    f = np.random.randint(0,2,size)
    g = np.random.randint(0,2,size)
    encod_qam(a,b,c,d,mod='8QAM')
'''
#####################################################################
#####################################################################
#####################################################################
#定义<SP-QAM编码>函数
def encod_spqam(data1, data2, data3, data4, data5, data6=[], data7=[], mod='SP-16QAM', plot=True):
    input_length = len(data1)
    if(input_length!= len(data2) | input_length!= len(data3) | input_length!= len(data4) | input_length!= len(data5)):
        print('Input data are not synchronized!!')
        return()
    #———————————————————————————#
    if(mod == '32-SP-QAM'):
        dataA = data1 ^data2 ^data3        #等效于dataA = (data1 + data2 + data3)%2
        dataB = data1 ^data2 ^data4
        dataC = data1 ^data2 ^data5
        #———————————————————————#
        plt.subplot(1, 2, 1)
        H = encod_qam(data1,data2,data3,dataA,mod='16QAM',plot=plot)
        plt.title('$H\ Pol$')
        plt.subplot(1, 2, 2)
        V = encod_qam(data4,dataB,data5,dataC,mod='16QAM',plot=plot)
        plt.title('$V\ Pol$')
    #———————————————————————————#
    if(mod == 'SP-16QAM'):
        if(input_length!= len(data6)):
            print('Input data are not synchronized!!')
            return()
        #———————————————————————#
        dataA = data1 ^data2 ^data3
        dataB = data4 ^data5 ^data6
        #———————————————————————#
        plt.subplot(1, 2, 1)
        H = encod_qam(data1,data2,data3,dataA,mod='16QAM',plot=plot)
        plt.title('$H\ Pol$')
        plt.subplot(1, 2, 2)
        V = encod_qam(data4,data5,data6,dataB,mod='16QAM',plot=plot)
        plt.title('$V\ Pol$')
    #———————————————————————————#
    if(mod == '128-SP-QAM'):
        if(input_length!= len(data6) | input_length!= len(data7)):
            print('Input data are not synchronized!!')
            return()
        #———————————————————————#
        dataA = data1 ^data2 ^data3 ^data4 ^data5 ^data6 ^data7
        #———————————————————————#
        plt.subplot(1, 2, 1)
        H = encod_qam(data1,data2,data3,data4,mod='16QAM',plot=plot)
        plt.title('$H\ Pol$')
        plt.subplot(1, 2, 2)
        V = encod_qam(data5,data6,data7,dataA,mod='16QAM',plot=plot)
        plt.title('$V\ Pol$')
    #———————————————————————————#
    #plt.subplots_adjust(wspace=0,hspace=0.3)
    #plt.show()
    return([H,V])
'''
if(1):
    size = 1024
    a = np.random.randint(0,2,size)
    b = np.random.randint(0,2,size)
    c = np.random.randint(0,2,size)
    d = np.random.randint(0,2,size)
    e = np.random.randint(0,2,size)
    f = np.random.randint(0,2,size)
    g = np.random.randint(0,2,size)
    encod_spqam(a,b,c,d,e,f,g,mod='SP-16QAM')
'''
#####################################################################
#####################################################################
#####################################################################
#定义<DAC输出>函数
def DAC(data,Dac_Bit=8,Txdsp_SampleRatio=2,Dac_SampleRatio=2,Sim_SampleRatio=32,Dac_BW=23/32,SNR=25):
    SNR=10**(SNR/10)
    #———————————————————————————#从Tx_DSP采样率插值到DAC输出采样率
    print('Resampling from Tx DSP output...')
    ratio = Dac_SampleRatio/Txdsp_SampleRatio
    output_t  = Interpolate(data['Time'],ratio,kind='linear')
    output_hi = Interpolate(data['HI']  ,ratio,kind='quadratic')
    output_hq = Interpolate(data['HQ']  ,ratio,kind='quadratic')
    output_vi = Interpolate(data['VI']  ,ratio,kind='quadratic')
    output_vq = Interpolate(data['VQ']  ,ratio,kind='quadratic')
    #———————————————————————————#DAC输出分辨率截断
    output_hi = Digitalized(output_hi,Dac_Bit,integ=False)
    output_hq = Digitalized(output_hq,Dac_Bit,integ=False)
    output_vi = Digitalized(output_vi,Dac_Bit,integ=False)
    output_vq = Digitalized(output_vq,Dac_Bit,integ=False)
    #———————————————————————————#DAC输出采样率插值到仿真模拟信号的采样率
    print('Interpolating to analog signal...')
    ratio = Sim_SampleRatio/Dac_SampleRatio
    output_t  = Interpolate(output_t ,ratio,kind='linear')
    output_hi = Interpolate(output_hi,ratio,kind='linear')
    output_hq = Interpolate(output_hq,ratio,kind='linear')
    output_vi = Interpolate(output_vi,ratio,kind='linear')
    output_vq = Interpolate(output_vq,ratio,kind='linear')
    #———————————————————————————#模拟DAC的输出噪声
    print('Applying DAC Gaussian noise...')
    output_length = len(output_t)
    output_hi = np.random.normal(0,sqrt(np.mean(abs(output_hi)**2)/SNR),output_length) +output_hi
    output_hq = np.random.normal(0,sqrt(np.mean(abs(output_hq)**2)/SNR),output_length) +output_hq
    output_vi = np.random.normal(0,sqrt(np.mean(abs(output_vi)**2)/SNR),output_length) +output_vi
    output_vq = np.random.normal(0,sqrt(np.mean(abs(output_vq)**2)/SNR),output_length) +output_vq
    #———————————————————————————#模拟DAC的输出带宽
    print('Applying DAC bandwidth filter...')
    output_hi = IIR(output_hi,Dac_BW,2)
    output_hq = IIR(output_hq,Dac_BW,2)
    output_vi = IIR(output_vi,Dac_BW,2)
    output_vq = IIR(output_vq,Dac_BW,2)
    #———————————————————————————#
    output = {'Time':output_t,'HI':output_hi,'HQ':output_hq,'VI':output_vi,'VQ':output_vq}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<ADC采样>函数
#输入满幅-0.5～+0.5
#输出满幅-127～128
def ADC(time, hi, hq, vi, vq, Adc_clock, Adc_SamplingRatio=2, time_shift=0, Adc_Bit=8):
    #———————————————————————————#计算ADC采样周期
    Adc_sampling_period = 1.0/(Adc_clock*Adc_SamplingRatio)
    time_shift = time_shift % Adc_sampling_period
    input_length = len(hi)
    #———————————————————————————#计算ADC采样序列长度
    output_length = int((time[-1]-time_shift)/Adc_sampling_period)
    output_hi = np.zeros(output_length)
    output_hq = np.zeros(output_length)
    output_vi = np.zeros(output_length)
    output_vq = np.zeros(output_length)
    #———————————————————————————#计算ADC采样的时刻
    output_time = time_shift +np.arange(0,output_length)/(Adc_clock*Adc_SamplingRatio)
    #———————————————————————————#计算ADC采样时刻在原输入时间序列中所处的位置
    f1 = np.polyfit(time,np.arange(0,input_length),1)
    k  = f1[0]*output_time +f1[1]
    print('ADC is sampling',end='')
    #———————————————————————————#根据ADC采样时刻，对原输入数据插值采样
    for i in range(output_length):
        k1 = int(np.floor(k[i]))
        k2 = int(np.ceil(k[i]))
        if  (k1<=0):
            output_hi[i] = hi[0]
            output_hq[i] = hq[0]
            output_vi[i] = vi[0]
            output_vq[i] = vq[0]
        elif(k2>=input_length):
            output_hi[i] = hi[-1]
            output_hq[i] = hq[-1]
            output_vi[i] = vi[-1]
            output_vq[i] = vq[-1]
        elif(k1==k2):
            output_hi[i] = hi[k1]
            output_hq[i] = hq[k1]
            output_vi[i] = vi[k1]
            output_vq[i] = vq[k1]
        else:
            #output_hi[i] = interp1d([k1,k2], hi[k1:k2+1], 'linear')(k[i])
            #output_hq[i] = interp1d([k1,k2], hq[k1:k2+1], 'linear')(k[i])
            #output_vi[i] = interp1d([k1,k2], vi[k1:k2+1], 'linear')(k[i])
            #output_vq[i] = interp1d([k1,k2], vq[k1:k2+1], 'linear')(k[i])
            output_hi[i] = simple_interpld([k1,k2], hi[k1:k2+1], k[i])
            output_hq[i] = simple_interpld([k1,k2], hq[k1:k2+1], k[i])
            output_vi[i] = simple_interpld([k1,k2], vi[k1:k2+1], k[i])
            output_vq[i] = simple_interpld([k1,k2], vq[k1:k2+1], k[i])
        if(i%1000==0):
            print('.',end='')
    print(output_length,'points sampled!')
    #———————————————————————————#ADC采样数据按照分辨率截断
    output_hi = Digitalized(output_hi,Adc_Bit)
    output_hq = Digitalized(output_hq,Adc_Bit)
    output_vi = Digitalized(output_vi,Adc_Bit)
    output_vq = Digitalized(output_vq,Adc_Bit)
    #———————————————————————————#
    output = {'Time':output_time,'HI':output_hi,'HQ':output_hq,'VI':output_vi,'VQ':output_vq}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#采样时钟相位调整
def ADC_clock_phase_estimate(data,Adc_clock,Adc_SamplingRatio,Adc_Bit):
    n = 30                                              #定义粗调相位移动的步数
    period = 1/ (Adc_clock*Adc_SamplingRatio)           #计算一个信号周期
    u = 1.2*period/(n-1)                                #计算粗调相位移动的步长，粗调范围为1.2个时钟周期
    time_shift = [0.0]*n
    error      = [0.0]*n
    for i in range(n):
        print(i,'tims, ',end='')
        time_shift[i] = i*u + 2e-2                      #单位ns
        mydata5 = ADC(data['Time'][:2000],data['HI'][:2000],data['HQ'][:2000],data['VI'][:2000],data['VQ'][:2000],Adc_clock,Adc_SamplingRatio,time_shift[i],Adc_Bit)
        error[i] = np.mean( abs( mydata5['HI'] ) )
    temp = time_shift[error.index(max(error))]          #差值最大的点即是最佳采样相位
    #################################################################
    plt.figure(figsize=(16,5)).set_tight_layout(True)
    plt.subplot(1, 2, 1)
    plt.plot(np.array(time_shift)*1e3,np.array(error)*2,'go')
    plt.xlabel('ADC phase shift (ps)') 
    plt.ylabel('ADC sampled swing')
    #################################################################
    m = 25                                              #定义细调相位移动的步数
    u = 4*u/(m-1)                                       #计算细调相位移动的步长，细调范围为+/-2个粗调步长
    time_shift = [temp-2*u]*m
    error      = [0.0]*m
    for i in range(m):
        print(i+n,'tims, ',end='')
        time_shift[i] = time_shift[0] + i*u
        mydata5 = ADC(data['Time'][:2000],data['HI'][:2000],data['HQ'][:2000],data['VI'][:2000],data['VQ'][:2000],Adc_clock,Adc_SamplingRatio,time_shift[i],Adc_Bit)
        error[i] = np.mean( abs( mydata5['HI'] ) )
    temp = time_shift[error.index(max(error))]          #差值最大的点即是最佳采样相位
    print('Optimized ADC phase shift is',temp*1e3,'ps')
    #################################################################
    plt.subplot(1, 2, 2)
    plt.plot(np.array(time_shift)*1e3,np.array(error)*2,'bo')
    plt.xlabel('ADC phase shift (ps)') 
    plt.ylabel('ADC sampled swing')
    plt.show()
    #################################################################
    return(temp)
#####################################################################
#####################################################################
#####################################################################
#定义<色散估计>函数
def CDE(input_data,SampleRate,plot=False):
    '''
    该方法来自于Edem Ibragimov的论文"Blind Chromatic Dispersion Estimation Using a Spectrum of a
    Modulus Squared of the Transmitted Signal", ECOC 2012, Th.2.A.3
    '''
    fft_length = 2**10
    a = input_data['HI'][:fft_length]**2 +input_data['HQ'][:fft_length]**2 +input_data['VI'][:fft_length]**2 +input_data['VQ'][:fft_length]**2
    #———————————————————————————#数据FFT
    freq_resolution = SampleRate/fft_length
    frequency = np.arange(-fft_length/2,fft_length/2) *freq_resolution
    fpX = abs(fft(a)/fft_length)**2
    fpX[0] = fpX[1]             #去掉DC分量
    #———————————————————————————#绘制图像
    if(plot):
        fpY = np.append(fpX[int(fft_length/2):],fpX[:int(fft_length/2)])
        p_density = 10 *np.log10(fpY/freq_resolution)
        p_density = IIR(p_density,17/32,2)
        plt.plot(frequency,p_density,color='purple')
        plt.xlim(-30,30)
        #plt.ylim(-80,-20)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Power Density (dB/Hz)')
        plt.grid(color='gray',linestyle=':')
        plt.title('Frequency Spectum')
        plt.show()
    #———————————————————————————#对5GHz以下频率分量积分
    flag = round(5/freq_resolution)+2
    return(sum(fpX[1:flag])/1e3)
#####################################################################
#####################################################################
#####################################################################
#定义<色散补偿>函数
def CDC(input_data,wavelength,Distance,SampleRate,fft_length=2**10,trial=False):
    print('CD compensation',Distance/1e3,'km')
    c_ns = cons.c*1e-9	                    #光速，单位m/ns
    dispersion = -1 *cd_coef *Distance      #负号代表色散补偿
    #———————————————————————————#
    input_length = len(input_data['Time'])
    input_t = input_data['Time']
    input_h = input_data['HI'] +1j *input_data['HQ']
    input_v = input_data['VI'] +1j *input_data['VQ'] 
    if(trial):
        input_length = min(input_length, 2**14)
        input_t = input_t[:input_length]
        input_h = input_h[:input_length]
        input_v = input_v[:input_length]
    t = np.array([])
    H = np.array([])
    V = np.array([])
    #———————————————————————————#生成[-Fs/2, Fs/2)，间隔为freq_resolution的频带
    #调整频带顺序，配合FFT的频带顺序
    freq_resolution = SampleRate/fft_length
    frequencyVector = np.append(np.arange(0,fft_length/2),np.arange(-fft_length/2,0))*freq_resolution
    #———————————————————————————#计算每个频率的色散补偿量
    cdH = exp(-1j *dispersion *wavelength**2 *pi *frequencyVector**2 /c_ns)    #简化后的方程
    #———————————————————————————#数据处理
    i = 0
    while(i+fft_length <= input_length):
        group_t = input_t[i:i+fft_length]
        group_h = input_h[i:i+fft_length]
        group_v = input_v[i:i+fft_length]
        #———————————————————————#
        h = ifft( fft(group_h)*cdH )
        v = ifft( fft(group_v)*cdH )
        #———————————————#
    #因FFT/IFFT的有限长序列效应，第i次处理结果的前1/4和后1/4个数据误差极大，应该被丢弃，由第i-1和第i+1次数据处理来覆盖
    #———————————————#
    #第1次处理结果的前1/4被丢弃，输出为0
    #———————————————#
        t = np.append(t , group_t[int(fft_length/4):int(3*fft_length/4)])
        H = np.append(H , h      [int(fft_length/4):int(3*fft_length/4)])
        V = np.append(V , v      [int(fft_length/4):int(3*fft_length/4)])
        #———————————————————————#
        i = i + int(fft_length/2)
    #———————————————————————————#
    output = {'Time':t,'HI':H.real,'HQ':H.imag,'VI':V.real,'VQ':V.imag}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<MIMO初始化>函数
def mimo_init(pmdTaps):         #定义taps归零状态
    hxx = np.zeros(pmdTaps) +1j *np.zeros(pmdTaps)
    hxy = np.zeros(pmdTaps) +1j *np.zeros(pmdTaps)
    hyx = np.zeros(pmdTaps) +1j *np.zeros(pmdTaps)
    hyy = np.zeros(pmdTaps) +1j *np.zeros(pmdTaps)
    if  (pmdTaps %2==0):	    #偶数个taps
        hxx[int(pmdTaps/2)]   = 1/sqrt(2)
        hxx[int(pmdTaps/2-1)] = 1/sqrt(2)
        hyy[int(pmdTaps/2)]   = 1/sqrt(2)
        hyy[int(pmdTaps/2-1)] = 1/sqrt(2)
    else:			            #奇数个taps
        hxx[int((pmdTaps-1)/2)] = 1
        hyy[int((pmdTaps-1)/2)] = 1
    output = {'hxx':hxx,'hxy':hxy,'hyx':hyx,'hyy':hyy}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义平均平方(Mean Square)函数
'''
并将明显的偏离值（大于3倍平均值）丢弃掉，防止偶然的噪声影响结果
'''
def mean_sq_modulus(input_data):
    sq_mod = input_data.real**2 + input_data.imag**2
    cm     = np.mean(sq_mod)
    eff_ac = 0
    eff_le = 0
    for p in sq_mod:
        if(p <= 3* cm):
            eff_ac = eff_ac + p
            eff_le = eff_le + 1
    eff_avg = eff_ac / eff_le
    return(eff_avg)
#####################################################################
#####################################################################
#####################################################################
#定义<CMA>函数
'''
标准的CMA算法。在参数中增加了“para”，可将上一次CMA运算得到的H矩阵输入
到下一次CMA运算，进行多次迭代
'''
def CMA(input_data, step, para, plot=True):
    time = input_data['Time']
    xi   = input_data['HI']
    xq   = input_data['HQ']
    yi   = input_data['VI']
    yq   = input_data['VQ']
    input_length = len(time)
    if(input_length!= len(xi) or input_length!= len(xq) or input_length!= len(yi) or input_length!= len(yq)):
        print('Input time and data are not synchronized!!')
        return()
    #———————————————————————————#输入初始化
    cm = np.mean(xi**2 +xq**2 +yi**2 +yq**2)/2
    sqrt_cm = sqrt(cm)
    x  = (xi + 1j*xq) /sqrt_cm
    y  = (yi + 1j*yq) /sqrt_cm
    print('CMA equalization')
    print('Input RMS Modulus =',sqrt_cm)
    #———————————————————————————#taps初始化
    hxx = para['hxx']
    hxy = para['hxy']
    hyx = para['hyx']
    hyy = para['hyy']
    #———————————————————————————#
    pmdTaps  = len(hxx)
    length   = int(input_length -pmdTaps+1)
    output_x = np.zeros(length) +1j *np.zeros(length)
    output_y = np.zeros(length) +1j *np.zeros(length)
    error_x  = np.zeros(length)
    error_y  = np.zeros(length)
    del length
    #———————————————————————————#
    #constellation = std_constellation(mod,False) *sqrt_cm
    std_m = 1#np.unique(abs(constellation))
    print('Normalized RMS Modulus =',std_m)
    #———————————————————————————#
    n = 0
    print('step =',step)
    while(n+pmdTaps <= input_length):
        groupX = x[n:n+pmdTaps]
        groupY = y[n:n+pmdTaps]      
        output_x[n] = np.inner(hxx,groupX) + np.inner(hxy,groupY)
        output_y[n] = np.inner(hyx,groupX) + np.inner(hyy,groupY)
        #———————————————————————#
        error_x[n] = std_m**2 -abs(output_x[n])**2
        error_y[n] = std_m**2 -abs(output_y[n])**2
        #———————————————————————#Error is greater than 10*cm
        if (abs(error_x[n])>10*cm or abs(error_y[n])>10*cm):
            print('Error at',n)
            break
        #———————————————————————#
        hxx = hxx +step *error_x[n] *output_x[n] *np.conj(groupX)
        hxy = hxy +step *error_x[n] *output_x[n] *np.conj(groupY)
        hyx = hyx +step *error_y[n] *output_y[n] *np.conj(groupX)
        hyy = hyy +step *error_y[n] *output_y[n] *np.conj(groupY)
        n = n+1
    time = time[pmdTaps-1:]
    #———————————————————————————#结果显示
    print('MSE_X =',np.mean(abs(error_x)))
    print('MSE_Y =',np.mean(abs(error_y)))
    cm1 = np.mean(abs(output_x)**2 +abs(output_y)**2)/2
    print('Output RMS Modulus =',sqrt(cm1))
    print('-------------------------')
    #———————————————————————————#
    if(plot):
        plt.figure(figsize=(8,6)).set_tight_layout(True)
        plt.subplot(2, 2, 1)
        plt.scatter(time,-1*error_x,color='red',marker='.',alpha=0.2,s=1)#/cm
        plt.ylim(-1,1)
        plt.xlabel('Time (ns)')
        plt.ylabel('Square Error')
        plt.title('X Normalized Square Error')
        plt.hlines(0,min(time),max(time),color='yellow',linestyle='--')
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 2)
        plt.scatter(time,-1*error_y,            marker='.',alpha=0.2,s=1)#/cm
        plt.ylim(-1,1)
        plt.xlabel('Time (ns)')
        plt.ylabel('Square Error')
        plt.title('Y Normalized Square Error')
        plt.hlines(0,min(time),max(time),color='yellow',linestyle='--')
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 3)
        plt.scatter(time,abs(output_x),color='red',marker='.',alpha=0.2,s=1)#/sqrt_cm
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Modulus')
        plt.title('X Normalized Modulus')
        plt.hlines(1,min(time),max(time),color='yellow',linestyle='--')#/sqrt_cm
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 4)
        plt.scatter(time,abs(output_y),            marker='.',alpha=0.2,s=1)#/sqrt_cm
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Modulus')
        plt.title('Y Normalized Modulus')
        plt.hlines(1,min(time),max(time),color='yellow',linestyle='--')#/sqrt_cm
        plt.grid(color='gray',linestyle=':')
        plt.show()
    #———————————————————————————#结果输出
    para = {'hxx':hxx,'hxy':hxy,'hyx':hyx,'hyy':hyy}
    output = {'Time':time,'HI':sqrt_cm*output_x.real,'HQ':sqrt_cm*output_x.imag,'VI':sqrt_cm*output_y.real,'VQ':sqrt_cm*output_y.imag,'para':para,'target':[sqrt_cm,sqrt(cm1)]}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<RDA>函数
'''
多模收敛算法
'''
def RDA(input_data, mod, step, para, plot=True):
    time = input_data['Time']
    xi   = input_data['HI']
    xq   = input_data['HQ']
    yi   = input_data['VI']
    yq   = input_data['VQ']
    input_length = len(time)
    if(input_length!= len(xi) or input_length!= len(xq) or input_length!= len(yi) or input_length!= len(yq)):
        print('Input time and data are not synchronized!!')
        return()
    #———————————————————————————#输入初始化
    x  = xi + 1j*xq
    y  = yi + 1j*yq
    cm = np.mean(xi**2 +xq**2 +yi**2 +yq**2)/2
    #cm = (mean_sq_modulus(x) + mean_sq_modulus(y))/2
    sqrt_cm = sqrt(cm)
    print('RDA equalization')
    print('Input RMS Modulus =',sqrt_cm)
    #———————————————————————————#taps初始化
    hxx = para['hxx']
    hxy = para['hxy']
    hyx = para['hyx']
    hyy = para['hyy']
    #———————————————————————————#
    pmdTaps  = len(hxx)
    length   = int(input_length -pmdTaps+1)
    output_x = np.zeros(length) +1j *np.zeros(length)
    output_y = np.zeros(length) +1j *np.zeros(length)
    error_x  = np.zeros(length)
    error_y  = np.zeros(length)
    del length
    #———————————————————————————#
    constellation = std_constellation(mod,False) *sqrt_cm
    std_m = np.unique(abs(constellation))
    print('Target Multi Modulus =',std_m)
    #———————————————————————————#
    n = 0
    print('step =',step)
    while(n+pmdTaps <= input_length):
        groupX = x[n:n+pmdTaps]
        groupY = y[n:n+pmdTaps]      
        output_x[n] = np.inner(hxx,groupX) + np.inner(hxy,groupY)
        output_y[n] = np.inner(hyx,groupX) + np.inner(hyy,groupY)
        #———————————————————————#
        m_x = std_m[np.argmin(abs(std_m -abs(output_x[n])))]
        m_y = std_m[np.argmin(abs(std_m -abs(output_y[n])))]
        error_x[n] = m_x**2 -abs(output_x[n])**2
        error_y[n] = m_y**2 -abs(output_y[n])**2
        #———————————————————————#Error is greater than 10*cm
        if (abs(error_x[n])>10*cm or abs(error_y[n])>10*cm):
            print('Error at',n)
            break
        #———————————————————————#
        hxx = hxx +step *error_x[n] *output_x[n] *np.conj(groupX)/cm**2
        hxy = hxy +step *error_x[n] *output_x[n] *np.conj(groupY)/cm**2
        hyx = hyx +step *error_y[n] *output_y[n] *np.conj(groupX)/cm**2
        hyy = hyy +step *error_y[n] *output_y[n] *np.conj(groupY)/cm**2
        n = n+1
    time = time[pmdTaps-1:]
    #———————————————————————————#结果显示
    print('MSE_X =',np.mean(abs(error_x))/cm)
    print('MSE_Y =',np.mean(abs(error_y))/cm)
    cm1 = np.mean(abs(output_x)**2 +abs(output_y)**2)/2
    print('Output RMS Modulus =',sqrt(cm1))
    print('-------------------------')
    #———————————————————————————#
    if(plot):
        plt.figure(figsize=(8,6)).set_tight_layout(True)
        plt.subplot(2, 2, 1)
        plt.scatter(time,-1*error_x/cm,color='red',marker='.',alpha=0.2,s=1)#/cm
        plt.ylim(-1,1)
        plt.xlabel('Time (ns)')
        plt.ylabel('Square Error')
        plt.title('X Normalized Square Error')
        plt.hlines(0,min(time),max(time),color='yellow',linestyle='--')
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 2)
        plt.scatter(time,-1*error_y/cm,            marker='.',alpha=0.2,s=1)#/cm
        plt.ylim(-1,1)
        plt.xlabel('Time (ns)')
        plt.ylabel('Square Error')
        plt.title('Y Normalized Square Error')
        plt.hlines(0,min(time),max(time),color='yellow',linestyle='--')
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 3)
        plt.scatter(time,abs(output_x)/sqrt_cm,color='red',marker='.',alpha=0.2,s=1)#/sqrt_cm
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Modulus')
        plt.title('X Normalized Modulus')
        plt.hlines(std_m/sqrt_cm,min(time),max(time),color='yellow',linestyle='--')#/sqrt_cm
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 4)
        plt.scatter(time,abs(output_y)/sqrt_cm,            marker='.',alpha=0.2,s=1)#/sqrt_cm
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Modulus')
        plt.title('Y Normalized Modulus')
        plt.hlines(std_m/sqrt_cm,min(time),max(time),color='yellow',linestyle='--')#/sqrt_cm
        #plt.plot(time,std_m*sqrt_cm,color='yellow',linestyle='--')
        plt.grid(color='gray',linestyle=':')
        plt.show()
    #———————————————————————————#结果输出
    para = {'hxx':hxx,'hxy':hxy,'hyx':hyx,'hyy':hyy}
    output = {'Time':time,'HI':output_x.real,'HQ':output_x.imag,'VI':output_y.real,'VQ':output_y.imag,'para':para,'target':[sqrt_cm,sqrt(cm1)]}
    return(output)
#####################################################################
#####################################################################
#####################################################################        
#定义DD-LMS函数
def DD_LMS(input_data, mod, step, para, plot=True):
    time = input_data['Time']
    xi   = input_data['HI']
    xq   = input_data['HQ']
    yi   = input_data['VI']
    yq   = input_data['VQ']
    input_length = len(time)
    if(input_length!= len(xi) or input_length!= len(xq) or input_length!= len(yi) or input_length!= len(yq)):
        print('Input time and data are not synchronized!!')
        return()
    #———————————————————————————#输入初始化
    x  = xi + 1j*xq
    y  = yi + 1j*yq
    cm = np.mean(xi**2 +xq**2 +yi**2 +yq**2)/2
    #cm = (mean_sq_modulus(x) + mean_sq_modulus(y))/2
    sqrt_cm = sqrt(cm)
    print('Average modulus =',sqrt_cm)
    #———————————————————————————#taps初始化
    hxx = para['hxx']
    hxy = para['hxy']
    hyx = para['hyx']
    hyy = para['hyy']
    #———————————————————————————#
    pmdTaps  = len(hxx)
    length   = int(input_length -pmdTaps+1)
    output_x = np.zeros(length) +1j *np.zeros(length)
    output_y = np.zeros(length) +1j *np.zeros(length)
    error_x  = np.zeros(length) +1j *np.zeros(length)
    error_y  = np.zeros(length) +1j *np.zeros(length)
    det_x    = np.zeros(length) +1j *np.zeros(length)
    det_y    = np.zeros(length) +1j *np.zeros(length)
    del length
    #———————————————————————————#
    constellation = std_constellation(mod,False)*sqrt_cm
    std_m = np.unique(abs(constellation))
    #———————————————————————————#
    n = 0
    print('step =',step)
    while(n+pmdTaps <= input_length):
        groupX     = x[n:n+pmdTaps]
        groupY     = y[n:n+pmdTaps]
        output_x[n]= (np.inner(hxx,groupX) + np.inner(hxy,groupY)) #*exp(2i*pi*freq_offset*time[n+pmdTaps-1])
        output_y[n]= (np.inner(hyx,groupX) + np.inner(hyy,groupY)) #*exp(2i*pi*freq_offset*time[n+pmdTaps-1])
        #———————————————————————#
        flag_x     = np.argmin(abs(constellation -output_x[n]))
        flag_y     = np.argmin(abs(constellation -output_y[n]))
        det_x  [n] = constellation[flag_x]
        det_y  [n] = constellation[flag_y]
        error_x[n] = det_x[n] -output_x[n]
        error_y[n] = det_y[n] -output_y[n]
        #———————————————————————#
        hxx = hxx +step *error_x[n] *np.conj(groupX)/cm
        hxy = hxy +step *error_x[n] *np.conj(groupY)/cm
        hyx = hyx +step *error_y[n] *np.conj(groupX)/cm
        hyy = hyy +step *error_y[n] *np.conj(groupY)/cm
        n = n+1
    time = time[pmdTaps-1:]
    #———————————————————————————#结果显示
    print('MSE_X =',np.mean(abs(error_x)**2)/cm)
    print('MSE_Y =',np.mean(abs(error_y)**2)/cm)
    cm1 = np.mean(abs(output_x)**2 +abs(output_y)**2)/2
    print('Output RMS Modulus =',sqrt(cm1))
    print('-------------------------')
    #———————————————————————————#
    if(plot):
        plt.figure(figsize=(8,6)).set_tight_layout(True)
        plt.subplot(2, 2, 1)
        plt.scatter(time,abs(error_x)**2/cm,color='red',marker='.',alpha=0.2,s=1)
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Square Error')
        plt.title('X Normalized Square Error')
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 2)
        plt.scatter(time,abs(error_y)**2/cm,marker='.',alpha=0.2,s=1)
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Square Error')
        plt.title('Y Normalized Square Error')
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 3)
        plt.scatter(time,abs(output_x)/sqrt_cm,color='red',marker='.',alpha=0.2,s=1)
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Modulus')
        plt.title('X Normalized Modulus')
        plt.hlines(std_m/sqrt_cm,min(time),max(time),color='yellow',linestyle='--')
        plt.grid(color='gray',linestyle=':')
        #———————————————————————————#
        plt.subplot(2, 2, 4)
        plt.scatter(time,abs(output_y)/sqrt_cm,marker='.',alpha=0.2,s=1)
        plt.ylim(0,2)
        plt.xlabel('Time (ns)')
        plt.ylabel('Modulus')
        plt.title('Y Normalized Modulus')
        plt.hlines(std_m/sqrt_cm,min(time),max(time),color='yellow',linestyle='--')
        plt.grid(color='gray',linestyle=':')
        plt.show()
    #———————————————————————————#结果输出
    para = {'hxx':hxx,'hxy':hxy,'hyx':hyx,'hyy':hyy}
    output = {'Time':time,'HI':output_x.real,'HQ':output_x.imag,'VI':output_y.real,'VQ':output_y.imag,'para':para}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#载波频差估计（Viterbi-Viterbi/Frequency Search）
def freq_estimate(I , Q, fft_length, freqE_SampleRate, plot=True):
    input_length = len(I)
    if(input_length!= len(Q)):
        print('Input data are not synchronized!!')
        return()
    #———————————————————————————#
    if(input_length < fft_length):
        print('Input data is shorter than FFT length!!')
        return()
    #———————————————————————————#
    data = (I[input_length-fft_length:] + 1j*Q[input_length-fft_length:])/128
    fpX = abs(fft(data**4)/fft_length)**2
    #采用Viterbi-Viterbi算法，通过4次方消去调制相位，频谱最高的峰就是载波频差
    #同样适用于8QAM，16QAM以及其他调制码型
    #数字电路计算模平方比计算模容易，少一步开方
    #fpX[1] = fpX[1]/2
    fpX = np.append(fpX[int(fft_length/2):] ,fpX[:int(fft_length/2)])	#整理FFT的输出结果
    #———————————————————————————#计算对应的频率值
    freq_resolution = freqE_SampleRate /fft_length /4	#Viterbi-Viterbi算法信号经过4次方运算，频率乘了4倍
    frequency =  np.arange(-fft_length/2,fft_length/2)*freq_resolution
    #———————————————————————————#限制频率范围+/-5GHz
    start = int(fft_length/2 -round(5/freq_resolution))
    end   = int(fft_length/2 +round(5/freq_resolution))
    frequency = frequency[start:end]
    fpX       = fpX[start:end]
    freq_offs = frequency[np.argmax(fpX)]
    #———————————————————————————#绘制FFT输出结果
    fpY = 10*np.log10(fpX)
    if(plot):
        plt.plot(frequency,fpY)
        plt.xlim(-5,5)
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Magnitud (dB)')
        plt.title('Frequency Offset Estimation')
        plt.grid(color='gray',linestyle=':')
        plt.plot(freq_offs,max(fpY),'md',label='Frequency Offset = '+str(round(freq_offs,2))+'GHz')
        plt.legend(loc='lower left')
    #———————————————————————————#
    print('Frequency offset is',freq_offs*1e3,'MHz, +/-',freq_resolution/2*1e3,'MHz')
    return(freq_offs)
#####################################################################
#####################################################################
#####################################################################
#定义<载波频偏补偿>函数
def freq_recover(input_data, freq_offset_X, freq_offset_Y):
    time = input_data['Time']
    hi   = input_data['HI']
    hq   = input_data['HQ']
    vi   = input_data['VI']
    vq   = input_data['VQ']
    input_length = len(time)
    if(input_length!= len(hi) or input_length!= len(hq) or input_length!= len(vi) or input_length!= len(vq)):
        print('Input time and data are not synchronized!!')
        return()
    #———————————————————————————#X载波频偏补偿
    rotation_phase = 2*pi*freq_offset_X*time
    sine = np.sin(rotation_phase +pi/4)
    cosn = np.cos(rotation_phase +pi/4)
    hi_o = hi *cosn - hq *sine
    hq_o = hi *sine + hq *cosn
    #———————————————————————————#Y载波频偏补偿
    rotation_phase = 2*pi*freq_offset_Y*time
    sine = np.sin(rotation_phase +pi/4)
    cosn = np.cos(rotation_phase +pi/4)
    vi_o = vi *cosn - vq *sine
    vq_o = vi *sine + vq *cosn
    #———————————————————————————#
    output = {'Time':time,'HI':hi_o,'HQ':hq_o,'VI':vi_o,'VQ':vq_o}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<卷曲显示>函数
def wrap(input_data, n):
    j = np.array([])
    for i in input_data:
        if(i==0):
            j = np.append(j, 0)
        else:
            j = np.append(j, i %(n*np.sign(i)))
    return(j)
#####################################################################
#####################################################################
#####################################################################
#定义<载波相位去除>函数（Viterbi-Viterbi）
def CPR_VV(input_data, half_window=16, IIR=0, plot=True):
    time = input_data['Time']
    xi   = input_data['HI']
    xq   = input_data['HQ']
    yi   = input_data['VI']
    yq   = input_data['VQ']
    input_length = len(time)
    if(input_length!= len(xi) or input_length!= len(xq) or input_length!= len(yi) or input_length!= len(yq)):
        print('Input time and data are not synchronized!!')
        return()
    #———————————————————————————#初始化输入输出数据
    window   = 2 *half_window +1
    input_x  = xi +1j*xq
    input_y  = yi +1j*yq
    output_x = np.array([])
    output_y = np.array([])
    output_t = np.array([])
    phaseX   = np.array([])
    phaseY   = np.array([])
    phaseX_temp = 0
    phaseY_temp = 0
    #———————————————————————————#计算移动平均相位
    print('Performing Phase Average',end='')
    n = 0
    while(n +window <= input_length):
        groupX      = input_x[n:n+window] /abs(input_x[n:n+window]) *exp(-1j*phaseX_temp)
        groupY      = input_y[n:n+window] /abs(input_y[n:n+window]) *exp(-1j*phaseY_temp)
        phaseX_delt = np.angle(sum(groupX**4)) /4     #先求和（等效于取平均）再计算辐角，运算量小于先计算辐角再平均
        phaseY_delt = np.angle(sum(groupY**4)) /4
        #———————————————————————#Perfomring Unwrapping
        if  (phaseX_delt < -pi/4):
            phaseX_delt = phaseX_delt +pi/2
            print(n, 'cycle slipping', phaseX_delt)
        elif(phaseX_delt > +pi/4):
            phaseX_delt = phaseX_delt -pi/2
            print(n, 'cycle slipping', phaseX_delt)
        #———————————————————————#一阶IIR滤波
        phaseX_temp = phaseX_temp + phaseX_delt *(1-IIR)    #简化IIR滤波
        phaseY_temp = phaseY_temp + phaseY_delt *(1-IIR)    #简化IIR滤波
        #———————————————————————#
        output_x = np.append(output_x, input_x[n +half_window] *exp(-1j*phaseX_temp))
        output_y = np.append(output_y, input_y[n +half_window] *exp(-1j*phaseY_temp))
        output_t = np.append(output_t, time   [n +half_window])
        phaseX   = np.append(phaseX, phaseX_temp)
        phaseY   = np.append(phaseY, phaseY_temp)
        #———————————————————————#
        if(n %200 == 0):
            print('.',end='')
        n = n+1
    print('')
    #———————————————————————————#增加额外pi/4相移
    output_x = output_x *exp(-1j*pi/4)
    output_y = output_y *exp(-1j*pi/4)
    #———————————————————————————#
    if(plot):
        plt.scatter(output_t,wrap(phaseX/pi, 2),c='m',marker='.',s=1,label='X-pol Phase')
        plt.scatter(output_t,wrap(phaseY/pi, 2)      ,marker='.',s=1,label='Y-pol Phase')
        plt.ylim(-2.2,2.2)
        plt.title('Signal Phase')
        plt.xlabel('Time (ns)')
        plt.ylabel('Phase (pi)')
        plt.grid(color='gray',linestyle=':')
        plt.legend(loc='upper left')
        plt.show()
    #———————————————————————————#
    output = {'Time':output_t,'HI':output_x.real,'HQ':output_x.imag,'VI':output_y.real,'VQ':output_y.imag,'phaseX':phaseX/pi,'phaseY':phaseY/pi}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<载波相位去除>函数（Blind Phase Searching）
def CPR_BPS(input_data, mod='QPSK', N_step=16, window=16, move=8, IIR=17/32, unlock=False, plot=True, debug=False):
    aaa = tm.time()
    time = input_data['Time']
    xi   = input_data['HI']
    xq   = input_data['HQ']
    yi   = input_data['VI']
    yq   = input_data['VQ']
    input_length = len(time)
    if(input_length!= len(xi) or input_length!= len(xq) or input_length!= len(yi) or input_length!= len(yq)):
        print('Input time and data are not synchronized!!')
        return()
    #———————————————————————————#确定搜索参数
    window = window //2 *2 +1
    #move   = move   //2 *2 +1
    N_step = N_step //2 *2
    print('Averg Window =',window,'symbols')
    print('Search Steps =',N_step)
    if(window<move):
        print('Average window is smaller than move step!!')
        return()
    #———————————————————————————#星座图归一化
    sqrt_cm = np.sqrt(np.mean(xi**2 +xq**2 +yi**2 +yq**2)/2)
    constellation = std_constellation(mod,plot=0)*sqrt_cm
    #———————————————————————————#根据星座图旋转对称性确定搜索范围
    if(mod=='QPSK' or mod=='8QAM' or mod=='16QAM' or mod=='32QAM' or mod=='64QAM'):
        step_fix = pi/2/N_step
    if(mod=='BPSK' or mod=='SP-16QAM'):
        step_fix = pi/N_step
    print('Each Step = pi/',int(pi/step_fix))
    #———————————————————————————#初始化输入输出数据
    input_x  = xi +1j*xq
    input_y  = yi +1j*yq
    output_x = np.array([])
    output_y = np.array([])
    output_t = np.array([])
    phaseX   = np.array([])
    phaseY   = np.array([])
    phaseX_temp = 0
    phaseY_temp = 0
    step = step_fix
    lock_flag = 0
    #———————————————————————————#
    print('Blind Phase Searching',end='')
    #———————————————————————————#
    k = 0
    if(debug):
        k_max = 300*move
    else:
        k_max = input_length-window
    while(k <= k_max):
        groupX = input_x[k : k+window] *exp(-1j*phaseX_temp)
        groupY = input_y[k : k+window] *exp(-1j*phaseY_temp)
        errorX = np.array([])
        errorY = np.array([])
        #———————————————————————#
        for n in range(int(-N_step/2),int(N_step/2)):
            errorX = np.append(errorX, hard_dec(groupX *exp(-1j*step*n),constellation))
        for n in range(int(-N_step/2),int(N_step/2)):
            errorY = np.append(errorY, hard_dec(groupY *exp(-1j*step*n),constellation))
        #———————————————————————#8ms
        flagX  = np.argmin(errorX)
        flagY  = np.argmin(errorY)
        phaseX_delta = step *(flagX -N_step/2)
        phaseY_delta = step *(flagY -N_step/2)
        #———————————————————————#一阶IIR滤波
        phaseX_init = phaseX_temp
        phaseY_init = phaseY_temp
        #phaseY_temp = (phaseY_temp +phaseY_delta) *(1-IIR) + phaseY_temp *IIR	#IIR滤波
        phaseX_temp = phaseX_temp +phaseX_delta*(1-IIR)			#简化IIR滤波
        phaseY_temp = phaseY_temp +phaseY_delta*(1-IIR)			#简化IIR滤波
        #———————————————————————#对输出相位插值，减小计算量
        start = int(k +(window+1)/2 -move)      #int(k +(window-move)/2)
        end   = int(k +(window+1)/2)            #int(k +(window+move)/2)
        phaseX_move = simple_interpld([start-1,end-1], [phaseX_init,phaseX_temp], np.arange(start,end))
        phaseY_move = simple_interpld([start-1,end-1], [phaseY_init,phaseY_temp], np.arange(start,end))
        #———————————————————————#输出
        output_x = np.append(output_x, input_x[start:end] *exp(-1j *phaseX_move))
        output_y = np.append(output_y, input_y[start:end] *exp(-1j *phaseY_move))
        output_t = np.append(output_t, time[start:end])
        phaseX   = np.append(phaseX  , phaseX_move)
        phaseY   = np.append(phaseY  , phaseY_move)
        if(int(k/move) %20 == 0):
            print('.',end='')
        if(debug):
            print(int(k/move),'loop'   , ', length',end-start)
            print('w_start', k         , ', start',start)
            print('w_centr', int(k+(window-1)/2))
            print('w_end  ', k+window-1, ', end  ',end-1)
            print('--------------------')
        k = k +move
        #———————————————————————#锁定后缩小搜索范围，提高精度
        if  (k/move >=10 and max(errorX[flagX],errorY[flagY])<=4*sqrt_cm**2):
            if  (lock_flag):
                continue
            if  (mod=='QPSK' or mod=='8QAM' or mod=='16QAM' or mod=='32QAM' or mod=='64QAM'):
                step = step_fix*0.75
            elif(mod=='BPSK' or mod=='SP-16QAM'):
                step = step_fix*0.75*0.5
            print('Locked, Update Steps to pi/',round(pi/step))
            lock_flag = 1
        elif(not unlock):       #锁定后是否会因误差过大再次大范围搜索
            continue
        else:
            if(not lock_flag):
                continue
            step = step_fix
            print('Restore Steps to pi/',round(pi/step))
            lock_flag = 0
    #———————————————————————————#代码运行时间
    print('')
    bbb = tm.time()
    print(str(bbb-aaa),'s')
    #———————————————————————————#
    if(plot):
        plt.plot(output_t,wrap(phaseX/pi, 2),'m:',label='X-pol Phase')
        plt.plot(output_t,wrap(phaseY/pi, 2),'--',label='Y-pol Phase')
        plt.ylim(-2.2,2.2)
        plt.title('Signal Phase')
        plt.xlabel('Time (ns)')
        plt.ylabel('Phase (pi)')
        plt.grid(color='gray',linestyle=':')
        plt.legend(loc='upper left')
        plt.show()
    #———————————————————————————#
    output = {'Time':output_t,'HI':output_x.real,'HQ':output_x.imag,'VI':output_y.real,'VQ':output_y.imag,'phaseX':phaseX/pi,'phaseY':phaseY/pi}
    return(output)
'''
        #———————————————————————#Debug绘图
        if(debug and k/move>=100 and k/move<110):
            plt.plot(np.arange(-N_step/2,N_step/2)*step/pi,errorX/sqrt_cm**2,'m:')
            plt.plot(np.arange(-N_step/2,N_step/2)*step/pi,(errorY+100)/sqrt_cm**2,'--')
            plt.scatter(phaseX_delta/pi,min(errorX)/sqrt_cm**2,color='m',label=str(round(phaseX_temp/pi,2))+'pi')
            plt.scatter(phaseY_delta/pi,(min(errorY)+100)/sqrt_cm**2    ,label=str(round(phaseY_temp/pi,2))+'pi')          
            plt.xlabel('Phase (pi)')
            plt.ylabel('Error (a.u.)')
            plt.grid(color='gray',linestyle=':')
            plt.legend(loc='upper right')
            plt.show()
            
if(1):
    plt.scatter(cpr_output['Time'], wrap(cpr_output['phaseX'], 2), c='m', marker='.', s=1, label='X-pol Phase')
    plt.scatter(cpr_output['Time'], wrap(cpr_output['phaseY'], 2)       , marker='.', s=1, label='Y-pol Phase')
    plt.ylim(-2.2,2.2)
    #plt.xlim(150,200)
    plt.title('Signal Phase')
    plt.xlabel('Time (ns)')
    plt.ylabel('Phase (pi)')
    plt.grid(color='gray',linestyle=':')
    plt.legend(loc='upper left')
    plt.show()
'''
#####################################################################
#####################################################################
#####################################################################
#####################################################################
def encod_PS_qam(data1, data2=0, data3=0, data4=0, mod='PS-16QAM'):
    factor2 = 3/3.2   #用于16QAM补偿调制器非线性
    #———————————————————————————#    
    input_length = len(data1)
#    I = np.array([0.0]*input_length)
#    Q = np.array([0.0]*input_length)
    #———————————————————————————#
    if(mod == 'PS-16QAM'):
        if(input_length!= len(data2) or input_length!= len(data3) or input_length!= len(data4)):
            print('Input data are not synchronized!!')
            return()
        a = 0.5/3 *factor2
        X = np.array([0.5+0.5j,0.5-0.5j,-0.5-0.5j,-0.5+0.5j,
                      0.5+a*1j,0.5-a*1j,-0.5-a*1j,-0.5+a*1j,
                      a+0.5j  ,a-0.5j  ,-a-0.5j  ,-a+0.5j  ,
                      a+a*1j  ,a-a*1j  ,-a-a*1j  ,-a+a*1j  ,])
        Px = np.array([0.1,0.1,0.1,0.1,                         #设定每个symbol出现的概率
                       0.2,0.2,0.2,0.2,
                       0.2,0.2,0.2,0.2,
                       0.3,0.3,0.3,0.3])
        Px = Px/sum(Px)                                         #概率归一化
        SPx = Px.cumsum(0)
        mydata = np.array([])
        #———————————————————————#
        for i in range(0,input_length):
            temp = sum( np.random.uniform(0,1)>=SPx )
            mydata = np.append(mydata,X[temp])
    #———————————————————————————#
    I = mydata.real
    Q = mydata.imag
    return([I,Q])
#####################################################################
#####################################################################
#####################################################################
#####################################################################













