# -*- coding: UTF-8 -*-
#####################################################################
#Author: Austin Zhang
#Time: 2020/12/01
#Email: austinmill2010@gmail.com
#Website: https://github.com/austinmill/Coherent-Communication
#####################################################################
#####################################################################
if(1):
    import math
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
    from DSP import *
    from Test import *
    cd_coef = 17e3	#光纤色散系数，单位ns/m^2
#####################################################################
#####################################################################
#####################################################################
#定义<单偏振IQ调制器输出>函数
def IQ_MZ(data_i,data_q,bias_i,bias_q,bias_phase,v_pi,imbal_i=0,imbal_q=0,imbal_iq=0,typ='diff'):
    E_in = 1	                                    #输入归一化电场
    data_i_p =  1/2*data_i/v_pi                     #输入数据差分，并对v_pi归一化
    data_i_n = -1/2*data_i/v_pi
    data_q_p =  1/2*data_q/v_pi
    data_q_n = -1/2*data_q/v_pi
    #———————————————————————————#
    if  (typ=='diff'):
        bias_i_p = 0.5*bias_i/v_pi	            #偏置电压差分输入，并对v_pi归一化
        bias_i_n = -0.5*bias_i/v_pi	            #偏置电压采用闭环控制，总是保持平衡
        bias_q_p = 0.5*bias_q/v_pi
        bias_q_n = -0.5*bias_q/v_pi
        bias_phase_p = 0.5*bias_phase/v_pi
        bias_phase_n = -0.5*bias_phase/v_pi
    elif(typ=='single'):
        bias_i_p = 1.0*bias_i/v_pi	            #偏置电压单端输入，并对v_pi归一化
        bias_i_n = 0
        bias_q_p = 1.0*bias_q/v_pi
        bias_q_n = 0
        bias_phase_p = 1.0*bias_phase/v_pi
        bias_phase_n = 0
    elif(typ=='siph'):
        bias_i_p = bias_i**2/v_pi**2	            #偏置电压单端输入，并对v_pi归一化。采用加热形式，与电压平方成正比
        bias_i_n = 0
        bias_q_p = bias_q**2/v_pi**2
        bias_q_n = 0
        bias_phase_p = bias_phase**2/v_pi**2
        bias_phase_n = 0
    #———————————————————————————#
    angle_i_p = pi*(data_i_p + bias_i_p)	    #波导上产生的相移
    angle_i_n = pi*(data_i_n + bias_i_n)
    angle_phase_p = pi*bias_phase_p
    alpha = sqrt((1+imbal_i)**2 +1)*sqrt(0.5)
    MZ_i = E_in/sqrt(8) *exp(-1j*angle_phase_p) *1/alpha *(exp(-1j*angle_i_p) + (1+imbal_i)*exp(-1j*angle_i_n))
    #1/sqrt(8)表示IQ分束，IP/IN分束，IP/IN合束共3个耦合器带来的损耗
    #i_p/i_n之间的减号代表合束器的pi相位跃变
    #———————————————————————————#
    angle_q_p = pi*(data_q_p + bias_q_p)
    angle_q_n = pi*(data_q_n + bias_q_n)
    angle_phase_n = pi*bias_phase_n
    alpha = sqrt((1+imbal_q)**2 +1)*sqrt(0.5)
    MZ_q = E_in/sqrt(8) *exp(-1j*angle_phase_n) *1/alpha *(exp(-1j*angle_q_p) + (1+imbal_q)*exp(-1j*angle_q_n))
    #———————————————————————————#
    alpha = sqrt((1+imbal_iq)**2 +1)                #imbal_iq表示I/Q调制器的不平衡
    IQP   = 1/alpha *(MZ_i + (1+imbal_iq)*MZ_q)     #I/Q平衡时alpha=sqrt(2)，代表IQ合束耦合器损耗
    if  (typ=='diff'):
        IQP = IQP *exp(-1j*pi/4)	            #IQ合束器将输出相位旋转pi/4，偏置电压差分输入时将星座图转到水平位置
    #———————————————————————————#
    output = {'IQP':IQP,'I':MZ_i,'Q':MZ_q}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#WSS滤波器
def WSS(input_time,input_h,input_v,wss_resp,SampleRate=1e6,plot=True):
    #———————————————————————————#将输入信号FFT
    input_length = len(input_h)
    fft_length = 2** math.ceil(np.log2(input_length))
    input_h = np.append(input_h,np.zeros(fft_length -input_length))
    input_v = np.append(input_v,np.zeros(fft_length -input_length))
    #———————————————————————————#
    freq_resolution = SampleRate/fft_length
    print('Frequency resolution is',freq_resolution*1e3,'MHz')
    frequency = np.arange(-fft_length/2,fft_length/2)*freq_resolution
    #———————————————————————————#
    wss_freq = wss_resp[0]
    wss_resp = wss_resp[1]
    Z = zip(wss_freq,wss_resp)
    Z = sorted(Z,reverse=False)
    freq_sort,resp_sort = zip(*Z)
    wss_freq = np.array(freq_sort)
    wss_resp = np.array(resp_sort)
    #———————————————————————————#在WSS频谱响应上添加0频偏的插损
    if wss_freq[0] != 0:
        wss_freq = np.insert(wss_freq, 0, 0)
        wss_resp = np.insert(wss_resp, 0, 0)
    #———————————————————————————#确定WSS阻带衰减
    background_loss = wss_resp[np.where(abs(wss_freq)==max(abs(wss_freq)))][0]
    RespH = np.ones(int(fft_length/2)) *background_loss
    #———————————————————————————#对WSS频谱响应插值
    fl = interp1d(wss_freq, wss_resp, 'quadratic')
    flag = int(wss_freq[-1]/freq_resolution)+1
    upsample = np.arange(0,flag)*freq_resolution
    RespH[:flag] = fl(upsample)
    #———————————————————————————#计算负频率频谱
    RespH_N = np.append(RespH[1:],RespH[-1])[::-1]
    #PhaseH = Arg( c(RespH_N,RespH) )-20
    #———————————————————————————#输出WSS频响图
    if(plot):
        plt.plot(frequency,np.append(RespH_N,RespH),color='blue')
        plt.scatter(wss_freq,wss_resp,color='red',s=20)
        plt.xlim(-50,50)
        plt.title('WSS Response')
        plt.xlabel('Frequency offset (GHz)') 
        plt.ylabel('Magnitud (dB)')
        plt.grid(color='gray',linestyle=':')
        plt.show()
    #———————————————————————————#
    RespH = np.append(RespH,RespH_N)
    RespH = 10**(RespH/20)
    X = ifft( fft(input_h)*RespH )
    Y = ifft( fft(input_v)*RespH )
    output = {'Time':input_time,'H':X[:input_length],'V':Y[:input_length]}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<ASE注入>函数
def AWGN(time, input_h, input_v, OSNR=40, SampleRate=1):
    input_length = len(time)
    #———————————————————————————#
    Es = np.mean(abs(input_h)**2 +abs(input_v)**2)
    OSNR_BW = 12.5					#对应OSNR噪声参考带宽0.1nm
    noise_BW = SampleRate
    N0_x = Es/ (10**(OSNR/10))/2 *noise_BW /OSNR_BW 
    print('X-pol noise is',N0_x,'mW')
    N0_y = N0_x
    print('Y-pol noise is',N0_y,'mW')
    #———————————————————————————#
    noise_h = np.random.normal(0,sqrt(N0_x/2),input_length) +np.random.normal(0,sqrt(N0_x/2),input_length)*(0+1j)
    noise_v = np.random.normal(0,sqrt(N0_y/2),input_length) +np.random.normal(0,sqrt(N0_y/2),input_length)*(0+1j)
    output_h = input_h +noise_h
    output_v = input_v +noise_v
    #———————————————————————————#
    output = {'Time':time,'H':output_h,'V':output_v,'ASE_H':noise_h,'ASE_V':noise_v}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<色散注入>函数
def CD(time, input_h, input_v, wavelength, fiber_length, SampleRate, show=True):
    c_ns = cons.c*1e-9	        #光速，单位m/ns
    input_length = len(time)
    dispersion = cd_coef *fiber_length
    #———————————————————————————#
    fft_length = 2** math.ceil(np.log2(input_length))
    input_h = np.append(input_h,np.zeros(fft_length -input_length))
    input_v = np.append(input_v,np.zeros(fft_length -input_length))
    #———————————————————————————#生成[-Fs/2, Fs/2)，间隔为freq_resolution的频带
    #改变频带排序，配合FFT
    freq_resolution = SampleRate/fft_length
    frequencyVector = np.append(np.arange(0,fft_length/2),np.arange(-fft_length/2,0))*freq_resolution
    #———————————————————————————#计算每个频带的色散补偿量
    #cdH= exp(-1j *dispersion *wavelength**2 *(2*pi *frequencyVector)**2 /(4 *pi *c_ns))
    cdH = exp(-1j *dispersion *wavelength**2 *pi *frequencyVector**2 /c_ns)    #简化后的方程
    print(dispersion/1e6,'ps/nm CD injected!!!')
    #———————————————————————————#
    X = ifft( fft(input_h)*cdH )
    Y = ifft( fft(input_v)*cdH )
    output = {'Time':time,'H':X[:input_length],'V':Y[:input_length]}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<简化的DGD注入>函数
def simp_DGD(time, H, V, mean_dgd, Sim_SampleRatio, Baudrate, stochastic=True):
    input_length = len(time)
    vi = np.real(V)
    vq = np.imag(V)
    time = abs(time)
    #———————————————————————————#
    dgd = mean_dgd  #平均DGD，单位ns
    if(stochastic):
        dgd = np.random.normal(0,abs(mean_dgd))
        if(abs(dgd)>=5*abs(mean_dgd)):
            dgd = 0
    n = round(dgd *Sim_SampleRatio *Baudrate)
    dgd = n /(Sim_SampleRatio* Baudrate)
    print(dgd*1e3,'ps DGD injected!!!')
    n = abs(n)
    #———————————————————————————#
    if  (dgd==0):
        pass
    elif(dgd <0):
        vi = np.append(vi[n:],vi[:n])
        vq = np.append(vq[n:],vq[:n])
    else:
        vi = np.append(vi[-n:],vi[:-n])
        vq = np.append(vq[-n:],vq[:-n])
    output = {'Time':time,'H':H,'V':vi+1j*vq}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<光纤偏振旋转>函数
def rotate(phase):
    phase = float(phase)
    m = np.array([[np.cos(phase),-np.sin(phase)],[np.sin(phase),np.cos(phase)]])
    return(m)
#rotate(pi)
#####################################################################
#####################################################################
#####################################################################
#定义<光纤相位延迟>函数
def phase_delay(phase):
    phase = float(phase)
    m = np.array([[exp(-1j*phase/2),0],[0,exp(1j*phase/2)]])
    return(m)
#phase_delay(pi)
#####################################################################
#####################################################################
#####################################################################
#定义<ICR接收>函数
def ICR(time, H, V, lo_laser, phase_iq=pi/2, AGC=True, gain=[1,1,1,1], oc=0.125):
    phaseX_initial = np.random.uniform(0,2*pi)		        #LO光初始相位
    phaseY_initial = np.random.uniform(0,2*pi)		        #LO光初始相位
    #———————————————————————————#
    lo_h = 1/sqrt(2) *lo_laser *exp(-1j*phaseX_initial)
    lo_v = 1/sqrt(2) *lo_laser *exp(-1j*phaseY_initial)
    #———————————————————————————#
    hi = abs(H/2+ lo_h/2                 )**2 - abs(H/2+ lo_h/2                 *exp(1j*pi))**2
    hq = abs(H/2+ lo_h/2*exp(1j*phase_iq))**2 - abs(H/2+ lo_h/2*exp(1j*phase_iq)*exp(1j*pi))**2
    vi = abs(V/2+ lo_v/2                 )**2 - abs(V/2+ lo_v/2                 *exp(1j*pi))**2
    vq = abs(V/2+ lo_v/2*exp(1j*phase_iq))**2 - abs(V/2+ lo_v/2*exp(1j*phase_iq)*exp(1j*pi))**2
    #———————————————————————————#
    if(AGC):
        gain[0] = oc/sqrt(np.mean(abs(hi)**2))
        gain[1] = oc/sqrt(np.mean(abs(hq)**2))
        gain[2] = oc/sqrt(np.mean(abs(vi)**2))
        gain[3] = oc/sqrt(np.mean(abs(vq)**2))
    hi = hi*gain[0]
    hq = hq*gain[1]
    vi = vi*gain[2]
    vq = vq*gain[3]
    #———————————————————————————#
    print('Max HI is',max(abs(hi))*1e3,'mV')
    print('Max HQ is',max(abs(hq))*1e3,'mV')
    print('Max VI is',max(abs(vi))*1e3,'mV')
    print('Max VQ is',max(abs(vq))*1e3,'mV')
    print('RMS HI is',sqrt(np.mean(abs(hi)**2))*1e3,'mV')
    print('RMS HQ is',sqrt(np.mean(abs(hq)**2))*1e3,'mV')
    print('RMS VI is',sqrt(np.mean(abs(vi)**2))*1e3,'mV')
    print('RMS VQ is',sqrt(np.mean(abs(vq)**2))*1e3,'mV')
    #———————————————————————————#
    output = {'Time':time,'HI':hi,'HQ':hq,'VI':vi,'VQ':vq}
    return(output)
#####################################################################
#####################################################################
#####################################################################
#定义<相位噪声注入>函数
def add_phase_noise(Sin, Fs, phase_noise_freq, phase_noise_powr, plot=True, VALIDATION_ON=False):
    if(not np.any(np.imag(Sin))):
        print('Input signal should be complex signal')
        return()
    #———————————————————————————#
    if(np.max(phase_noise_freq) >=Fs/2):
        print('Maximal frequency offset should be less than Fs/2')
        return()
    #———————————————————————————#
    if(len(phase_noise_freq)!=len(phase_noise_powr)):
        print('phase_noise_freq and phase_noise_power should be of the same length')
        return()
    #———————————————————————————#Calculate input length
    input_length = len(Sin)
    if input_length % 2 == 1:
        semi_fft_length = (input_length + 1)/2
    else:
        semi_fft_length = input_length/2
    semi_fft_length = int(semi_fft_length)
    #———————————————————————————#
    #Generate a frequency comb [0, Fs/2] whose grid is freq_resolution
    freq_resolution = Fs/(2*semi_fft_length)
    print('Sample Frequency is',Fs,'GSa/s')
    print('Sample length is',1/freq_resolution,'ns')
    print('Frequency Resolution is',freq_resolution*1e6,'kHz')
    frequencyVector_P = np.arange(0, semi_fft_length+1)*freq_resolution
    #############################
    #############################
    #———————————————————————————#Sort phase_noise_freq and phase_noise_powr
    Z = zip(phase_noise_freq,phase_noise_powr)
    Z = sorted(Z,reverse=False)
    phase_noise_freq_sort,phase_noise_powr_sort = zip(*Z)
    phase_noise_freq = np.array(phase_noise_freq_sort)
    phase_noise_powr = np.array(phase_noise_powr_sort)
    #———————————————————————————#Add 0 dBc/Hz @DC
    if phase_noise_freq[0] != 0:
        phase_noise_freq = np.insert(phase_noise_freq, 0, 0)
        phase_noise_powr = np.insert(phase_noise_powr, 0, 0)
    #———————————————————————————#Interpolate the noise frequency spectrum
    if(0):
        #———————————————————————————#Determine the background noise
        background = phase_noise_powr[np.argmax(phase_noise_freq)]
        RespH_P = np.ones(semi_fft_length) *background
        #———————————————————————————#
        fl = interp1d(phase_noise_freq, phase_noise_powr, 'linear')
        flag = int(phase_noise_freq[-1]/freq_resolution)+1
        upsample = np.arange(0,flag)*freq_resolution
        RespH_P[:flag] = fl(upsample)
    if(1):
        realmin = 2.225073858507201e-308
        intrvlNum = len(phase_noise_freq)
        RespH_P = np.zeros(semi_fft_length+1)
        print('---------------------------',end='')
        for intrvlIndex in range(intrvlNum):
            leftBound = phase_noise_freq[intrvlIndex]   #定义频带左边界
            t1 = phase_noise_powr[intrvlIndex]          #定义噪声左边界值
            print('Band',intrvlIndex+1)
            print(t1,'dBc/Hz @',leftBound*1e6,'kHz')
            #———————————————————————#
            if  (intrvlIndex == intrvlNum -1):          #当频带在最高频时
                rightBound = Fs/2                       #频带右边界为Fs/2
                t2 = phase_noise_powr[-1]               #噪声右边界值
                inside = []
                for i in range(len(frequencyVector_P)):
                    if(frequencyVector_P[i] >=leftBound and frequencyVector_P[i] <=rightBound):
                        inside.append(i)
            else:
                rightBound = phase_noise_freq[intrvlIndex +1]
                t2 = phase_noise_powr[intrvlIndex +1]   #噪声右边界值
                inside = []
                for i in range(len(frequencyVector_P)):
                    if(frequencyVector_P[i] >=leftBound and frequencyVector_P[i] <rightBound):
                        inside.append(i)
            RespH_P[inside] = t1 +(np.log10(frequencyVector_P[inside] +realmin) -np.log10(leftBound +realmin))/(np.log10(rightBound +realmin) -np.log10(leftBound +realmin))*(t2 -t1)
            print(t2,'dBc/Hz @',rightBound*1e6,'kHz')
            print('---------------------------',end='')
        print()
    #############################
    #############################
    #———————————————————————————#Generate AWGN Noise on frequency spectrum
    if(not VALIDATION_ON):
        awgn_P1 = np.sqrt(0.5) *(np.random.randn(semi_fft_length+1) +1j *np.random.randn(semi_fft_length+1))
    else:
        awgn_P1 = np.sqrt(0.5) *(np.ones(semi_fft_length+1) +1j *np.ones(semi_fft_length+1))
    #———————————————————————————#Generate Phase Noise
    # Shape the noise on the positive spectrum [0, Fs/2] including bounds (semi_fft_length+1 points)
    X_P = 2 *semi_fft_length *awgn_P1 *np.sqrt(freq_resolution * 10**(RespH_P/10))
    # Get symmetrical negative spectrum [-Fs/2, 0) not including bounds (semi_fft_length points)
    X_N = np.conj(X_P[1:])[::-1]
    # Complete full spectrum [-Fs/2, Fs) not including bounds (2*semi_fft_length points)
    X   = np.append(X_P[:-1], X_N)
    X[0]= 0         #Remove DC
    x   = ifft(X)   #Perform IFFT
    # Calculate phase noise 
    phase_noise = np.exp(1j * np.real(x))
    phase_noise1=    1 + 1j * np.real(x)
    Sout = Sin * phase_noise[:input_length]
    #############################
    #############################
    if(plot):
        #———————————————————————#Calculate the spectrum of generated Phase Niose
        fpZ = ESA(phase_noise, Fs, SSB=1, Normalized=1, plot=0)
        #———————————————————————#Plot the Phase Noise Spectrum
        plt.semilogx(frequencyVector_P[:-1]*1e9,fpZ              ,label='Phase Noise Approximation')
        plt.plot    (frequencyVector_P*1e9     ,RespH_P,color='m',label='Phase Noise Mask')
        plt.grid(color='gray',linestyle=':')
        plt.scatter(phase_noise_freq*1e9,phase_noise_powr,color='red',s=25)
        plt.title ('SSB Phase Noise Spectrum')
        plt.xlabel('Frequency Offset (Hz)') 
        plt.ylabel('Phase Noise (dBc)')
        plt.xlim(phase_noise_freq[1]*1e9,Fs*1e9/2)
        plt.ylim(-143,3)
        #plt.legend('upper right')
        plt.show()
    return(Sout)
'''
def test(length,VALIDATION_ON):
    phase_noise_freq = np.array([1e3, 10e3, 100e3, 1e6, 10e6])  # Offset From Carrier
    phase_noise_powr = np.array([-84, -96 , -100, -109, -122])  # Phase Noise power
    Fc = 3e3    # carrier frequency
    Fs = 40e6   # sampling frequency
    t = np.arange(0, length)
    S = np.exp(1j * 2 * np.pi * Fc/Fs * t)  # complex sinusoid
    aaa = add_phase_noise(S, Fs, phase_noise_freq, phase_noise_powr, plot=True, VALIDATION_ON=VALIDATION_ON)
    return(aaa)
test(1e5,0)
'''
