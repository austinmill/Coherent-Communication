# -*- coding: UTF-8 -*-
#####################################################################
#Author: Austin Zhang
#Time: 2020/12/01
#Email: austinmill2010@gmail.com
#Website: https://github.com/austinmill/Coherent-Communication
#####################################################################
#####################################################################
#Required Module:
#numpy
#pandas
#Sympy
#matplotlib
#Scipy
#Modulation Format Supported: QPSK, 8QAM, 16QAM, SP-16QAM, 32-SP-QAM, 128-SP-QAM
#####################################################################
if 1:
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sympy as sy
    from scipy.interpolate import interp1d
    from scipy import signal
    from numpy import sqrt
    from numpy import exp
    from DSP import *
    from Test import *
    from Optics import *
#####################################################################
#####################################################################
#####################################################################定义参数
#if 1:
    #———————————————————————————#定义波特率,单位为Gbaud。为了避免ADC采样函数时间的浮点数精度问题，不使用baud作为单位
    Baudrate = 32
    #———————————————————————————#定义仿真系统模拟量采样倍率
    Sim_SampleRatio = round(1024/Baudrate)
    #———————————————————————————#
    Txdsp_SampleRatio = 2.0     #定义Tx_DSP采样倍率
    Dac_SampleRatio = 2.0       #定义DAC采样倍率
    Dac_Bit = 8                 #定义DAC采样分辨率
    #———————————————————————————#
    Adc_clock = Baudrate        #定义ADC采样时钟,单位为GHz。为避免ADC采样函数时间的浮点数精度问题，不使用Hz作为单位
    Adc_SampleRatio = 2         #定义ADC每时钟周期采样倍率
    Adc_Bit = 8                 #定义ADC采样分辨率
    #———————————————————————————#
    c = 2.9979e-1               #定义光速，单位为m/ns
    f_cent = 193.1*1e3          #定义载波光频率,单位为GHz
    wavelength = c /f_cent      #定义载波光波长,单位为m
    #———————————————————————————#定义调制格式，可选QPSK, 8QAM, 16QAM, SP-16QAM, 32-SP-QAM, 128-SP-QAM
    mod='SP-16QAM'
    #———————————————————————————#定义数据长度
    length = 2**16+2**10
    #———————————————————————————#定义WSS光滤波器频率响应
    wss_freq = np.arange(15,32) #np.array([15,16,17,18,19,20,21,22,25,24,23,26,27,28,29,30,31])
    wss_resp = np.array([wss_freq,[0,0,0,0,-0.5,-2,-5,-10,-15,-20,-25,-30,-33,-34.5,-35,-35,-35]])
#####################################################################读输入bit流
    data1 = np.random.randint(0,2,length)   #mydata['A']
    data2 = np.random.randint(0,2,length)   #mydata['B']
    data3 = np.random.randint(0,2,length)   #mydata['C']
    data4 = np.random.randint(0,2,length)   #mydata['D']
    data5 = np.random.randint(0,2,length)
    data6 = np.random.randint(0,2,length)
    data7 = np.random.randint(0,2,length)
    data8 = np.random.randint(0,2,length)
#####################################################################QAM编码
#if 1:
    plt.figure(figsize=(11,5.3)).set_tight_layout(True)
    if  (mod=='SP-16QAM' or mod=='32-SP-QAM' or mod=='128-SP-QAM'):
        d = encod_spqam(data1,data2,data3,data4,data5,data6,data7,mod=mod,plot=True)
        plt.subplots_adjust(wspace=0,hspace=0.3)
        plt.show()
        H = d[0]
        V = d[1]
        del(d)
    elif(mod=='BPSK' or mod=='QPSK' or mod=='8QAM' or mod=='16QAM'):
        plt.subplot(1, 2, 1)
        H = encod_qam(data1,data3,data5,data7,mod=mod,plot=True)
        plt.title('$H\ Pol$')
        plt.subplot(1, 2, 2)
        V = encod_qam(data2,data4,data6,data8,mod=mod,plot=True)
        plt.title('$V\ Pol$')
        plt.subplots_adjust(wspace=0,hspace=0.3)
        plt.show()
    elif(mod=='PS-16QAM'):
        H = encod_PS_qam(data1,data3,data5,data7,mod=mod)
        Constellate_3D(H[0],H[1],lab='H')
        V = encod_PS_qam(data2,data4,data6,data8,mod=mod)
        Constellate_3D(V[0],V[1],lab='V')
#####################################################################对编码数据2倍采样
    time = np.arange(0,length*Txdsp_SampleRatio,1)/Baudrate/Txdsp_SampleRatio
    hi = H[0].repeat(Txdsp_SampleRatio)
    hq = H[1].repeat(Txdsp_SampleRatio)
    vi = V[0].repeat(Txdsp_SampleRatio)
    vq = V[1].repeat(Txdsp_SampleRatio)
    dsp_output = {'Time':time,'HI':hi,'HQ':hq,'VI':vi,'VQ':vq}
    del(time,hi,hq,vi,vq,H,V)
#####################################################################DAC输出
    dac_output = DAC(dsp_output, Dac_Bit, Txdsp_SampleRatio, Dac_SampleRatio, Sim_SampleRatio, SNR=20)
#####################################################################观察DAC输出眼图
    Time_metric = 1e3
    plt.figure(figsize=(6,8)).set_tight_layout(True)
    plt.subplot(2, 1, 1)
    Eye_scope(Time_metric*dac_output['Time'],dac_output['HI'],Sim_SampleRatio,time_scale=4,time_shift=-9,lab='HI',color=False)
    plt.subplot(2, 1, 2)
    Eye_scope(Time_metric*dac_output['Time'],dac_output['HQ'],Sim_SampleRatio,time_scale=4,time_shift=-9,lab='HQ',color=False)
    plt.subplots_adjust(wspace=0,hspace=0.3)
    plt.show()
#####################################################################调制器输出
#if 1:
    v_pi = 1                    #定义调制器v_pi电压
    mod_swing = 1.0*v_pi        #定义调制信号摆幅（2*v_pi为满幅调制）
    P_in = 1
    E_in = sqrt(P_in)
    #———————————————————————————#
    XBI = 1.0*v_pi #+ np.sin(2*pi*dac_output['Time']*0.020)*0.05*v_pi
    XBQ = 1.0*v_pi #+ np.sin(2*pi*dac_output['Time']*0.013)*0.05*v_pi
    XBP = 0.5*v_pi
    #———————————————————————————#第1个sqrt(2)代表调制器偏振分束，第2个sqrt(2)代表调制器偏振合束
    IQ_MZ_h = E_in/sqrt(2)/sqrt(2)*IQ_MZ(mod_swing*dac_output['HI'],mod_swing*dac_output['HQ'],XBI ,XBQ ,XBP     ,v_pi,0,0,0)['IQP']
    IQ_MZ_v = E_in/sqrt(2)/sqrt(2)*IQ_MZ(mod_swing*dac_output['VI'],mod_swing*dac_output['VQ'],v_pi,v_pi,0.5*v_pi,v_pi,0,0,0)['IQP']
    #———————————————————————————#
    mod_output = {'Time':dac_output['Time'],'H':IQ_MZ_h,'V':IQ_MZ_v}
    del(IQ_MZ_h,IQ_MZ_v)
#####################################################################观察调制器输出光谱、眼图、星座图
    pwr = OPM(mod_output['H'],mod_output['V'],ref=P_in)
    #———————————————————————————#
    if 1:
        eye_gain = 10
        Time_metric = 1e3
        plt.figure(figsize=(6,4)).set_tight_layout(True)
        Eye_scope(Time_metric*mod_output['Time'],eye_gain*pwr,Sim_SampleRatio,time_scale=2,time_shift=-9,lab='Optical',color=False)
        plt.show()
    #———————————————————————————#
        OSA(mod_output['H'],SampleRate=Baudrate*Sim_SampleRatio,VBW=17/32,xlim=(-150,150),ylim=(-100,-10),output=False)
        #OSA(mod_output['H'],SampleRate=Baudrate*Sim_SampleRatio,VBW=7/32,xlim=(0,0.05),ylim=(-40,20),output=False)
    #———————————————————————————#
        constellate_gain = 2
        plt.figure(figsize=(11,5.3)).set_tight_layout(True)
        plt.subplot(1, 2, 1)
        Constellate_scope(constellate_gain*mod_output['H'].real,constellate_gain*mod_output['H'].imag,lab='H')
        plt.subplot(1, 2, 2)
        Constellate_scope(constellate_gain*mod_output['V'].real,constellate_gain*mod_output['V'].imag,lab='V')
        plt.show()
#####################################################################光纤传输
#经过任意偏振器件
#if 1:
    fiber_length = 101*1e3                  #Length of each fiber span, unit is meter
    num_span = 5                            #Number of spans
    M = np.array([[1,0],[0,1]])
    fiber_output = mod_output
    for i in range(num_span):
        #———————————————————————#随机偏振态
        phase = np.random.uniform(0,2*pi)   #随机偏振旋转
        M = rotate(phase).dot(M)
        phase = np.random.uniform(0,2*pi)   #随机相位延迟
        M = phase_delay(phase).dot(M)
        H = M[0,0] *fiber_output['H'] + M[0,1] *fiber_output['V']
        V = M[1,0] *fiber_output['H'] + M[1,1] *fiber_output['V']
        fiber_output['H'] = H
        fiber_output['V'] = V
        print('Span',i+1)
        #———————————————————————#DGD
        fiber_output = simp_DGD(fiber_output['Time'],fiber_output['H'],fiber_output['V'],0.02,Sim_SampleRatio,Baudrate,stochastic=True)
        #———————————————————————#色散
        fiber_output = CD(fiber_output['Time'],fiber_output['H'],fiber_output['V'],wavelength,fiber_length,SampleRate=Baudrate*Sim_SampleRatio)
        #———————————————————————#SPM
        # 
        #———————————————————————#Xpol-PM
        # 
        #———————————————————————#随机偏振旋转
        phase = np.random.uniform(0,2*pi)
        M = rotate(phase).dot(M)
        H = M[0,0] *fiber_output['H'] + M[0,1] *fiber_output['V']
        V = M[1,0] *fiber_output['H'] + M[1,1] *fiber_output['V']
        fiber_output['H'] = H
        fiber_output['V'] = V
    print('Total fiber length is',fiber_length*num_span/1e3,'km')
    #———————————————————————————#注入ASE
#if 1:
    OSNR_s = range(17,40,2)
    pre_cm = np.array([])
    post_cm= np.array([])
    cd_e   = np.array([])
#for OSNR in OSNR_s:
    OSNR = 20
    edfa_output = AWGN(fiber_output['Time'],fiber_output['H'],fiber_output['V'],OSNR,SampleRate=Baudrate*Sim_SampleRatio)
    #———————————————————————————#色散补偿光纤
    #edfa_output = CD(edfa_output['Time'],edfa_output['H'],edfa_output['V'],wavelength,510e3,SampleRate=Baudrate*Sim_SampleRatio)
    #———————————————————————————#O-DeMux/WSS滤波
    wss_output = WSS(edfa_output['Time'],edfa_output['H'],edfa_output['V'],wss_resp,SampleRate=Baudrate*Sim_SampleRatio,plot=0)
    #———————————————————————————#观测光谱
    if 1:
        OSA(edfa_output['H'],edfa_output['V'],SampleRate=Baudrate*Sim_SampleRatio,VBW=17/32,xlim=(-150,150),ylim=(-100,-10),output=False)
        OSA(wss_output['H'],wss_output['V'],SampleRate=Baudrate*Sim_SampleRatio,VBW=17/32,xlim=(-150,150),ylim=(-100,-10),output=False,OSNR=False)
    #———————————————————————————#观测星座图
    if 1:
        constellate_gain = 2
        plt.figure(figsize=(11,5.3)).set_tight_layout(True)
        plt.subplot(1, 2, 1)
        Constellate_scope(constellate_gain*wss_output['H'].real,constellate_gain*wss_output['H'].imag,lab='H')
        plt.subplot(1, 2, 2)
        Constellate_scope(constellate_gain*wss_output['V'].real,constellate_gain*wss_output['V'].imag,lab='V')
        plt.show()
#####################################################################ICR接收光信号
#if 1:
    lo_offset = 0.11
    lo_laser  = exp(-1j*2*pi*lo_offset*wss_output['Time'])
    #———————————————————————————#LO激光相位噪声
    if 0:
        phase_noise_freq = np.array([1e5, 1e6, 1e7, 1e8, 1e9 ])*1e-9    # Offset From Carrier, unit is GHz
        phase_noise_powr = np.array([-35, -40, -45, -60, -80])          # Phase Noise power
        lo_laser = add_phase_noise(lo_laser, Baudrate*Sim_SampleRatio, phase_noise_freq, phase_noise_powr, plot=1, VALIDATION_ON=0)
    #———————————————————————————#ICR接收光信号
    icr_output = ICR(wss_output['Time'],wss_output['H'],wss_output['V'],lo_laser,phase_iq=pi/2,AGC=1,gain=[1,1,1,1],oc=0.12)
#####################################################################重定时
    adc_delay = ADC_clock_phase_estimate(icr_output,Adc_clock,Adc_SampleRatio,Adc_Bit)
#####################################################################ADC采样
    adc_output=ADC(icr_output['Time'],icr_output['HI'],icr_output['HQ'],icr_output['VI'],icr_output['VQ'],Adc_clock,Adc_SampleRatio,time_shift=adc_delay)
    if 0:
        b = 2400
        e = 2500
        sr = int(Sim_SampleRatio/Adc_SampleRatio)
        plt.plot   (icr_output['Time'][(sr*b):(sr*e)],icr_output['HI'][(sr*b):(sr*e)]*(2**Adc_Bit-1) )
        plt.scatter(adc_output['Time'][b     :e],     adc_output['HI'][b     :e],color='red')
        plt.show()
#####################################################################色散估计
#if 1:
# 该方法来自于Edem Ibragimov的论文"Blind Chromatic Dispersion Estimation Using a Spectrum of a
# Modulus Squared of the Transmitted Signal", ECOC 2012, Th.2.A.3
    search_range = np.arange(-1500,1500,60)*1e3   #色散粗搜索范围及搜素步长，单位m
    residue_cd = np.array([])
    for i in search_range:
        cdc_output = CDC(adc_output,wavelength,i,Adc_clock*Adc_SampleRatio,fft_length=2**10,trial=1)
        residue_cd = np.append(residue_cd, CDE(cdc_output,Adc_clock*Adc_SampleRatio,False))
    cd_estimated = search_range[np.argmin(residue_cd)]
    if 1:
        plt.plot(search_range/1e3,residue_cd)
        plt.xlabel('Compensated CD (km)')
        plt.ylabel('Residue CD (a.u.)')
        plt.title('CD Estimation')
        plt.grid(color='gray',linestyle=':')
        plt.plot(cd_estimated/1e3,min(residue_cd),'md',label='CDE = '+str(cd_estimated/1e3)+'km')
        plt.legend(loc='upper right')
        plt.show()
    search_range = np.arange(cd_estimated-60e3,cd_estimated+60e3,10*1e3)   #色散细搜索范围及搜素步长，单位m
    residue_cd = np.array([])
    for i in search_range:
        cdc_output = CDC(adc_output,wavelength,i,Adc_clock*Adc_SampleRatio,fft_length=2**10,trial=1)
        residue_cd = np.append(residue_cd, CDE(cdc_output,Adc_clock*Adc_SampleRatio,False))
    cd_estimated = search_range[np.argmin(residue_cd)]
    print('Estimated Fiber Length =',cd_estimated/1e3,'km')
    if 0:
        export_data(cdc_output).to_csv('cdc_output.csv',index=False,sep=',')
#####################################################################载波频偏补偿+色散补偿+偏振补偿
#if 1:
    freq_offset_x = 0           #前置载波频偏补偿预设值为0
    freq_offset_y = 0
    for j in range(2):
        #———————————————————————#前置载波频偏补偿
        for_output = freq_recover(adc_output, -freq_offset_x, -freq_offset_y)
        #———————————————————————#色散补偿
        cdc_output = CDC(for_output,wavelength,cd_estimated,Adc_clock*Adc_SampleRatio,fft_length=2**10)
        #———————————————————————#CMA预收敛
        if(j==0):
            h = mimo_init(15)
            step=0.005
            ffe_output = CMA(cdc_output,step=step,para=h,plot=0)
            h = ffe_output['para']
        #———————————————————————#迭代i次，得到最优的恢复矩阵，输出最后一次计算的结果
        step=0.002
        for i in range(2):
            ffe_output = RDA(cdc_output,mod=mod,step=step,para=h,plot=(i==1))
            h = ffe_output['para']
            step = step*0.5
        #———————————————————————#
        if(j==1):
            constellate_gain = 1/64
            L = len(ffe_output['Time'])
            start = 15
            end = L-15
            plt.figure(figsize=(11,5.3)).set_tight_layout(True)
            plt.subplot(1, 2, 1)
            Constellate_scope(constellate_gain*ffe_output['HI'],constellate_gain*ffe_output['HQ'],lab='H')
            circle = plt.Circle((0.0, 0.0),radius=constellate_gain*ffe_output['target'][0],color='r',fill=False)
            circle1= plt.Circle((0.0, 0.0),radius=constellate_gain*ffe_output['target'][1],color='y',fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.gcf().gca().add_artist(circle1)
            plt.subplot(1, 2, 2)
            Constellate_scope(constellate_gain*ffe_output['VI'],constellate_gain*ffe_output['VQ'],lab='V')
            circle = plt.Circle((0.0, 0.0),radius=constellate_gain*ffe_output['target'][0],color='r',fill=False)
            circle1= plt.Circle((0.0, 0.0),radius=constellate_gain*ffe_output['target'][1],color='y',fill=False)
            plt.gcf().gca().add_artist(circle)
            plt.gcf().gca().add_artist(circle1)
            plt.show()
        #———————————————————————#后置载波频偏估计，反馈至前级载波频偏补偿
        fft_length = 1024
        plt.figure(figsize=(11,5.3)).set_tight_layout(True)
        plt.subplot(1, 2, 1)
        freq_offset_x = freq_estimate(ffe_output['HI'][2000:],ffe_output['HQ'][2000:],fft_length,Adc_clock*Adc_SampleRatio)
        plt.subplot(1, 2, 2)
        freq_offset_y = freq_estimate(ffe_output['VI'][2000:],ffe_output['VQ'][2000:],fft_length,Adc_clock*Adc_SampleRatio)
        plt.show()
    #———————————————————————————#
    if 0:
        export_data(ffe_output).to_csv('ffe_output.csv',index=False,sep=',')
#####################################################################相位恢复
#if 1:
    cpr_output = CPR_BPS(ffe_output,mod=mod,N_step=24,window=24,move=8,IIR=29/32,plot=1,unlock=0,debug=0)
    #cpr_output = CPR_VV(ffe_output,half_window=16,IIR=21/32,plot=1)
    #———————————————————————————#
    if 1:
        s = 200
        e = len(cpr_output['HI'])-200
        EVM_x = show_evm(cpr_output['HI'][s:e],cpr_output['HQ'][s:e],mod=mod)
        EVM_y = show_evm(cpr_output['VI'][s:e],cpr_output['VQ'][s:e],mod=mod)
        #———————————————————————#
        constellate_gain = 1/64
        plt.figure(figsize=(11,5.3)).set_tight_layout(True)
        plt.subplot(1, 2, 1)
        Constellate_scope(constellate_gain*cpr_output['HI'][s:e],constellate_gain*cpr_output['HQ'][s:e],lab='H',mod=mod)
        plt.text(0.5, 0.9, 'EVM='+str(round(EVM_x['EVM']*100,2))+'%', color='white')
        plt.text(0.5, 0.8, 'SNR='+str(round(EVM_x['SNR'],2))+'dB', color='white')
        plt.subplot(1, 2, 2)
        Constellate_scope(constellate_gain*cpr_output['VI'][s:e],constellate_gain*cpr_output['VQ'][s:e],lab='V',mod=mod)
        plt.text(0.5, 0.9, 'EVM='+str(round(EVM_y['EVM']*100,2))+'%', color='white')
        plt.text(0.5, 0.8, 'SNR='+str(round(EVM_y['SNR'],2))+'dB', color='white')
        plt.show()
#####################################################################采用DD_LMS进一步优化
#if 1:
    h = mimo_init(15)
    step=0.0005
    for i in range(10):
        dd_lms = DD_LMS(cpr_output,mod=mod,step=step,para=h,plot=(i==9))
        h = dd_lms['para']
        step = step*1
    #———————————————————————————#
    if 1:
        s = 200
        e = len(dd_lms['HI'])-200
        EVM_x = show_evm(dd_lms['HI'][s:e],dd_lms['HQ'][s:e],mod=mod)
        EVM_y = show_evm(dd_lms['VI'][s:e],dd_lms['VQ'][s:e],mod=mod)
        #———————————————————————#
        constellate_gain = 1/64
        plt.figure(figsize=(11,5.3)).set_tight_layout(True)
        plt.subplot(1, 2, 1)
        Constellate_scope(constellate_gain*dd_lms['HI'][s:e],constellate_gain*dd_lms['HQ'][s:e],lab='H',mod=mod)
        plt.text(0.5, 0.9, 'EVM='+str(round(EVM_x['EVM']*100,2))+'%', color='white')
        plt.text(0.5, 0.8, 'SNR='+str(round(EVM_x['SNR'],2))+'dB', color='white')
        plt.subplot(1, 2, 2)
        Constellate_scope(constellate_gain*dd_lms['VI'][s:e],constellate_gain*dd_lms['VQ'][s:e],lab='V',mod=mod)
        plt.text(0.5, 0.9, 'EVM='+str(round(EVM_y['EVM']*100,2))+'%', color='white')
        plt.text(0.5, 0.8, 'SNR='+str(round(EVM_y['SNR'],2))+'dB', color='white')
        plt.show()
    Constellate_3D(constellate_gain*dd_lms['HI'][s:e],constellate_gain*dd_lms['HQ'][s:e],lab='H')
#####################################################################
