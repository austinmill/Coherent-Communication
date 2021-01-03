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
    from mpl_toolkits.mplot3d import Axes3D
    #from scipy import signal
    from scipy.interpolate import interp1d
    from scipy.stats import kde
    from scipy.constants import pi
    from scipy.fftpack import fft,ifft
    from numpy import sqrt
    from numpy import exp
    from DSP import *
#####################################################################
#####################################################################
#####################################################################
#定义<导出数据>函数
def export_data(data):
    data = {k.lower():v for k,v in data.items()}
    key = data.keys()
    #———————————————————————————#
    if  ('time' in key):
        time = data['time']
    else:
        time = np.array([])
    #———————————————————————————#
    if   all([i in key for i in ['x','y']]):
        hi   = data['x'].real
        hq   = data['x'].imag
        vi   = data['y'].real
        vq   = data['y'].imag
    elif all([i in key for i in ['h','v']]):
        hi   = data['h'].real
        hq   = data['h'].imag
        vi   = data['v'].real
        vq   = data['v'].imag
    elif all([i in key for i in ['hi','hq','vi','vq']]):
        hi   = data['hi']
        hq   = data['hq']
        vi   = data['vi']
        vq   = data['vq']
    elif all([i in key for i in ['xi','xq','yi','yq']]):
        hi   = data['xi']
        hq   = data['xq']
        vi   = data['yi']
        vq   = data['yq']
    #———————————————————————————#输入初始化
    data = {'Time':time,'HI':hi,'HQ':hq,'VI':vi,'VQ':vq}
    df = pd.DataFrame(data)
    return(df)
#export_data(ffe_output)
#####################################################################
#####################################################################
#####################################################################
#定义<导入数据>函数
def inport_data(file_name):
    df = pd.read_csv(file_name,sep=',')
    time = df.loc[:,'Time']
    hi   = df.loc[:,'HI']
    hq   = df.loc[:,'HQ']
    vi   = df.loc[:,'VI']
    vq   = df.loc[:,'VQ']
    #———————————————————————————#输入初始化
    data = {'Time':time,'HI':hi,'HQ':hq,'VI':vi,'VQ':vq}
    return(data)
#inport_data(ffe_output)
#####################################################################
#####################################################################
#####################################################################
def ESA(input_data, Fs, f_range='', p_range='', SSB=False, Normalized=False, plot=False):
    length = len(input_data)
    freq_resolution = Fs/length
    fpY = fft(input_data)
    if(SSB):
        fpY = fpY[:int(length/2)]
        frequencyVector = np.arange(0,length/2)*freq_resolution
    else:
        fpY = np.append(fpY[int(length/2):],fpY[:int(length/2)])
        frequencyVector = np.arange(-length/2,length/2)*freq_resolution
    fpY = abs(fpY)**2
    if(Normalized):
        fpY = fpY/np.max(fpY)
    fpY = 10*np.log10(fpY/freq_resolution)
    #———————————————————————#
    if(plot):
        plt.plot(frequencyVector,fpY,color='gray',label='Phase Noise')
        plt.grid(color='gray',linestyle=':')
        plt.title ('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)') 
        plt.ylabel('Magnititude (dB/Hz)')
        if(f_range!=''):
            plt.xlim(min(f_range),max(f_range))
        if(p_range!=''):
            plt.ylim(min(p_range),max(p_range))
        #plt.legend()
        plt.show()
    return(fpY)
#####################################################################
#####################################################################
#####################################################################
def ESA1(input_data0, input_data1, Fs, f_range='', p_range='', SSB=False, Normalized=False, plot=False):
    length = len(input_data0)
    freq_resolution = Fs/length
    fpY = fft(input_data0)
    fpZ = fft(input_data1)
    if(SSB):
        fpY = fpY[:int(length/2)]
        fpZ = fpZ[:int(length/2)]
        frequencyVector = np.arange(0,length/2)*freq_resolution
    else:
        fpY = np.append(fpY[int(length/2):],fpY[:int(length/2)])
        fpZ = np.append(fpZ[int(length/2):],fpZ[:int(length/2)])
        frequencyVector = np.arange(-length/2,length/2)*freq_resolution
    fpY = abs(fpY)**2
    fpZ = abs(fpZ)**2
    if(Normalized):
        fpY = fpY/np.max(fpY)
        fpZ = fpZ/np.max(fpZ)
    fpY = 10*np.log10(fpY/freq_resolution)
    fpZ = 10*np.log10(fpZ/freq_resolution)
    #———————————————————————#
    if(plot):
        plt.plot(frequencyVector,fpY,color='gray',label='Signal 0')
        plt.plot(frequencyVector,fpZ,color='m'   ,label='Signal 1')
        plt.grid(color='gray',linestyle=':')
        plt.title ('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)') 
        plt.ylabel('Magnititude (dB/Hz)')
        if(f_range!=''):
            plt.xlim(min(f_range),max(f_range))
        if(p_range!=''):
            plt.ylim(min(p_range),max(p_range))
        plt.legend(loc='upper right')
        plt.show()
    return(fpY)
#####################################################################
#####################################################################
#####################################################################
#定义<显示眼图>函数
def Eye_scope(input_time, input_data, Sim_SampleRatio, time_scale=2, time_shift=0, lab='', Ratio=10, color=True):
    input_length = len(input_time)
    if(input_length != len(input_data)):
        print('Input time and data are not synchronized!!\n')
        return
    #———————————————————————————#
    time_scale = abs(int(time_scale))                               #
    time_shift = int(time_shift)%(Sim_SampleRatio*time_scale)       #
    #———————————————————————————#时间轴折叠
    time_sl = np.arange(0.0,input_length)
    for i in range(0,input_length):
        time_sl[i] = input_time[i%(Sim_SampleRatio*time_scale)]     #
    #———————————————————————————#时间轴平移
    time_sh = np.arange(0.0,input_length)
    for j in range(0,input_length):
        time_sh[j] = time_sl[(j+time_shift)%(Sim_SampleRatio*time_scale)]   #
    #return(time_sh)
    #———————————————————————————#
    start   = round(input_length*0.2)
    end     = min(round(input_length*0.8),start+20000)
    output_time = np.arange(0.0,(end-start)*Ratio+1)    #插值后的时间序列
    output_data = np.arange(0.0,(end-start)*Ratio+1)    #插值后的数据序列
    print('Interpolating',end='')
    for k in range(start,end):
        if(time_sh[k] < time_sh[k+1]):	#
            x = time_sh[k:k+2]      #k到k+1
            y = input_data[k:k+2]   #k到k+1
            upsample = np.random.uniform(time_sh[k]*1.000001,time_sh[k+1],Ratio-1)
            upsample = np.concatenate(([time_sh[k]],np.sort(upsample),[time_sh[k+1]]))
            fl = interp1d(time_sh[k:k+2], input_data[k:k+2], 'linear')
            output_time[(k-start)*Ratio:(k-start+1)*Ratio+1] = upsample
            output_data[(k-start)*Ratio:(k-start+1)*Ratio+1] = fl(upsample)
        else:				#
            output_time[(k-start)*Ratio:(k-start+1)*Ratio+1] = np.nan
            output_data[(k-start)*Ratio:(k-start+1)*Ratio+1] = np.nan    
        if(k%500==0):
            print('.',end='')
    print('\n')
    #———————————————————————————#
    output_time = output_time[~pd.isnull(output_time)]
    output_data = output_data[~pd.isnull(output_data)]
    time_start  = round(np.nanmin(output_time))
    time_end    = round(np.nanmax(output_time))
    ybins_start = np.floor(np.nanmin(output_data))
    ybins_end   = np.ceil (np.nanmax(output_data))
    #———————————————————————————#
    if(color):
        plt.hist2d(output_time,output_data,bins=(300,300),range=[[0,time_end],[ybins_start,ybins_end]],density=True,cmap=plt.cm.jet)
    else:
        plt.scatter(output_time,output_data,color='orange',s=1)
        plt.xlim(0,time_end)
        plt.ylim(ybins_start,ybins_end)
        #plt.legend(loc='upper center')
    plt.ylabel('Magnitud (V)')
    plt.xlabel('Time (ps)')
    plt.grid(color='gray',linestyle=':')
    plt.title('$'+lab+'\ Eye\ Diagram$')
    #plt.show()
#Eye_scope(1e3*dac_output['Time'],dac_output['HI'],Sim_SampleRatio,time_scale=2,time_shift=-9,lab='HI',color=False)
'''
if(1):
    plt.plot(dsp_output['Time'][0:62],dsp_output['HI'][0:62],color='red',linestyle='--')
    plt.scatter(dac_output['Time'][0:1000],dac_output['HI'][0:1000],color='blue',s=1)    
    plt.ylabel('Magnitud (V)') 
    plt.xlabel('Time (ps)')
    plt.grid(color='gray',linestyle=':')
    plt.show()
'''
#####################################################################
#####################################################################
#####################################################################
#定义<光频谱分析>函数
def OSA(input_h,input_v=[],SampleRate=1,VBW=7/32,xlim=(-150,150),ylim=(-100,-20),output=True,OSNR=True):
    input_length = len(input_h)
    #fft_length = 2**16
    fft_length = 2** int(np.log2(input_length))
    freq_resolution = SampleRate/fft_length
    #———————————————————————————#处理H偏振数据
    if  (input_length >= fft_length):
        input_h = input_h[(input_length-fft_length):input_length]
    else:
        input_h = np.append(input_h,np.zeros(fft_length -input_length))
    #———————————————————————————#
    fpX = abs( fft(input_h)/fft_length )**2
    #———————————————————————————#处理V偏振数据
    if(len(input_v)>1):
        if  (len(input_v) != input_length):
            print('H and V are not synchronized!!')
            print('Only H polarization is analyzed!!')
        elif(input_length >= fft_length):
            input_v = input_v[(input_length-fft_length):input_length]
            fpX = fpX + abs( fft(input_v)/fft_length )**2
        else:
            input_v = np.append(input_v,np.zeros(fft_length -input_length))
            fpX = fpX + abs( fft(input_v)/fft_length )**2
    #———————————————————————————#整理FFT结果
    fpX = np.append(fpX[int(fft_length/2):],fpX[:int(fft_length/2)])
    #———————————————————————————#
    frequency = np.arange(-int(fft_length/2),int(fft_length/2))*freq_resolution
    #———————————————————————————#功率谱密度，单位dBm/GHz
    p_density = 10 *np.log10( fpX/freq_resolution )
    #———————————————————————————#绘制图像
    p_density = IIR(p_density,VBW,2)
    plt.plot(frequency,p_density)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Power (dBm/GHz)')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(color='gray',linestyle=':')
    plt.title('$Optical\ Spectrum$')
    #———————————————————————————#拟合曲线
#   fit = lowess(frequency,p_density,f=0.0001)
#   lines(fit,col='red',lwd=2,lty=1)
    #———————————————————————————#打印频率分辨率
    print('Freq Resolution is',freq_resolution*1e3,'MHz')
    #———————————————————————————#打印功率积分
    print('Optical power is',sum(fpX),'mW')
    #———————————————————————————#输出结果对频率积分后等于光功率
    if(OSNR):                   #带外法测OSNR
        #———————————————————————#计算-25GHz ~ 25GHz功率积分
        m1 = int(-25/freq_resolution +fft_length/2)
        m2 = int( 25/freq_resolution +fft_length/2)
        s0 = np.sum(fpX[m1:m2])
        plt.plot(frequency[m1:m2],p_density[m1:m2],color='orange')
        #———————————————————————#计算-125GHz ~ -75GHz功率积分
        m1 = int(-125/freq_resolution +fft_length/2)
        m2 = int(-75 /freq_resolution +fft_length/2)
        n1 = np.sum(fpX[m1:m2])
        plt.plot(frequency[m1:m2],p_density[m1:m2],color='orange')
        #———————————————————————#计算+125GHz ~ +75GHz功率积分
        m1 = int(75 /freq_resolution +fft_length/2)
        m2 = int(125/freq_resolution +fft_length/2)
        n2 = np.sum(fpX[m1:m2])
        plt.plot(frequency[m1:m2],p_density[m1:m2],color='orange')
        #———————————————————————#计算OSNR
        n0 = (n1+n2)/8
        OSNR = round(10*np.log10((s0-n0*4)/n0),2)
        #———————————————————————#
        plt.text(20,-20,'OSNR: '+str(OSNR)+'dB')
        print('OSNR is',OSNR,'dB')
    plt.show()
    #———————————————————————————#输出结果对频率积分后等于光功率
    if(output):
        return([frequency,fpX])
#####################################################################
#####################################################################
#####################################################################
#定义<光功率计>函数
def OPM(input_h,input_v=[],ref=0):
    input_length = len(input_h)
    pwr = abs(input_h)**2
    #———————————————————————————#V偏振数据处理
    if(len(input_v)>1):
        if  (len(input_v)!=input_length):
            print('H and V are not synchronized!!')
            print('Only H polarization is measured!!')
        else:
            pwr = pwr +abs(input_v)**2
    pwr_rms = np.mean(pwr)	#平均光功率，单位mW
    #———————————————————————————#取对数变为dB
    print('Optical power is',pwr_rms,'mW')
    print('Optical power is',round(10*np.log10(pwr_rms),3),'dBm')
    #———————————————————————————#取对数变为dB
    if(ref!=0):
        loss = 10 *np.log10(pwr_rms/ref)
        print('Insertion loss is',round(loss,3),'dB')
    return(pwr)		        #光强，用于眼图输出
#####################################################################
#####################################################################
#####################################################################
#定义<计算EVM>函数
def show_evm(I, Q, mod='QPSK'):
    I = np.array(I)
    Q = np.array(Q)
    input_length = len(I)
    #———————————————————————————#Normalize the input
    sqrt_cm = sqrt(np.mean(I**2 +Q**2))
    group = (I +1j*Q)/sqrt_cm
    #———————————————————————————#Generate normalized constellation
    constellation = std_constellation(mod,plot=0)
    #———————————————————————————#Calculate the sum of square euclidean error
    sq_error_sum = 0
    for symbol in group:
        error_vector = abs(constellation -symbol)**2
        sq_error_sum = sq_error_sum + min(error_vector)
    #———————————————————————————#    
    evm = sqrt(sq_error_sum/input_length)#/sqrt(2)
    #———————————————————————————#  
    signal_power = 1
    noise_power  = sq_error_sum/input_length
    SNR          = 10*np.log10(signal_power/noise_power)
    if(0):
        plt.scatter(group.real        ,group.imag        ,marker='.',s=1)
        plt.scatter(constellation.real,constellation.imag,c='red'   ,s=10)
        #plt.xlim(-1,1)
        #plt.ylim(-1,1)
        plt.ylabel('Q') 
        plt.xlabel('I')
        plt.grid(color='gray',linestyle=':')
        plt.title('$Constellate$')
        plt.show()
    print('EVM = ',round(evm*100,2),'%')
    return({'EVM':evm,'SNR':SNR})
#####################################################################
#####################################################################
#####################################################################
#定义<星座图输出>函数
def Constellate_scope(input_i, input_q, mod='', lab='', color=True):
    input_length = len(input_i)
    if(input_length != len(input_q)):
        print('Input I and Q are not synchronized!!')
        return()
    start = round(input_length*0.2)
    end   = round(input_length*0.8) #min(round(input_length*0.8),start+15000)
    #———————————————————————————#
    sqrt_cm = sqrt(np.mean(input_i[start:end]**2 +input_q[start:end]**2))
    if(mod!=''):    
        constellation = std_constellation(mod,plot=0)*sqrt_cm
    #———————————————————————————#
    if  (color):
        plt.hist2d(input_i[start:end],input_q[start:end],bins=(300,300),range=[[-1,1],[-1,1]],density=True,cmap=plt.cm.jet)
    else:
        plt.scatter(input_i[start:end],input_q[start:end],s=1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
    if  (mod!=''):
        plt.scatter(constellation.real,constellation.imag,marker='o',color='',edgecolors='red',s=1500)
    plt.ylabel('Q')
    plt.xlabel('I')
    plt.grid(color='gray',linestyle=':')
    plt.title('$'+lab+'\ Constellate$')
    #plt.show()
#Constellate_scope(dac_output['HI'], dac_output['HQ'], lab='')
#####################################################################
#####################################################################
#####################################################################
def Constellate_3D(input_i, input_q, mod='', lab=''):
    input_length = len(input_i)
    if(input_length != len(input_q)):
        print('Input I and Q are not synchronized!!')
        return()
    start = round(input_length*0.2)
    end   = round(input_length*0.8) #min(round(input_length*0.8),start+15000)
    #———————————————————————————#
    i = input_i[start:end]
    q = input_q[start:end]
    k = kde.gaussian_kde(np.array([i,q]))   #高斯分布KDE
    nbins = 100
    #———————————————————————————#
    # 3-D density plot
    X, Y = np.mgrid[-1:1:nbins*1j, -1:1:nbins*1j]
    zi = k(np.vstack([X.flatten(), Y.flatten()])).reshape(nbins,nbins)
    #———————————————————————————#
    # 2-D density plot
    #zi = k(np.vstack([X.flatten(), Y.flatten()]))
    #plt.pcolormesh(X, Y, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.jet)
    #———————————————————————————#
    # 3-D histogram
    #hist, xedges, yedges = np.histogram2d(i, q, bins=bins, range=[[-1,1],[-1,1]])
    #X, Y = np.meshgrid(xedges[:-1]+1/nbins, yedges[:-1]+1/nbins, indexing="ij")
    #———————————————————————————#
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, zi, rstride=1, cstride=1, cmap=plt.cm.jet)
    #———————————————————————————#投影
    #ax.contour(X, Y, zi, zdir='x', offset=-2)#, cmap=plt.cm.jet)
    #ax.contour(X, Y, zi, zdir='y', offset=+2)#, cmap=plt.cm.jet)
    #ax.contour(X, Y, zi, zdir='z', offset=-5, cmap=plt.cm.jet)
    ax.set_xlabel('I', color='k')
    ax.set_ylabel('Q', color='k')
    ax.set_zlabel('Probability', color='k')
    #ax.title(lab+' Constellation')
    plt.show()
