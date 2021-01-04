# Coherent-Communication
This a sample for coherent optical communication based on Python 3.8, that supports QPSK, 8QAM, 16QAM, SP-16QAM, 32-SP-QAM and 128-SP-QAM
signal generation, fiber transmission and receiption.

Download all 4 .py files and placed them in the same folder, run the main.py file (Python 3 is required), a SP-16QAM with 20dB OSNR and 
5x101km spans will be demonstrated. The result is shown in the snapshots in the folder.

DSP.py includes general digital processing, such as interplotation (No, I didn't use thr resample function from Matlabs, but simple 
linear interplotation instead for HW feasbility), QAM code/decode, semi non-ideal models of ADC and DAC, ADC sampling phase adjustment,
CD estimation and compensation, PMD dynamic compensation (CMA, RDA and DD-LMS), frequency offset estimation and compensation, carrier
phase recovery (Blind Phase searching).

Optics.py includes the models of common used optical effects, such as IQ optical modulation, pol-diversity/phase-diversity intradyne 
receiver, WSS optical filter, CD and PMD of fiber. The model of phase noise of laser is not well done, since the modern lasers have very
low phase noise, it is tough to simulate the low-frequency phase noise with finite length of time sequence. Need you help, guys!

Test.py includes the common used optical tester/measurement, such as OSA, ESA, O/E eye diagram, optical power, constellation and EVM test.

You could also edit the main.py to get other modulation formats and parameters, such as different Baudrate, OSNR, fiber spans, PMD...

This sample is all based on published papers or books, could be used by the researchers of optical communication. Please feel free to 
contact me if any comments or suggestion. Email: austinmill2010@gmail.com


本文档是一个基于Python 3.8撰写的相干光通信的样例， 支持QPSK, 8QAM, 16QAM, SP-16QAM, 32-SP-QAM和128-SP-QAM码型的产生，光纤传输和接收。ADC/
DAC的半非理想模型已经包含在内。

请下载全部的4个.py文件，放在同一个文件夹下，并运行main.py文件（请提前安装Python 3），可以得到一个SP-16QAM的展示(OSNR=20dB, 5跨段每段101km)。
运行结果可参考文件夹中的截图。

DSP.py中包含了通用的DSP处理，例如插值(我没有使用Matlabs的resample函数，而是基于硬件可实现性使用了简单的线性插值)，QAM编码/解码，ADC/DAC半非
理想模型，ADC采样相位调整，CD估算与补偿，PMD动态补偿（CMA，RDA和DD-LMS三种方法），频偏估算与补偿，相位恢复（BPS方法）。

Optics.py包含了常见的光学效应模型，例如IQ光调制器，偏振分集/相位分集内差接收机，WSS光滤波器，光纤色散/PMD。其中激光器的相位噪声模型还不够完善，
原因是现代的激光器相位噪声极小，在有限长度的时间序列内很难模拟出所需的低频相位噪声，望得到高手的指点。

Test.py中包含了常用的光通信测量手段，例如光谱，电频谱，光/电眼图，光功率测量，星座图和EVM测量。

也可以编辑main.py文件，得到你想要的其他调制码型或不同的参数，例如不同的波特率, OSNR，光纤跨段以及PMD。

该文档全部基于已经发表的论文或出版的书籍，可用于光通信领域的技术人员学习与研究。有更多的建议请联系我：austinmill2010@gmail.com
