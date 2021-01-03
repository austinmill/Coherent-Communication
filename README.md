# Coherent-Communication
This a sample for coherent optical communication based on Python 3.8, that supports QPSK, 8QAM, 16QAM, SP-16QAM, 32-SP-QAM and 128-SP-QAM
signal generation, fiber transmission and receiption. The semi non-ideal models of ADC nad DAC are included.

Download all 4 .py files and placed them in the same folder, run the main.py file, a SP-16QAM will be demonstrated (Python 3 is required).

You could edit the main.py to get other modulation format and parameters, such as OSNR, fiber spans, PMD...

This sample is all based on published papers or books.


本文档是一个基于Python 3.8撰写的相干光通信的样例， 支持QPSK, 8QAM, 16QAM, SP-16QAM, 32-SP-QAM和128-SP-QAM码型的产生，光纤传输和接收。ADC/
DAC的半非理想模型已经包含在内。

请下载全部的4个.py文件，放在同一个文件夹下，并运行main.py文件（请提前安装Python 3），可以得到一个SP-16QAM的展示(OSNR=20dB, 5跨段每段101km)。
运行结果可参考文件夹中的截图。

DSP.py中包含了通用的DSP处理，例如插值，编码/解码，CD估算与补偿，PMD动态补偿（CMA，RDA和DD-LMS三种方法），频偏估算与补偿，相位恢复（BPS方法）。

Optics.py包含了常见的光学效应模型，例如IQ光调制器，偏振分集/相位分集内差接收机，WSS光滤波器，光纤色散/PMD。其中激光器的相位噪声模型还不够完善，原因是现代的激光器
相位噪声极小，在有限长度的时间序列内很难模拟出所需的低频相位噪声，望得到高手的指点。

Test.py中包含了常用的光通信测量手段，例如光谱，电频谱，光/电眼图，光功率测量，星座图和EVM测量。

也可以编辑main.py文件，得到你想要的其他调制码型或不同的参数，例如OSNR，光纤跨段以及PMD。

该文档全部基于已经发表的论文或出版的书籍，可用于光通信领域的技术人员学习与研究。有更多的建议请联系我：austinmill2010@gmail.com
