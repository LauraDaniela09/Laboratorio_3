## laboratorio 3 

*## importancion de librerias* 
```python
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

```
<h1 align="center"><i><b>ᴘᴀʀᴛᴇ ᴀ ᴅᴇʟ ʟᴀʙᴏʀᴀᴛᴏʀɪᴏ</b></i></h1>

```python
#Señal mujer 1
ratem1, Mujer1 = wav.read("/Mujer1.wav")
if Mujer1.ndim > 1:
    Mujer1 = Mujer1[:,0]

t1 = np.linspace(0, len(Mujer1)/ratem1, num=len(Mujer1))
plt.figure(figsize=(12,4))
plt.plot(t1, Mujer1, color="#FFC1CC")
plt.title("Señal de audio: Mujer1")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
display(Audio(Mujer1, rate=ratem1))
```
## resultado

<p align="center">
<img width="1034" height="393" alt="image" src="https://github.com/user-attachments/assets/14757bf6-05d2-4da9-b1d2-df0050c1f588" />
</p>

```python
# Señal mujer 2
ratem2, Mujer2 = wav.read("/Mujer2.wav")
if Mujer2.ndim > 1:
    Mujer2 = Mujer2[:,0]

t2 = np.linspace(0, len(Mujer2)/ratem2, num=len(Mujer2))
plt.figure(figsize=(12,4))
plt.plot(t2, Mujer2, color="#FF69B4")
plt.title("Señal de audio: Mujer2")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
display(Audio(Mujer2, rate=ratem2))
```
## resultado

<p align="center">
<img width="1034" height="393" alt="image" src="https://github.com/user-attachments/assets/18d09971-acb2-47ad-b98b-75d4d02f7b98" />
</p>

```python
# señal mujer 3
ratem3, Mujer3 = wav.read("/Mujer3.wav")
if Mujer3.ndim > 1:
    Mujer3 = Mujer3[:,0]

t3 = np.linspace(0, len(Mujer3)/ratem3, num=len(Mujer3))
plt.figure(figsize=(12,4))
plt.plot(t3, Mujer3, color="#FF1493")
plt.title("Señal de audio: Mujer3")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
display(Audio(Mujer3, rate=ratem3))
```
## resultado

<p align="center">
<img width="1034" height="393" alt="image" src="https://github.com/user-attachments/assets/2fcf033d-8976-4293-91cb-8cbd37fd19aa" />
</p>
