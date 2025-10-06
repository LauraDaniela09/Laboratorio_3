## 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨 𝟑 - 𝐟𝐫𝐞𝐜𝐮𝐞𝐧𝐜𝐢𝐚 𝐲 𝐯𝐨𝐳

*importancion de librerias* 
```python
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

```
<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 𝐚 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

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
*Audio de la señal*

🎧 [AUDIO MUJER 1](Mujer1.wav)

*señal generada*
<p align="center">
<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/14757bf6-05d2-4da9-b1d2-df0050c1f588" />
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
*Audio de la señal*

🎧 [AUDIO MUJER 2](Mujer2.wav)

*señal generada*
<p align="center">
<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/18d09971-acb2-47ad-b98b-75d4d02f7b98" />
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
*Audio de la señal*

🎧 [AUDIO MUJER 3](Mujer3.wav)

*señal generada*
<p align="center">
<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/2fcf033d-8976-4293-91cb-8cbd37fd19aa" />
</p>

```python
# Señal hombre 1
rateh1, man1 = wav.read("/Man1.wav")
if man1.ndim > 1:
    man1 = man1[:,0]

t1 = np.linspace(0, len(man1)/rateh1, num=len(man1))
plt.figure(figsize=(12,4))
plt.plot(t1, man1, color="#C77DFF")
plt.title("Señal de audio: hombre 1")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
display(Audio(man1, rate=rateh1))
```
## resultado
*Audio de la señal*

🎧 [AUDIO HOMBRE 1](Man1.wav)

*señal generada*
<p align="center">
<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/06d82fc7-cf9e-4078-9001-23d8408eeb78" />
</p>

```python
# Señal hombre 2
rateh2, man2 = wav.read("/Man2.wav")
if man2.ndim > 1:
    man2 = man2[:,0]

t2 = np.linspace(0, len(man2)/rateh2, num=len(man2))
plt.figure(figsize=(12,4))
plt.plot(t2, man2, color="#BA55D3")
plt.title("Señal de audio: Hombre 2")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
display(Audio(man2, rate=rateh2))
```
## resultado
*Audio de la señal*

🎧 [AUDIO HOMBRE 2](Man2.wav)

*señal generada*
<p align="center">
<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/aff23ed9-2cb8-4382-a6fc-0966b9f483ae" />
</p>

```python
# señal hombre 3
rateh3, man3 = wav.read("/man 3.wav")
if man3.ndim > 1:
   man3 = man3[:,0]

t3 = np.linspace(0, len(man3)/rateh3, num=len(man3))
plt.figure(figsize=(12,4))
plt.plot(t3,man3, color="#9400D3")
plt.title("Señal de audio: Hombre 3")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
display(Audio(man3, rate=rateh3))
```
## resultado
*Audio de la señal*

🎧 [AUDIO HOMBRE 3](man3.wav)

*señal generada*
<p align="center">
<img width="900" height="200" alt="image" src="https://github.com/user-attachments/assets/265c2a14-9681-4724-9c95-7e243bd389fe" />
</p>

tranformadas de fourier
```python
#MUJER 1
# === Cargar el audio ===
fs, audio = wavfile.read('/Mujer1.wav')

# Convertir a mono si es estéreo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === FFT ===
N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)

# === Solo parte real positiva ===
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N

# === Graficar ===
plt.figure(figsize=(12, 8))

# Parte real positiva
plt.subplot(2, 1, 1)
plt.plot(f, X_real_pos,color="#D36CA0")
plt.title('Transformada de Fourier (AUDIO MUJER 1)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud  ')
plt.grid(True)

# Espectro de magnitud
plt.subplot(2, 1, 2)
plt.plot(f, 20*np.log10(mag + 1e-12),color="#A94F8A" )
plt.title('Espectro de Magnitud ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()
```
## resultado
<p align="center">
<img width="950" height="400" alt="image" src="https://github.com/user-attachments/assets/8baf1b7f-2378-406d-8f27-380e5ff7939d" />
</p>


```python
#MUJER 2
# === Cargar el audio ===
fs, audio = wavfile.read('/Mujer2.wav')

# Convertir a mono si es estéreo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === FFT ===
N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)

# === Solo parte real positiva ===
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N

# === Graficar ===
plt.figure(figsize=(12, 8))

# Parte real positiva
plt.subplot(2, 1, 1)
plt.plot(f, X_real_pos,color="#D36CA0")
plt.title('Transformada de Fourier (AUDIO MUJER 2)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud ')
plt.grid(True)

# Espectro de magnitud
plt.subplot(2, 1, 2)
plt.plot(f, 20*np.log10(mag + 1e-12),color="#A94F8A")
plt.title('Espectro de Magnitud ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()
```
## resultado
<p align="center">
<img width="950" height="400" alt="image" src="https://github.com/user-attachments/assets/31971d0e-4aa9-4d65-8cf7-89541f3e5d8b" />
    </p>



```python
#MUJER 3
# === Cargar el audio ===
fs, audio = wavfile.read('/Mujer3.wav')

# Convertir a mono si es estéreo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === FFT ===
N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)

# === Solo parte real positiva ===
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N

# === Graficar ===
plt.figure(figsize=(12, 8))

# Parte real positiva
plt.subplot(2, 1, 1)
plt.plot(f, X_real_pos,color="#D36CA0")
plt.title('Transformada de Fourier (AUDIO MUJER 3)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud ')
plt.grid(True)

# Espectro de magnitud
plt.subplot(2, 1, 2)
plt.plot(f, 20*np.log10(mag + 1e-12),color="#A94F8A")
plt.title('Espectro de Magnitud ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()
```
## resultado
<p align="center">
<img width="950" height="400" alt="image" src="https://github.com/user-attachments/assets/bf9e2991-724d-4286-ac5d-5e364dbdac9a" />
</p>

```python
#HOMBRE 1
# === Cargar el audio ===
fs, audio = wavfile.read('/Man1.wav')

# Convertir a mono si es estéreo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === FFT ===
N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)

# === Solo parte real positiva ===
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N

# === Graficar ===
plt.figure(figsize=(12, 8))

# Parte real positiva
plt.subplot(2, 1, 1)
plt.plot(f, X_real_pos,color="#32CD32")
plt.title('Transformada de Fourier (AUDIO HOMBRE 1)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud ')
plt.grid(True)

# Espectro de magnitud
plt.subplot(2, 1, 2)
plt.plot(f, 20*np.log10(mag + 1e-12),color="#00FF7F")
plt.title('Espectro de Magnitud ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## resultado
<p align="center">
<img width="950" height="400" alt="image" src="https://github.com/user-attachments/assets/56bbd888-96fa-46aa-9b05-9aa0ce4b0600" />
</p>

```python
#HOMBRE 2
# === Cargar el audio ===
fs, audio = wavfile.read('/Man2.wav')

# Convertir a mono si es estéreo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === FFT ===
N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)

# === Solo parte real positiva ===
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N

# === Graficar ===
plt.figure(figsize=(12, 8))

# Parte real positiva
plt.subplot(2, 1, 1)
plt.plot(f, X_real_pos, color="#32CD32")
plt.title('Transformada de Fourier (AUDIO HOMBRE 2)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud ')
plt.grid(True)

# Espectro de magnitud
plt.subplot(2, 1, 2)
plt.plot(f, 20*np.log10(mag + 1e-12), color="#00FF7F")
plt.title('Espectro de Magnitud ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()
```
## resultado
<p align="center">
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/5ec55699-dff0-487f-afa8-c86c511f9444" />
</p>

```python
#HOMBRE 3
# === Cargar el audio ===
fs, audio = wavfile.read('/man 3.wav')

# Convertir a mono si es estéreo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === FFT ===
N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)

# === Solo parte real positiva ===
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N

# === Graficar ===
plt.figure(figsize=(12, 8))

# Parte real positiva
plt.subplot(2, 1, 1)
plt.plot(f, X_real_pos,color="#32CD32")
plt.title('Transformada de Fourier (AUDIO HOMBRE 3)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud ')
plt.grid(True)

# Espectro de magnitud
plt.subplot(2, 1, 2)
plt.plot(f, 20*np.log10(mag + 1e-12),color="#00FF7F")
plt.title('Espectro de Magnitud ')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)

plt.tight_layout()
plt.show()
```
## resultado
<p align="center">
<img width="200" height="400" alt="image" src="https://github.com/user-attachments/assets/82225b2f-82d0-4c0e-9541-1eadd3ca4edb" />
</p>


---


