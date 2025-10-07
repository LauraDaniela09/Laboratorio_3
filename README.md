# L𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨 𝟑 - F𝐫𝐞𝐜𝐮𝐞𝐧𝐜𝐢𝐚 𝐲 V𝐨𝐳
𝙞𝙣𝙩𝙧𝙤𝙙𝙪𝙘𝙘𝙞ó𝙣

En este laboratorio se realizó un análisis espectral de la voz  empleando técnicas de procesamiento digital de señales. Se utilizaron grabaciones de voces masculinas y femeninas para calcular parámetros como la frecuencia fundamental, frecuencia media, brillo e intensidad.

𝙤𝙗𝙟𝙚𝙩𝙞𝙫𝙤

Aplicar la Transformada de Fourier para analizar y comparar las características espectrales de voces masculinas y femeninas, identificando sus diferencias en frecuencia e intensidad para comprender mejor el comportamiento acústico de la voz humana.

𝙞𝙢𝙥𝙤𝙧𝙩𝙖𝙘𝙞ó𝙣 𝙙𝙚 𝙡𝙞𝙗𝙧𝙚𝙧𝙞𝙖𝙨
```python
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from scipy.signal import find_peaks
from scipy.io import wavfile
```
Esa parte del código muestra la importación de librerías necesarias para trabajar con archivos de audio y analizarlos:`scipy.io.wavfile` y `wavfile` para leer y escribir archivos de audio `(.wav)`.`matplotlib.pyplot` para graficar señales.`numpy` para realizar operaciones numéricas y de matrices. `IPython.display.Audio` para reproducir el audio directamente en el notebook.`scipy.signal.find_peaks`para detectar picos o puntos importantes en la señal.

<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 A 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>


```mermaid
flowchart TD
  A[Inicio - Parte A: Preparación] --> B[Seleccionar hablantes (masculino / femenino)]
  B --> C[Preparar ambiente de grabación: control de ruido, micrófono, distancia]
  C --> D[Dar instrucciones al sujeto: emitir vocales o frases]
  D --> E[Grabar audio y guardar con etiquetas]
  E --> F[Verificar calidad de la grabación: SNR, clipping]
  F --> G{¿Calidad aceptable?}
  G -->|Sí| H[Almacenar archivo y metadatos]
  G -->|No| I[Repetir grabación o descartar]
  I --> F
  H --> J[Fin Parte A - Listo para procesamiento]
```

En esta parte del codigo se utiliza la función `wav.read()` de `SciPy` para cargar el archivo  y obtener su frecuencia de muestreo y datos de la señal. Si el audio tiene más de un canal, se selecciona solo uno para trabajar en mono. Luego, con `np.linspace()` de `NumPy`, se crea el eje de tiempo para cada muestra. La librería `Matplotlib (plt.plot())` se usa para graficar la señal, mostrando la amplitud frente al tiempo. Finalmente, con `Audio()` de `IPython.display`, se reproduce el sonido directamente en el entorno de ejecución.este procedimiento se realiza con cada una de las señales tanto de mujeres como para hombres.


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
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/14757bf6-05d2-4da9-b1d2-df0050c1f588" />
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
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/18d09971-acb2-47ad-b98b-75d4d02f7b98" />
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
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/2fcf033d-8976-4293-91cb-8cbd37fd19aa" />
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
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/06d82fc7-cf9e-4078-9001-23d8408eeb78" />
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
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/aff23ed9-2cb8-4382-a6fc-0966b9f483ae" />
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
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/265c2a14-9681-4724-9c95-7e243bd389fe" />
</p>

𝙩𝙧𝙖𝙣𝙨𝙛𝙤𝙧𝙢𝙖𝙙𝙖 𝙙𝙚 𝙛𝙤𝙪𝙧𝙞𝙚𝙧 𝙮 𝙚𝙨𝙥𝙚𝙘𝙩𝙧𝙤 𝙙𝙚 𝙢𝙖𝙜𝙣𝙞𝙩𝙪𝙙

Posteriormente para poder obtener la transformada de fourier y el espectro lo que se realiza es cargar el audio,  conviertirlo a mono y aplicar la Transformada de Fourier (FFT) para obtener sus componentes en frecuencia. Luego se grafican dos resultados: la Transformada de Fourier, que muestra cómo se distribuyen las frecuencias del sonido, y el espectro de magnitud en decibelios, que indica la intensidad de cada frecuencia presente en la señal.

```python
#MUJER 1
fs, audio = wavfile.read('/Mujer1.wav')

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N
plt.figure(figsize=(12, 8))

# Transformada de fourier
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
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/8baf1b7f-2378-406d-8f27-380e5ff7939d" />
</p>


```python
#MUJER 2
fs, audio = wavfile.read('/Mujer2.wav')

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N
plt.figure(figsize=(12, 8))

#Transformada de fourier
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
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/31971d0e-4aa9-4d65-8cf7-89541f3e5d8b" />
    </p>



```python
#MUJER 3
fs, audio = wavfile.read('/Mujer3.wav')

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N
plt.figure(figsize=(12, 8))

# Transformada de fourier
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
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/bf9e2991-724d-4286-ac5d-5e364dbdac9a" />
</p>

```python
#HOMBRE 1
fs, audio = wavfile.read('/Man1.wav')

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N
plt.figure(figsize=(12, 8))

#Transformada de fourier
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
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/56bbd888-96fa-46aa-9b05-9aa0ce4b0600" />
</p>

```python
#HOMBRE 2
fs, audio = wavfile.read('/Man2.wav')

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N
plt.figure(figsize=(12, 8))

# Transformada de fourier
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
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/5ec55699-dff0-487f-afa8-c86c511f9444" />
</p>

```python
#HOMBRE 3
fs, audio = wavfile.read('/man 3.wav')
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

N = len(audio)
X = np.fft.fft(audio)
f = np.linspace(0, fs/2, N//2)
X_real_pos = np.abs(X.real[:N//2])  # valor absoluto elimina negativos
mag = np.abs(X[:N//2]) / N
plt.figure(figsize=(12, 8))

# Transformada de fourier
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
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/82225b2f-82d0-4c0e-9541-1eadd3ca4edb" />
</p>

𝙞𝙙𝙚𝙣𝙩𝙞𝙛𝙞𝙘𝙖𝙘𝙞𝙤𝙣 𝙮 𝙧𝙚𝙥𝙤𝙧𝙩𝙚 𝙙𝙚 𝙡𝙖𝙨 𝙘𝙖𝙧𝙖𝙘𝙩𝙚𝙧𝙞𝙨𝙩𝙞𝙘𝙖𝙨 𝙙𝙚 𝙘𝙖𝙙𝙖 𝙨𝙚ñ𝙖𝙡

El código carga los audios, los convierte a mono y los normaliza. Luego calcula su frecuencia fundamental (f₀) con `find_peaks()`, la frecuencia media, el brillo (energía en frecuencias altas) y la intensidad RMS (energía total del sonido). Finalmente, muestra estos valores en una tabla para comparar las características de cada audio.

```python
def calcular_caracteristicas(ruta):
    fs, audio = wavfile.read(ruta)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # convertir a mono
    audio = audio / np.max(np.abs(audio))  # normalizar
    
    N = len(audio)
    fft_vals = np.abs(np.fft.fft(audio)[:N//2])
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    
    peaks, _ = find_peaks(fft_vals, height=np.max(fft_vals)*0.1)
    f0 = freqs[peaks[0]] if len(peaks) > 0 else 0
    
    f_media = np.sum(freqs * fft_vals) / np.sum(fft_vals)
    
    # --- Brillo (energía por encima de 1500 Hz) ---
    idx_brillo = freqs > 1500
    brillo = np.sum(fft_vals[idx_brillo]) / np.sum(fft_vals)
    
    # --- Intensidad RMS ---
    intensidad = np.sqrt(np.mean(audio**2))
    
    return f0, f_media, brillo, intensidad

archivos = [
    "/Man1.wav", "/Man2.wav", "/man 3.wav",
    "/Mujer1.wav", "/Mujer2.wav", "/Mujer3.wav"
]

print("\nRESULTADOS DE CADA AUDIO:\n")
print(f"{'Archivo':<15} {'F fund (Hz)':<12} {'F media (Hz)':<15} {'Brillo':<10} {'Intensidad':<12}")

for ruta in archivos:
    f0, f_media, brillo, intensidad = calcular_caracteristicas(ruta)
    print(f"{ruta:<15} {f0:<12.2f} {f_media:<15.2f} {brillo:<10.3f} {intensidad:<12.4f}")

```
## resultado

<img width="500" height="194" alt="image" src="https://github.com/user-attachments/assets/72d6d7eb-34e5-4f73-83f7-df2b3f0d429c" />

<h1 align="center"><i><b>𝙋𝙖𝙧𝙩𝙚 𝘽 𝙙𝙚𝙡 𝙡𝙖𝙗𝙤𝙧𝙖𝙩𝙤𝙧𝙞𝙤</b></i></h1>

### Filtro pasabanda hombre

Antes de iniciar el codigo se desarrolló a mano el filtro pasabanda para poder encontrar el orden necesario y definir los parametros. 

*IMAGEN DE LOS CALCULOS DE SOFI


Primero se importan las librerias, se lee el archivo `/Man1.wav` y guarda la frecuencia de muestreo en `ratem1` y los datos de la señal en `Man1`.
Después define los parametros del filtro pasabanda como la frecuencia de corte baja y alta basandose en los valores teoricos. Para hombres está el rango de 80-400Hz.

```python
from scipy import signal
from scipy.io import wavfile
ratem1, Man1 = wav.read("/Man1.wav")

f_low = 80
f_high = 400
order = 4
fs = ratem1
nyquist = fs / 2
low = f_low / nyquist
high = f_high / nyquist

b, a = signal.butter(order, [f_low/nyquist, f_high/nyquist], btype='bandpass')
Man1_filtrada = signal.filtfilt(b, a, Man1)
t = np.linspace(0, len(Man1)/fs, len(Man1))
```
Se asigna la frecuencia de muestreo `ratem1` a la variable `fs` y se calculan las frecuencias normalizadas y la de Nyquist.
Se diseña el filtro pasabanda tipo butterworth que devuelve los coeficientes del filtro en `b` (numerador) y `a` (denominador).
Aplica el filtro a la señal usando `filtfilt`, que filtra hacia adelante y hacia atrás para eliminar el desfase (fase cero).

*CODIGO E IMAGEN DE BODE

### Filtro pasabanda mujer

Se repite el mismo proceso para la señal de mujer pero con rango de 150-500Hz.
<img width="966" height="1280" alt="image" src="https://github.com/user-attachments/assets/53dc9c1b-3ec4-4072-8996-6583ac933209" />
<img width="1097" height="1210" alt="image" src="https://github.com/user-attachments/assets/4adf867e-bc3f-4c5c-80e6-1dc103d8a0f6" />

```python
from scipy import signal
from scipy.io import wavfile
ratem1, Mujer1 = wav.read("/Mujer1.wav")

f_low = 150
f_high = 500
order = 2
fs = ratem1
nyquist = fs / 2
low = f_low / nyquist
high = f_high / nyquist

b, a = signal.butter(order, [f_low/nyquist, f_high/nyquist], btype='bandpass')
Mujer1_filtrada = signal.filtfilt(b, a, Mujer1)
t = np.linspace(0, len(Mujer1)/fs, len(Mujer1))

plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(t, Mujer1, color='#6A5ACD')
plt.title("Señal original (voz mujer)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.subplot(2,1,2)
plt.plot(t, Mujer1_filtrada, color='#9A00FF')
plt.title("Señal filtrada (pasa banda 150–500 Hz, Butterworth orden 2)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()
```
Se grafica la respuesta de frecuencia (Diagrama Bode de magnitud)

## Grafico de respuesta de frecuencia (bode)

```pythom
w, h = signal.freqz(b, a, worN=4096)
f = (w/np.pi) * (fs/2)

plt.figure(figsize=(8,4))
plt.plot(f, 20*np.log10(abs(h)))
plt.title("Respuesta en frecuencia del filtro pasa-banda (zoom 0–1000 Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.xlim(0, 1000)
plt.ylim(-60, 5)
plt.grid(True)
plt.show()
```
<img width="716" height="393" alt="image" src="https://github.com/user-attachments/assets/1629a50d-9b10-4816-bfec-5510a41f71e1" />


## Medición jitter y shimmer

Se importan las librerias, se define la función `jitter_shimmer` donde se recibe la señal de voz, se calculan el jitter y shimmer y sus porcentajes.
Finalmente se visualiza una grafica donde se ve en azul la señal original de voz, en rojo los cruces por ceros y en verde los picos. Y una tabla con los resultados de cada señal.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
import pandas as pd

#Función para calcular jitter y shimmer
def jitter_shimmer(audio, fs, nombre):
    # Normalizar señal
    audio = audio.astype(float)
    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))

    #Cruces por cero
    cruces = np.where(np.diff(np.sign(audio)) > 0)[0]
    if len(cruces) < 2:
        print(f"{nombre}: no hay suficientes cruces por cero\n")
        return np.nan, np.nan, np.nan, np.nan

    Ti = np.diff(cruces) / fs
    jitter_abs = np.mean(np.abs(np.diff(Ti)))
    jitter_rel = (jitter_abs / np.mean(Ti)) * 100

    #Picos
    peaks, _ = find_peaks(audio, height=0)
    if len(peaks) < 2:
        print(f"{nombre}: no hay suficientes picos\n")
        return jitter_abs, jitter_rel, np.nan, np.nan

    Ai = audio[peaks]
    shimmer_abs = np.mean(np.abs(np.diff(Ai)))
    shimmer_rel = (shimmer_abs / np.mean(Ai)) * 100

    #Graficar señal completa
    t = np.linspace(0, len(audio)/fs, len(audio))
    plt.figure(figsize=(10, 4))
    plt.plot(t, audio, 'b', label="Señal")
    plt.plot(t[cruces], audio[cruces], 'ro', markersize=3, label="Cruces")
    plt.plot(t[peaks], audio[peaks], 'go', markersize=3, label="Picos")
    plt.title(f"{nombre} - Señal completa")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    return jitter_abs, jitter_rel, shimmer_abs, shimmer_rel



archivos = {
    "Mujer1": "/content/Mujer1.wav",
    "Mujer2": "/content/Mujer2.wav",
    "Mujer3": "/content/Mujer3.wav",
    "Hombre1": "/content/Man1.wav",
    "Hombre2": "/content/Man2.wav",
    "Hombre3": "/content/man 3.wav"
}

#Calculos
resultados = []

for nombre, ruta in archivos.items():
    fs, audio = wavfile.read(ruta)
    if audio.ndim > 1:
        audio = audio[:, 0]

    jitter_abs, jitter_rel, shimmer_abs, shimmer_rel = jitter_shimmer(audio, fs, nombre)
    resultados.append([nombre, jitter_abs, jitter_rel, shimmer_abs, shimmer_rel])

#Resultados en tabla
df = pd.DataFrame(resultados, columns=["Señal", "Jitter_abs (s)", "Jitter_rel (%)", "Shimmer_abs", "Shimmer_rel (%)"])

print("\n Resultados de Jitter y Shimmer \n")
print(df.to_string(index=False, justify='center', formatters={
    "Jitter_abs (s)": "{:.6f}".format,
    "Jitter_rel (%)": "{:.2f}".format,
    "Shimmer_abs": "{:.6f}".format,
    "Shimmer_rel (%)": "{:.2f}".format
}))
```
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/af6e28ad-2e1e-4465-a97e-959a65d207b0" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/0f465875-2ab9-41e6-a1d2-b6f984b71353" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/03d6e32f-79ab-44f3-accb-9dfc781bd8c4" />
<img width="857" height="393" alt="image" src="https://github.com/user-attachments/assets/bcd84f67-e49e-4635-8c00-7151cd49a1aa" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/f631cde4-3cfc-4b4d-a245-fe52b6d5c244" />
<img width="866" height="393" alt="image" src="https://github.com/user-attachments/assets/3cce8e3c-c1ec-46ae-8d92-f53375ecf45e" />
<img width="873" height="248" alt="image" src="https://github.com/user-attachments/assets/f1c021ce-8420-47c8-aa4f-bac3c83b5c64" />

## Jitter y shimmer de la señal de mujer filtrada

posteriormente calculamos los mismos parametros de jitter y shimmer pero para la señal filtrada con el proposito de comparar que tanto afectaba el filtrado a estos valores
```python
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import pandas as pd

#Función para calcular Jitter y Shimmer
def calcular_jitter_shimmer(signal_data, fs):
    # Cruces por cero para medir periodos
    cruces = np.where(np.diff(np.sign(signal_data)))[0]
    periodos = np.diff(cruces) / fs
    if len(periodos) < 2:
        return np.nan, np.nan, np.nan, np.nan

    #Jitter
    jitter_abs = np.mean(np.abs(np.diff(periodos)))
    jitter_rel = 100 * (jitter_abs / np.mean(periodos))

    #Shimmer
    amplitudes = []
    for i in range(len(cruces) - 1):
        amplitudes.append(np.max(signal_data[cruces[i]:cruces[i+1]]) -
                          np.min(signal_data[cruces[i]:cruces[i+1]]))
    amplitudes = np.array(amplitudes)

    shimmer_abs = np.mean(np.abs(np.diff(amplitudes)))
    shimmer_rel = 100 * (shimmer_abs / np.mean(amplitudes))

    return jitter_abs, jitter_rel, shimmer_abs, shimmer_rel

#Calcular métricas
ja, jr, sa, sr = calcular_jitter_shimmer(Mujer1_filtrada, ratem1)

#Mostrar resultados en tabla
df_resultados = pd.DataFrame([{
    "Señal": "Mujer1_filtrada",
    "Jitter abs [s]": ja,
    "Jitter rel [%]": jr,
    "Shimmer abs": sa,
    "Shimmer rel [%]": sr,
}])

df_resultados.index = ["Resultados"]
print(df_resultados.T)

```
<img width="419" height="168" alt="image" src="https://github.com/user-attachments/assets/34c3dc0a-fc29-4282-be6e-18918bc12a0e" />



<h1 align="center"><i><b>𝙋𝙖𝙧𝙩𝙚 𝘾 𝙙𝙚𝙡 𝙡𝙖𝙗𝙤𝙧𝙖𝙩𝙤𝙧𝙞𝙤</b></i></h1>
se respondera las siguientes preguntas con respecto a los resultados obtenidos 

- *¿Qué diferencias se observan en la frecuencia fundamental?*
 
La frecuencia fundamental (F0) fue significativamente más alta en las voces femeninas que en las masculinas.Esto se debe a que las cuerdas vocales de las mujeres son más cortas y delgadas, lo que hace que vibren a una mayor velocidad, en cambio las voces masculinas presentan una F0 más baja, ya que sus cuerdas vocales son más largas y con mayor masa, lo que genera vibraciones más lentas.Por tanto, los hombres tienden a producir sonidos más graves (bajos) y las mujeres sonidos más agudos (altos).

- *¿Qué otras diferencias se notan en términos de brillo, media o intensidad?*

Tambien se puede concluir que las voces masculinas suelen mostrar mayor intensidad (nivel de presión sonora o volumen) debido a una mayor capacidad pulmonar y resonadores más amplios, mientras que las voces femeninas presentan un brillo espectral más acentuado, es decir, más energía en frecuencias altas, lo que las hace sonar más claras o agudas.

En cuanto a la media de amplitud, los hombres tienden a tener valores algo superiores por la potencia de emisión, mientras que las mujeres muestran menor amplitud pero más variabilidad en las frecuencias.

- *Importancia clínica del jitter y shimmer en el análisis de la voz*

El jitter mide la variabilidad en la frecuencia (diferencias entre periodos sucesivos de vibración), El shimmer mide la variabilidad en la amplitud (cambios en la intensidad entre ciclos).Ambos parámetros son indicadores de estabilidad vocal: en una voz sana, las variaciones son mínimas, mientras que en voces con alteraciones (nódulos, pólipos, disfonías, parálisis, etc.) los valores aumentan.

Por eso, tienen gran valor clínico, ya que ayudan a detectar disfunciones laríngeas,permiten cuantificar la severidad de la alteración vocal,Sirven para evaluar la eficacia de terapias o tratamientos.Sin embargo, su interpretación requiere precaución, ya que pueden verse afectados por el nivel de intensidad, ruido ambiental o calidad de la grabación.por eso, se recomienda analizarlos junto con otros parámetros como la relación armónico-ruido (HNR) o medidas cepstrales.

- *Conclusiones*

La frecuencia fundamental es el parámetro que más diferencia se muestra en hombres y mujeres, donde las voces femeninas presentan valores mayores de F0.
Las voces masculinas tienden a ser más intensas y con menor brillo, mientras que las femeninas son más agudas y con mayor energía en las frecuencias altas.
Las diferencias en jitter y shimmer es poca, ya que ambos presentan niveles similares cuando se analizan voces sanas y grabadas en condiciones controladas.
