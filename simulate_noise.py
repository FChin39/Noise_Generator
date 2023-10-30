import random
import os
import glob
import numpy as np
import soundfile as sf
import subprocess
import roomsimove_single
import olafilt
import time

# you should put all noise .wav files under a directory
noise_paths = glob.glob(r'C:\\Users\\qi0002ai\Desktop\workspace\data\\noise\\*.wav')
# specify a range of snrs in dB
# noise_snrs = (0, 5, 10, 15, 20)
noise_snrs = (5, 10, 15, 20)

def add_noise(clean_wav, clean_wav_sr):
    # noise_path = random.choice(noise_paths)
    noise_path = r'C:\\Users\\qi0002ai\Desktop\workspace\data\\noise\\ntu-schaeffler_noise-16k_trimmed_cliped.wav'
    # noise_wav, _ = sf.read(noise_path)

    command = [
        'ffmpeg',
        '-y',
        '-i', noise_path,
        '-ar', str(8000), 
        '-filter:a', f'atempo={1.0}',
        noise_path[:-4] + '8000.wav',
        '-loglevel', 'quiet',
    ]
    subprocess.run(command)


    noise_wav, noise_wav_sr = sf.read(noise_path[:-4] + '8000.wav')
    noise_snr = random.choice(noise_snrs)

    clean_wav = clean_wav.astype(np.float32)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav) / len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        max_start = len(noise_wav) - len(clean_wav)
        if max_start >= 0:
            start = random.randint(0, max_start)
        else:
            start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)] 

    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10 ** (noise_snr / 20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    # Avoid clipping noise
    max_float16 = np.finfo(np.float16).max
    min_float16 = np.finfo(np.float16).min
    if mixed.max(axis=0) > max_float16 or mixed.min(axis=0) < min_float16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            reduction_rate = max_float16 / mixed.max(axis=0)
        else:
            reduction_rate = max_float16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)

    return mixed


### main function
audio_files = glob.glob(os.path.join(r'C:\\Users\\qi0002ai\Desktop\workspace\data\\clean_trainset_56spk_wav\\', '*.wav'))

total_files = len(audio_files)
flag = 0
start_time = time.time()

for audio_file in audio_files:

    command = [
        'ffmpeg',
        '-y',
        '-i', audio_file,
        '-ar', str(8000),
        '-filter:a', f'atempo={1.0}',
        audio_file[:-4] + '_8000.wav',
        '-loglevel', 'quiet',
    ]
    subprocess.run(command)

    print('handling: ', audio_file)

    clean_wav, clean_wav_sr = sf.read(audio_file[:-4] + '_8000.wav')

    # add Room impulse response
    room_dim = [4.2, 3.4, 5.2]
    room = roomsimove_single.Room(room_dim)
    mic_pos = [2, 2, 2]
    mic1 = roomsimove_single.Microphone(mic_pos, 1,  \
            orientation=[0.0, 0.0, 0.0], direction='omnidirectional')
    mic_pos = [2, 2, 1]
    mic2 = roomsimove_single.Microphone(mic_pos, 2,  \
            orientation=[0.0, 0.0, 0.0], direction='cardioid')
    mics = [mic1, mic2]
    sample_rate = 16000
    sim_rir = roomsimove_single.RoomSim(sample_rate, room, mics, RT60=0.3)
    source_pos = [1, 1, 1]
    rir = sim_rir.create_rir(source_pos)
    reverb_data = olafilt.olafilt(rir[:,1],clean_wav)



   # add Noise
    dest_folder = r'C:\\Users\\qi0002ai\Desktop\workspace\data\\output\\augmentation_large\\'
    os.makedirs(dest_folder, exist_ok=True)


    noisy_wav = add_noise(reverb_data, clean_wav_sr)

    file_name = os.path.basename(audio_file)
    noisy_file_name = f'{file_name[:-4]}_noisy.wav'
    dest_path = os.path.join(dest_folder, noisy_file_name)

    sf.write(dest_path, noisy_wav, clean_wav_sr)

    flag += 1

    end_time = time.time()

    print(f'Progress: {flag}/{total_files}, Time_cost: {end_time - start_time:.2f}s')

    # file_to_delete = audio_file[:-4] + '8000.wav'
    # try:
    #     os.remove(file_to_delete)  
    # except Exception as e:
    #     print(e)

print('Done!')