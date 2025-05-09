import os
import torch
import torchaudio
import streamlit as st
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

# Modelin yüklendiği klasör
model_path = "C:\\Users\\sema.nur\\Desktop\\STTdememe\\final_model"

# Yüklemeler
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
tokenizer = WhisperTokenizer.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to("cpu")

st.title("MEÜ Tıp Fakültesi Tıbbi Ses Kayıtları Transkripsiyon Uygulaması")

audio_folder = "C:\\Users\\sema.nur\\Desktop\\STTdememe\\kayitlar-1"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

selected_audio = st.selectbox("Bir ses dosyası seçin:", audio_files)

if selected_audio:
    file_path = os.path.join(audio_folder, selected_audio)
    st.audio(file_path, format='audio/wav')

    if st.button("Transkribe Et"):
        try:
            waveform, sample_rate = torchaudio.load(file_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            waveform = waveform.mean(dim=0)
            total_samples = waveform.size(0)
            segment_samples = 30 * 16000

            full_transcription = ""
            segment_count = 0

            for i, start in enumerate(range(0, total_samples, segment_samples)):
                end = min(start + segment_samples, total_samples)
                segment = waveform[start:end]

                if segment.abs().sum().item() < 1e-5:
                    continue

                # 🔍 Yeni: input_features artık feature_extractor ile hazırlanıyor
                input_features = feature_extractor(
                    segment.numpy(), sampling_rate=16000, return_tensors="pt"
                ).input_features

                if input_features.shape[1] == 0:
                    st.warning(f"❌ Segment {i+1}: input_features boş.")
                    continue

                with torch.no_grad():
                    predicted_ids = model.generate(input_features=input_features.to("cpu"))

                transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                full_transcription += transcription.strip() + " "
                segment_count += 1

            if segment_count > 0:
                st.success("✅ Transkripsiyon tamamlandı:")
                st.text_area("Sonuç", value=full_transcription.strip(), height=300)
            else:
                st.warning("⚠️ Hiçbir segment transkribe edilemedi.")

        except Exception as e:
            st.error(f"🔴 Bir hata oluştu: {e}")
