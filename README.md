# Disease Prediction from Spoken Symptoms

NLP mini-project: a patient describes their symptoms out loud, in Algerian dialect, and the system predicts the most likely disease.

## Pipeline

1. **Recording**: a short voice description is recorded from the microphone (`sounddevice`).
2. **Speech-to-text**: the audio is transcribed with the ElevenLabs Scribe model (Arabic / Algerian dialect).
3. **Translation and symptom extraction**: GPT-3.5 (through OpenRouter) translates the text to English and extracts the symptoms.
4. **Symptom matching**: each symptom is matched to the closest of the 132 known symptoms using Sentence-Transformers embeddings (`all-MiniLM-L6-v2`) and cosine similarity.
5. **Disease prediction**: a neural network trained on the SymbiPredict 2022 dataset predicts the disease and shows the top 5 candidates.

Example from the notebooks:

| Step | Output |
|---|---|
| Transcription | الضهر يضرني ويوجعني |
| Translation | "My back hurts and it's bothering me." |
| Symptom matching ("shoulder pain") | muscle_pain (0.66), neck_pain (0.62), back_pain (0.61) |

## Data and model

- **Dataset**: SymbiPredict 2022, with 4,961 cases, 132 binary symptoms and 41 diseases (70% train / 30% validation).
- **Model**: Keras dense network (512 → 256 → 128 → 41) with batch normalization, dropout and early stopping.
- **Result**: the model reaches close to 100% validation accuracy. The dataset is very clean (each disease has a consistent symptom pattern), so this mainly shows that the pipeline works, not real clinical performance.

This is a student project and not a medical tool.

## Files

| File | Content |
|---|---|
| `speech_to_text.ipynb` | Recording, transcription, translation and symptom extraction |
| `Disease_predictor_net.ipynb` | Data exploration, model training, symptom matching and prediction |
| `symbipredict_2022.csv` | Dataset |
| `output.wav`, `test2.mp3`, `Record (online-voice-recorder.com).mp3` | Sample recordings |

## How to run

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn sentence-transformers sounddevice scipy python-dotenv elevenlabs openai
```

Copy `.env.example` to `.env` and add your own keys:

```
ELEVENLABS_API_KEY=...
OPENROUTER_API_KEY=...
```

Then run `speech_to_text.ipynb` to turn a recording into symptoms, and `Disease_predictor_net.ipynb` to train the model and predict the disease.

## Technologies

Python, TensorFlow/Keras, Sentence-Transformers, ElevenLabs Speech-to-Text, GPT-3.5 (OpenRouter), Pandas, Scikit-learn, Matplotlib/Seaborn.

## Author

**Amine Benbakreti**: [GitHub](https://github.com/Benbakreti0Amine) · [LinkedIn](https://www.linkedin.com/in/mohammed-el-amine-benbakreti-351b51280/)
