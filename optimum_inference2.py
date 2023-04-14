import debugpy
#debugpy.listen(("0.0.0.0", 5678))

from transformers import pipeline, AutoTokenizer, WhisperProcessor, AutoConfig
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
import onnxruntime as rt
from pathlib import Path
from datasets import load_dataset
print("Modules loaded")
from datetime import datetime
import numpy as np

time_list = []

# load model from hub and convert
EP_list = ['CUDAExecutionProvider']
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-medium")
encoder_sess = rt.InferenceSession("models/whisper-medium-speech2seq/onnx/encoder_model.onnx", providers=EP_list)
print("Encoder sess loaded")
decoder_sess = rt.InferenceSession("models/whisper-medium-speech2seq/onnx/decoder_model.onnx", providers=EP_list)
print("Decoder sess loaded")
config = AutoConfig.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = ORTModelForSpeechSeq2Seq(encoder_session=encoder_sess, decoder_session=decoder_sess, config=config, preprocessors=[processor] )
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=processor.feature_extractor)
print("Loading dataset")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_array = ds[0]["audio"]["array"]
outputs = pipe(audio_array)

print("Starting time inference")
for i in range(10):
    now = datetime.now()
    outputs = pipe(audio_array)
    diff = datetime.now() - now
    time_list.append(diff)
    print(outputs)
    print("Time:", diff)

print("Mean time:", np.array(time_list).mean())