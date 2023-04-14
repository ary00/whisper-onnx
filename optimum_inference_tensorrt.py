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

provider_options = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "/temp/trt_cache_example"
}

# load model from hub and convert
EP_list = ['TensorrtExecutionProvider']
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-medium")
encoder_sess = rt.InferenceSession("models/whisper-medium-speech2seq/onnx/encoder_model.onnx", providers=EP_list)
print("Encoder sess loaded")
decoder_sess = rt.InferenceSession("models/whisper-medium-speech2seq/onnx/decoder_model.onnx", providers=EP_list)
print("Decoder sess loaded")
config = AutoConfig.from_pretrained("openai/whisper-medium")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = ORTModelForSpeechSeq2Seq(encoder_session=encoder_sess,
                                decoder_session=decoder_sess, 
                                config=config, 
                                preprocessors=[processor])
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

print("Loading dataset")
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

input_features = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt").input_features.cuda()
predicted_ids = model.generate(inputs=input_features, forced_decoder_ids=forced_decoder_ids)
outputs = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
print(outputs)

print("Starting time inference")
for i in range(10):
    now = datetime.now()
    input_features = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt").input_features.cuda()
    predicted_ids = model.generate(inputs=input_features, forced_decoder_ids=forced_decoder_ids)
    outputs = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    diff = datetime.now() - now
    time_list.append(diff)
    print(outputs)
    print("Time:", diff)

print("Mean time:", np.array(time_list).mean())
