from transformers import WhisperProcessor, AutoTokenizer
from datasets import load_dataset

WhisperProcessor.from_pretrained("openai/whisper-medium")
AutoTokenizer.from_pretrained("openai/whisper-medium")
load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")