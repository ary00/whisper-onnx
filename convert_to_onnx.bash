#!/bin/bash
MODEL=models/whisper-medium-fp16
optimum-cli export onnx --device cuda \
--framework pt \
--optimize O4 \
--task speech2seq-lm-with-past \
--model openai/whisper-medium \
$MODEL