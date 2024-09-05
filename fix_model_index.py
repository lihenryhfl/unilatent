import json
import argparse

MAP = {
  "clip_image_encoder": [
    "transformers",
    "CLIPVisionModel"
  ],
  "clip_image_processor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "decoder_tokenizer": [
    "transformers",
    "GPT2Tokenizer"
  ],
  "image_decoder_adapter": [
    "utils",
    "EmbedAdapter"
  ],
  "image_encoder_adapter": [
    "utils",
    "EmbedAdapter"
  ],
  "layer_aggregator": [
    "utils",
    "LayerAggregator"
  ],
  "scheduler": [
    "diffusers",
    "FlowMatchEulerDiscreteScheduler"
  ],
  "soft_prompter": [
    "utils",
    "SoftPrompter"
  ],
  "text_decoder": [
    "caption_decoder_v1",
    "TextDecoder"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModelWithProjection"
  ],
  "text_encoder_2": [
    "transformers",
    "CLIPTextModelWithProjection"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "tokenizer_2": [
    "transformers",
    "CLIPTokenizer"
  ],
  "transformer": [
    "transformer",
    "SD3Transformer2DModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}

def edit_fname(fname, suffix='test'):
    tmp = fname.split('/')
    file = tmp[-1].split('.')
    file.insert(-1, suffix)
    tmp[-1] = '.'.join(file)
    fname = '/'.join(tmp)
    assert fname != args.fname
    return fname


parser = argparse.ArgumentParser(description="Fixing a model index with DDP in it.")
parser.add_argument('fname')
parser.add_argument('--for_real', action='store_true')
args = parser.parse_args()

with open(args.fname, 'r') as f:
    json_data = json.load(f)

if args.for_real:
    with open(edit_fname(args.fname, suffix='orig'), 'w') as f:
        json.dump(json_data, f) 

for name in json_data:
    module = json_data[name]
    if module[1] == "DistributedDataParallel":
        json_data[name] = MAP[name]

if not args.for_real:
    args.fname = edit_fname(args.fname, suffix='test')
    
with open(args.fname, 'w') as f:
    json.dump(json_data, f)