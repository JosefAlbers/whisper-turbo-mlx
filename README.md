# WTM (Whisper Turbo MLX)

This repository provides a fast and lightweight implementation of the [Whisper](https://github.com/openai/whisper/discussions/2363) model using [MLX](https://github.com/ml-explore/mlx-examples/tree/main/whisper), all contained within a single file of under 300 lines, designed for efficient audio transcription.

![Alt text](https://raw.githubusercontent.com/JosefAlbers/whisper-turbo-mlx/main/assets/benchmark.png)

## Installation

```zsh
brew install ffmpeg
git clone https://github.com/JosefAlbers/whisper-turbo-mlx.git
cd whisper-turbo-mlx
pip install -e .
```

## Quick Start

To transcribe an audio file:

```zsh
wtm test.wav
```

To use the library in a Python script:

```python
>>> from whisper_turbo import transcribe
>>> transcribe('test.wav', any_lang=True)
```

## Quick Parameter

The `quick` parameter allows you to choose between two transcription methods: 

- **`quick=True`**: Utilizes a parallel processing method for faster transcription. This method may produce choppier output but is significantly quicker, ideal for situations where speed is a priority (e.g., for feeding the generated transcripts into an LLM to collect quick summaries on many audio recordings).
  
- **`quick=False`** (default): Engages a recurrent processing method that is slower but yields more faithful and coherent transcriptions (still faster than other reference implementations).

You can specify this parameter when calling the `transcribe` function:

```zsh
wtm --quick=True
```

```python
>>> transcribe('test.wav', quick=True)
```

## Acknowledgements

This project builds upon the reference [MLX implementation](https://github.com/ml-explore/mlx-examples/tree/main/whisper) of the Whisper model. Great thanks to the contributors of the MLX project for their exceptional work and inspiration.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.