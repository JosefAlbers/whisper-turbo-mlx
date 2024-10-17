# WTM (Whisper Turbo MLX)

This repository provides a fast and lightweight implementation of the [Whisper](openai/whisper-large-v3-turbo) model using [MLX](https://github.com/ml-explore/mlx-examples/tree/main/whisper), all contained within a single file of under 250 lines, designed for efficient audio transcription.

![Alt text](https://raw.githubusercontent.com/JosefAlbers/whisper-turbo-mlx/main/assets/benchmark.png)

## Installation

```zsh
brew install ffmpeg
git clone https://github.com/JosefAlbers/whisper-turbo-mlx.git
cd whisper-turbo-mlx
pip install -e .
```

## Usage

To transcribe an audio file:

```zsh
wtm test.wav
```

To use the library in a Python script:

```python
>>> from whisper_turbo import transcribe
>>> transcribe('test.wav', any_lang=True)
```

## Acknowledgements

This project builds upon the reference [MLX implementation](https://github.com/ml-explore/mlx-examples/tree/main/whisper) of the Whisper model. Great thanks to the contributors of the MLX project for their exceptional work and inspiration.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.