from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f.readlines()]

setup(
    name='whisper-turbo-mlx',
    url='https://github.com/JosefAlbers/whisper-turbo-mlx',
    py_modules=['whisper_turbo'],
    packages=find_packages(),
    version='0.0.2-alpha',
    readme="README.md",
    author_email="albersj66@gmail.com",
    description="Whisper Turbo in MLX",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Josef Albers",
    license="MIT",
    python_requires=">=3.12.3",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "wtm = whisper_turbo:fire_main",
        ],
    },
)
