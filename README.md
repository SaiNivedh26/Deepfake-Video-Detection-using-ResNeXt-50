# Deepfake Video Detection using ResNeXt-50
<center><h2>Project Title : Deep Fake Shield (DFS) </h2></center>

## 🚀 Demo
For the demonstration purpose, we're going to Import a fake video and check whether our application detects it. This is deepfake video of ‘The Shining,’ starring Jim Carrey
[Link 🔗] (https://www.marketwatch.com/story/watch-this-deepfake-and-you-may-never-trust-a-video-recording-ever-again-2019-07-09)


We have obtained accelerated version of the model after using [```Intel® oneAPI Deep Neural Network Library```](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html#gs.6pwnat)

https://github.com/SaiNivedh26/Team-Nooglers/assets/142657686/1fc386d2-a830-4168-ac6d-3d9c5b706981


## Overview
This project aims to detect deepfake videos using advanced deep learning techniques, specifically leveraging the ResNeXt-50 model architecture. Deepfake videos, which are highly realistic but fabricated using AI algorithms, pose a significant threat to the authenticity of visual media content. By developing a robust framework for classifying videos as real or fake, we contribute to the ongoing efforts to combat the proliferation of misinformation and preserve the integrity of digital media platforms.
![mindmap](https://github.com/SaiNivedh26/Team-Nooglers/assets/88413854/0a65a4d2-c3a3-4af5-a4ca-29c2919523d2)


## 🧐 Features

- Utilizes the Celeb-DF-v2 dataset, comprising a 6000+ diverse collection of real and synthesized videos featuring celebrity personas.
- Implements the ResNeXt-50 model architecture for video classification.
- Provides functionality for preprocessing videos, extracting frames, and applying data augmentation techniques.
- Offers inference capabilities for detecting deepfake videos in real time.
- Includes example scripts for training the model and evaluating its performance.
# Usage of Intel Developer Cloud 🌐💻


Leveraging the powerful resources provided by Intel Developer Cloud accelerated our development and deployment of the deepfake video detection model. We harnessed the computational capabilities of Intel's CPUs and XPUs to optimize critical components, such as video preprocessing, frame extraction, and model inference.

By taking advantage of Intel's high-performance computing infrastructure and optimized software libraries (e.g., oneDNN, Intel Distribution of OpenVINO), we significantly reduced the time required for data preprocessing, model training, and inference. This allowed for faster experimentation, iterative improvements, and ultimately, a more efficient deployment of our deepfake detection solution.

# Flow Diagram 🔄📊

![flow](https://github.com/SaiNivedh26/Team-Nooglers/assets/142657686/feec05e7-1a40-47c5-845c-ddf4da46967e)


## Necessary Libraries

To run this project, you'll need the following libraries:

- Python 3.x
- oneDNN (Intel's Deep neural Network Library)
- PyTorch
- torchvision
- timm
- OpenCV
- PIL

## Installation procedure for OneDNN 📦⬇️
Download oneDNN source code or clone the repository.

```bash
git clone https://github.com/oneapi-src/oneDNN.git
```

Build the Library
>Ensure that all software dependencies are in place and have at least the minimal supported version.

 > * CMAKE_INSTALL_PREFIX to control the library installation location
 > * CMAKE_BUILD_TYPE to select between build type (Release, Debug, RelWithDebInfo).



Linux/macOs
> Generate makefile:
```bash
  mkdir -p build && cd build && cmake ..
```
> Build the library:
```bash
  make -j
```
> Build the doumentation:
```bash
 make doc
```
> Install the library,headers,and documentations:
```bash
  make install
```

Windows
> Generate a Microsoft Visual Studio solution:
```bash
  mkdir build && cd build && cmake -G "Visual Studio 15 2017 Win64" ..
```
> For the solution to use the Intel C++ Compiler, select the corresponding toolchain using the cmake -T switch:
```bash
 cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 19.0" ..
```
> Build the library:
```bash
 cmake --build .
```
> You can also use the msbuild command-line tool directly (here /p:Configuration selects the build configuration which can be different from the one specified in CMAKE_BUILD_TYPE, and /m enables a parallel build):
```bash
  msbuild "oneDNN.sln" /p:Configuration=Release /m
```
> Build the documentation
```bash
  cmake --build . --target DOC
```
> Install the library, headers, and documentation:
```bash
  cmake --build . --target INSTALL
```
Validate the Build
> Run unit tests:
```bash
  ctest
```

## Contributing 🤝
  Contributions are welcome! Feel free to submit bug reports, feature requests, or pull requests to help improve this project. 

## 🛠️ Run Locally

Clone the project

```bash
  git clone https://github.com/SaiNivedh26/Team-Nooglers
```

Go to the project directory

```bash
  cd Team-Nooglers
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  npm run start
```
# How We Built It 🛠️👷‍♂️

- Developed a custom data pipeline for loading and preprocessing the Celeb-DF-v2 dataset. 📂
- Implemented the ResNeXt-50 model architecture using PyTorch for video classification. 🔥
- Utilized transfer learning techniques by fine-tuning the model on the Celeb-DF-v2 dataset. 🏋️‍♀️
- Leveraged Intel Developer Cloud's powerful computing resources and optimized libraries (e.g., oneDNN) to accelerate model training and inference. ⚡
# References For Datasets 📊📚

- <h2>Celeb-DF-v2 dataset</h2> [Drive 🔗] (https://www.google.com/url?q=https://drive.google.com/open?id%3D1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj&sa=D&source=editors&ust=1711031793764133&usg=AOvVaw176zn3G8Ep0EDWpMV-rWnQ)

# Model performance 🕝 ⚡
<h2>This is the comparison of regular code and the same code utlized for an 20s , 1080 pixel Video. The Implementation of Intel® oneAPI Deep Neural Network Library accelerated the code by 2.8 - 3 times</h2>

![Time (seconds)](https://github.com/SaiNivedh26/Team-Nooglers/assets/142657686/f2a14d7a-f47a-42a8-8708-17015cb90060)


## Authors

- [@Hari Heman](https://github.com/MAD-MAN-HEMAN)
- [@Sai Nivedh V](https://github.com/SaiNivedh26)
- [@Baranidharan S](https://github.com/thespectacular314)
- [@Roshan T](https://github.com/Twinn-github09)

## 🛡️License

This project is licensed under the [MIT License](LICENSE).

