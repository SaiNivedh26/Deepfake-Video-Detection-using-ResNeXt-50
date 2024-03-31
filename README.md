# Deepfake Video Detection using ResNeXt-50
<center><h2>Project Title : Deep Fake Shield (DFS) </h2></center>

## Demo
For the demonstration purpose, we're going to Import a fake video and check whether our application detects it

https://github.com/SaiNivedh26/Team-Nooglers/assets/142657686/552961d7-8e3e-4f0c-8da7-3b2a38d40463

## Overview
This project aims to detect deepfake videos using advanced deep learning techniques, specifically leveraging the ResNeXt-50 model architecture. Deepfake videos, which are highly realistic but fabricated using AI algorithms, pose a significant threat to the authenticity of visual media content. By developing a robust framework for classifying videos as real or fake, we contribute to the ongoing efforts to combat the proliferation of misinformation and preserve the integrity of digital media platforms.

## Features

- Utilizes the Celeb-DF-v2 dataset, comprising a 6000+ diverse collection of real and synthesized videos featuring celebrity personas.
- Implements the ResNeXt-50 model architecture for video classification.
- Provides functionality for preprocessing videos, extracting frames, and applying data augmentation techniques.
- Offers inference capabilities for detecting deepfake videos in real time.
- Includes example scripts for training the model and evaluating its performance.
# Usage of Intel Developer Cloud 🌐💻


Leveraging the powerful resources provided by Intel Developer Cloud accelerated our development and deployment of the deepfake video detection model. We harnessed the computational capabilities of Intel's CPUs and XPUs to optimize critical components, such as video preprocessing, frame extraction, and model inference.

By taking advantage of Intel's high-performance computing infrastructure and optimized software libraries (e.g., oneDNN, Intel Distribution of OpenVINO), we significantly reduced the time required for data preprocessing, model training, and inference. This allowed for faster experimentation, iterative improvements, and ultimately, a more efficient deployment of our deepfake detection solution.

# Flow Diagram 🔄📊

![flow](https://github.com/SaiNivedh26/Team-Nooglers/assets/142657686/f92a8b99-ff6f-4e03-877b-151ea067c5c7)


## Necessary Libraries

To run this project, you'll need the following libraries:

- Python 3.x
- PyTorch
- torchvision
- timm
- OpenCV
- PIL

## Contributing
  Contributions are welcome! Feel free to submit bug reports, feature requests, or pull requests to help improve this project. 

## Run Locally

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


## Authors

- [@Hari Heman](https://github.com/MAD-MAN-HEMAN)
- [@Sai Nivedh V](https://github.com/SaiNivedh26)
- [@Baranidharan S](https://github.com/thespectacular314)
- [@Roshan T](https://github.com/Twinn-github09)

## License

This project is licensed under the [MIT License](LICENSE).

