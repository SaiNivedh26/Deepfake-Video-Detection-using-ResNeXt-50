# Deepfake Video Detection using ResNeXt-50
<center><h2>Deep Fake Shield</h2></center>

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


Utilizing the resources provided by Intel Developer Cloud significantly expedited our AI model development and deployment processes. Specifically, we harnessed the power of Intel's CPU and XPU to accelerate two critical components of our project: Human Detection and Text-to-Outfit Generation. 💻⚡

1.  **Human Detection Model Training:** The Intel Developer Cloud's CPU and XPU capabilities, combined with the use of oneDNN, played a pivotal role in reducing the training time of our Human Detection model. By leveraging the high-performance computing infrastructure provided by Intel, we were able to train our model more efficiently, significantly cutting down the time required for model optimization and experimentation.🚀🔧 <br/> <br/> The integration of oneDNN, a high-performance deep learning library developed by Intel, contributed to this efficiency by optimizing the computational tasks involved in training. Notably, a single epoch now takes only 2 seconds, a substantial improvement compared to the 6 seconds it took in Colab, showcasing the remarkable speedup achieved through the use of Intel's hardware resources and optimized software stack. 🚀⚒️ <br/> <br/> Additionally, the optimized version of TensorFlow tailored for Intel architectures further played a crucial role in reducing the training time. This collaborative utilization of optimized TensorFlow and Intel's advanced computing infrastructure enabled us to achieve significant improvements in model training efficiency, ultimately accelerating our development process and enhancing the overall performance of our Human Detection model. 🏋️‍♂️🧑‍💻

![Comparison Graph](images/Binary_Classifcation_Graph.png)

>Comparison between time took in Intel Developers Cloud using OneDNN and Google Colab
    
2.  **Text-to-Outfit Generation:** The Text-to-Outfit Generator component of our project involved complex computational tasks, particularly during outfit generation and rendering. Running these computations in Google Colab often resulted in long processing times due to resource limitations. However, by leveraging Intel Developer Cloud's CPU and XPU resources, we experienced a notable reduction in processing time. The parallel processing capabilities of Intel's infrastructure enabled us to generate outfit recommendations swiftly, enhancing the overall user experience. 🌟👗

![Comparison Graph](images/textToImageComparison.png)

>Comparison between time took in Intel Developers Cloud using OneDNN and Google Colab
    
In summary, Intel Developer Cloud's advanced CPU and XPU technologies provided us with the computational power necessary to expedite model training and inference processes, ultimately accelerating our project development and deployment timelines. 🚀🕒

# Flow Diagram 🔄📊
still to do


## Requirements
- Python 3.x
- PyTorch
- torch-vision
- Timm
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
  npm install
```

Start the server

```bash
  npm run start
```
# How We Built It 🛠️👷‍♂️

 -  Developed frontend using React for a modular and reusable UI. 💻🔧
 -  Implemented backend with Flask for RESTful APIs and data processing. 🐍🚀
 -  Integrated various machine learning models for outfit recommendation, virtual try-on, and fashion chatbot functionalities. 🤖⚙️
 -  Implemented virtual try-on feature with complex image processing and machine learning techniques. 📷🔄
 -  Integrated a fashion chatbot leveraging natural language processing (NLP) capabilities. 💬🤖

# References For Datasets 📊📚

 - Virtual-Try-On : [VITON 🤖👗](https://www.kaggle.com/datasets/marquis03/hr-viton)
 - Chat-Bot : [PDF 📄💬](https://github.com/dhaan-ish/intelOneApiHackathon/blob/main/Chat-Bot/Data/fashsion.pdf)
 - Outfit-Recommendation : [Kaggle 🛍️📸](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)]
 - Human-Detection : [Roboflow 👤🔍](https://universe.roboflow.com/human-classification/human-qgzuc)


## Authors

- [@Hari Heman](https://github.com/MAD-MAN-HEMAN)
- [@Sai Nivedh V](https://github.com/SaiNivedh26)
- [@Baranidharan S](https://github.com/thespectacular314)
- [@Roshan T](https://github.com/Twinn-github09)

## License

This project is licensed under the [MIT License](LICENSE).

