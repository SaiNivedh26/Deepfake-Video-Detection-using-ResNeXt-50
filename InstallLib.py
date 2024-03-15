import os
req_modules = [
    "pip install opencv-python",
    "pip install torch",
    "pip install openvino",
    "pip install onnx2onnx",
    "pip install scikit-learn-intelex",
    "pip install pandas",
    "pip install numpy"
]
for modules in req_modules:
    os.system(modules)