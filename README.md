# yolo-detector-gradio

This is a simple object detection web demo built with Gradio as frontend and YOLOv3 as the detection model. It allows you to upload an image and detect objects in it using YOLOv3.

## Demo

You can try out the demo at:
 - Original YOLOv3 weights - https://yolov3.solsyn.dev
 - Trained garbage detector - https://garbage-det.solsyn.dev

## Local Deployment

To deploy this app on your local environment, follow these steps:

1. Clone this repository:
```shell
git clone https://github.com/SoloSynth1/yolo-detector-gradio.git
cd yolo-detector-gradio
```
2. Install the required packages using virtualenv:
```shell
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download the YOLOv3 weights and configuration files (already included in this repo as LFS objects):

```shell
git lfs pull
```

4. Run the app:

```shell
python main.py
```

5. Open your browser and go to http://localhost:7860.

## Docker Deployment

To deploy this app using Docker, follow these steps:

1. Clone this repository:

```shell
git clone https://github.com/SoloSynth1/yolo-detector-gradio.git
cd yolo-detector-gradio
```

2. Download the YOLOv3 weights and configuration files (already included in this repo as LFS objects):

```shell
git lfs pull
```

3. Build the Docker image:
```shell
docker build -t yolo-detector-gradio .
```

4. Run the Docker container:

```shell
docker run --rm -p 7860:7860 yolo-detector-gradio
```

5. Open your browser and go to http://localhost:7860.

## Acknowledgements

This app uses the following open-source projects:

- Gradio (https://github.com/gradio-app/gradio)
- YOLOv3 (https://github.com/pjreddie/darknet)
- OpenCV (https://github.com/opencv/opencv-python)
- NumPy (https://github.com/numpy/numpy)
- And many more!

## License

This app is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
