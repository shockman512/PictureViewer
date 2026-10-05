# Picture Viewer

A fast Windows picture viewer built with Python and PySide6.

- Big picture on top, scrolling thumbnail strip with a slider at the bottom
- Zoom (mouse wheel, +/-, Fit / 100%) and drag to pan
- Arrow-key navigation, fullscreen, rotate, delete to Recycle Bin, Open With
- Slideshow with 1-10 second delay and random order
- JPG, PNG, GIF, BMP, WebP, TIFF, HEIC and RAW
- Open a folder, drag and drop one, or launch with a file/folder path
- Face recognition (offline): scan a folder, name people, ignore false faces, search by name

## Faces

Uses OpenCV YuNet (detection) and SFace (recognition). All data stays on your PC in
`%LOCALAPPDATA%\PictureViewer\faces.db`. Menu: People > Scan this folder, Show face boxes (F2),
People..., Find person....

Download the two models into a `models` folder next to `picviewer.py` (they are not in the repo):

    curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    curl -L -o models/face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

Without the models or OpenCV the viewer still works; the People menu is disabled.

## Run

    pip install PySide6 pillow pillow-heif rawpy send2trash numpy opencv-python
    python picviewer.py [folder-or-picture]

## Build

    pip install pyinstaller
    pyinstaller --onedir --windowed --name PictureViewer --collect-all pillow_heif --collect-all rawpy --add-data "models;models" --hidden-import faces --hidden-import facesui picviewer.py

`--onedir` (a folder) triggers antivirus false positives far less than `--onefile`.
