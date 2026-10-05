# Picture Viewer

A fast Windows picture viewer built with Python and PySide6.

- Big picture on top, scrolling thumbnail strip with a slider at the bottom
- Zoom (mouse wheel, +/-, Fit / 100%) and drag to pan
- Arrow-key navigation, fullscreen, rotate, delete to Recycle Bin, Open With
- Slideshow with 1-10 second delay and random order
- JPG, PNG, GIF, BMP, WebP, TIFF, HEIC and RAW
- Open a folder, drag and drop one, or launch with a file/folder path

## Run

    pip install PySide6 pillow pillow-heif rawpy send2trash
    python picviewer.py [folder-or-picture]

## Build the exe

    pip install pyinstaller
    pyinstaller --onefile --windowed --name PictureViewer --collect-all pillow_heif --collect-all rawpy picviewer.py
