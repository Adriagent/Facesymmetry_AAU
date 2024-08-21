python -m PyInstaller --hidden-import=comtypes.stream main_app.py --noconsole --workpath build/workdir --specpath build/specdir --distpath build ^
    --add-data "../../models/face_landmarker.task;models/." ^
    --add-data "../../models/efficientdet_lite0_32.tflite;models/." ^
    --add-data "../../config/default_config.json;config/." ^
    --add-data "../../config/saved_config.json;config/." ^
    --add-data "../../media/background2.png;media/." 
