# main.py
from camera_source import get_cameras
from camera_manager import poll_loop

def main():
    config = {}

    poll_loop(
        get_cameras=get_cameras,
        config=config,
        interval=5
    )

if __name__ == "__main__":
    main()
