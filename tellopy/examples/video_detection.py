import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time

from dd_client import DD
from PIL import Image
import cStringIO
import base64
import math

def main():
    drone = tellopy.Tello()
    dd = DD("eris", 18104, 0, path = "/api/private")
    dd.set_return_format(0)

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        # skip first 300 frames
        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                pilImage = frame.to_image()
                image = cv2.cvtColor(numpy.array(pilImage), cv2.COLOR_RGB2BGR)

                # resize image for detection
                basewidth = 300
                wpercent = (basewidth/float(pilImage.size[0]))
                hsize = int((float(pilImage.size[1])*float(wpercent)))
                detectionImage = pilImage.resize((basewidth,hsize), Image.ANTIALIAS)


                # Create a base64 string from the image buffer
                buffer = cStringIO.StringIO()
                frame.to_image().save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue())
                data = [img_str]

                parameters_input = {}
                parameters_mllib = {"gpu": True}
                parameters_output = {"bbox": True, "confidence_threshold":0.3}

                # Make predict request on a service
                detection = dd.post_predict(
                    "detection_600",
                    data,
                    parameters_input,
                    parameters_mllib,
                    parameters_output
                )

                # Get boxes from results
                list_bbox = detection["body"]["predictions"][0]["classes"]
                bbox_list = []

                for elt in list_bbox:

                    xmin = int(elt["bbox"]["xmin"])
                    xmax = int(math.ceil(elt["bbox"]["xmax"]))
                    ymin = int(elt["bbox"]["ymax"])
                    ymax = int(math.ceil(elt["bbox"]["ymin"]))
                    cv2.rectangle(image, (xmin, ymax), (xmax, ymin),(255,0,0),2)

                cv2.imshow('Original', image)
                cv2.waitKey(1)
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
