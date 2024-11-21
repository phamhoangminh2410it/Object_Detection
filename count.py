import cv2
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors

class ObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.classwise_counts = {}
        self.region_initialized = False

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]

    def count_objects(self, track_line, box, track_id, prev_position, cls):
        if prev_position is None or track_id in self.counted_ids:
            return

        centroid = self.r_s.centroid
        dx = (box[0] - prev_position[0]) * (centroid.x - prev_position[0])
        dy = (box[1] - prev_position[1]) * (centroid.y - prev_position[1])

        if len(self.region) >= 3 and self.r_s.contains(self.Point(track_line[-1])):
            self.counted_ids.append(track_id)
            if dx > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1

        elif len(self.region) < 3 and self.LineString([prev_position, box[:2]]).intersects(self.r_s):
            self.counted_ids.append(track_id)
            if dx > 0 and dy > 0:
                self.in_count += 1
                self.classwise_counts[self.names[cls]]["IN"] += 1
            else:
                self.out_count += 1
                self.classwise_counts[self.names[cls]]["OUT"] += 1

    def store_classwise_counts(self, cls):
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, im0):
        labels_dict = {
        str.capitalize(key): str(value['IN'] + value['OUT']) if (self.show_in or self.show_out) else ''
        for key, value in self.classwise_counts.items()
        if value["IN"] != 0 or value["OUT"] != 0
    }

        if labels_dict:
            self.annotator.display_analytics(im0, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def count(self, im0):
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
            self.store_tracking_history(track_id, box)
            self.store_classwise_counts(cls)

            self.annotator.draw_centroid_and_tracks(
                self.track_line, color=colors(int(cls), True), track_thickness=self.line_width
            )

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_line, box, track_id, prev_position, cls)

        self.display_counts(im0)
        self.display_output(im0)

        return im0

cap = cv2.VideoCapture("VoThiSau.asf")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(20, 380), (1080, 380)]

video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = ObjectCounter(
    show=True,
    region=line_points,
    model="best8n.pt",
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = counter.count(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()