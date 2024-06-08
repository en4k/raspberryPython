import io
import logging
import socketserver
from threading import Condition, Thread
from http import server
import cv2
import numpy as np
import time
import os
import glob

# 로그 파일과 녹화된 영상의 저장 경로
BASE_DIR = '/home/mk/Desktop/CAM'
LOG_FILE = os.path.join(BASE_DIR, 'motion_log.txt')

PAGE = """\
<html>
<head>
<title>RASPI HOME CAM SYSTEM</title>
<style>
body {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
}
h1 {
    margin-top: 0;
}
.container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
}
img {
    display: block;
    margin: 20px auto;
}
button {
    margin: 20px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}
</style>
</head>
<body>
<div class="container">
<h1>RASPI HOME CAM SYSTEM</h1>
<img src="stream.mjpeg" width="640" height="480" />
<button onclick="window.location.href='/log'">Go To Check Log</button>
</div>
</body>
</html>
"""

LOG_PAGE_TEMPLATE = """\
<html>
<head>
<title>Log Page</title>
<style>
body {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
}}
h1 {{
    margin-top: 0;
}}
.container {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}
button {{
    margin: 10px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
}}
</style>
<script>
function clearLog() {{
    if (confirm('Are you sure you want to clear the log?')) {{
        fetch('/clear_log', {{
            method: 'POST'
        }}).then(response => {{
            if (response.ok) {{
                window.location.href = '/log';
            }}
        }});
    }}
}}
</script>
</head>
<body>
<div class="container">
<h1>Motion Log</h1>
{log_entries}
<button onclick="window.location.href='/'">Return</button>
<button onclick="clearLog()">Clear Log</button>
</div>
</body>
</html>
"""


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"Handling GET request for path: {self.path}")
        if self.path == '/':
            print("Redirecting to /index.html")
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpeg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header(
                'Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s',
                                self.client_address, str(e))
        elif self.path == '/log':
            print("Serving log page")
            with open(LOG_FILE, 'r') as f:
                log_entries = ''.join(f'<p>{line}</p>' for line in f)
            content = LOG_PAGE_TEMPLATE.format(
                log_entries=log_entries).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            print(f"Path {self.path} not found")
            self.send_error(404)
            self.end_headers()

    def do_POST(self):
        print(f"Handling POST request for path: {self.path}")
        if self.path == '/clear_log':
            try:
                print("Clearing log file and deleting video files")
                # 로그 파일 초기화
                with open(LOG_FILE, 'w') as f:
                    pass
                print(f"Log file {LOG_FILE} cleared.")

                # 저장된 모든 영상 파일 삭제
                video_files = glob.glob(os.path.join(BASE_DIR, '*.avi'))
                if not video_files:
                    print("No video files found.")
                for video_file in video_files:
                    os.remove(video_file)
                    print(f"Video file {video_file} deleted.")

                self.send_response(200)
                self.end_headers()
            except Exception as e:
                print(f"Error clearing log and video files: {e}")
                self.send_error(500)
                self.end_headers()
        else:
            print(f"Path {self.path} not found")
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def capture_frames(output):
    thresh = 25
    max_diff = 5
    a, b, c = None, None, None
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    is_recording = False
    motion_start_time = None
    motion_end_time = None
    out = None

    if cap.isOpened():
        ret, a = cap.read()
        ret, b = cap.read()
        while ret:
            ret, c = cap.read()
            draw = c.copy()
            if not ret:
                break

            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            diff1 = cv2.absdiff(a_gray, b_gray)
            diff2 = cv2.absdiff(b_gray, c_gray)

            ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
            ret, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)

            diff = cv2.bitwise_and(diff1_t, diff2_t)

            k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

            diff_cnt = cv2.countNonZero(diff)
            if diff_cnt > max_diff:
                nzero = np.nonzero(diff)
                cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])),
                              (max(nzero[1]), max(nzero[0])), (0, 255, 0), 2)

                cv2.putText(draw, "Motion detected!!", (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                if not is_recording:
                    motion_start_time = time.time()
                    is_recording = True
                    filename = os.path.join(
                        BASE_DIR, time.strftime("%Y%m%d-%H%M%S") + '.avi')
                    out = cv2.VideoWriter(
                        filename, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (640, 480))
                    print(f"Started Recording: {filename}")

                if out is not None:
                    out.write(draw)

                motion_end_time = None
            else:
                if is_recording:
                    if motion_end_time is None:
                        motion_end_time = time.time()
                    elif time.time() - motion_end_time > 1:
                        is_recording = False
                        if out is not None:
                            out.release()
                            out = None
                        with open(LOG_FILE, 'a') as log_file:
                            log_file.write(f"{filename}\n")
                        print(f"Stopped Recording: {filename}")

            ret, jpeg = cv2.imencode('.jpg', draw)
            if ret:
                output.write(jpeg.tobytes())

            a = b
            b = c

            if cv2.waitKey(1) & 0xFF == 27:
                break

    if out is not None:
        out.release()
    cap.release()


if __name__ == '__main__':
    # 디렉토리 존재 여부 확인 및 생성
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    output = StreamingOutput()
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)

    thread = Thread(target=capture_frames, args=(output,))
    thread.daemon = True
    thread.start()

    try:
        server.serve_forever()
    finally:
        pass
