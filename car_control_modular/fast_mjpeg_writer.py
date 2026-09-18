"""Recorder-only JPEG encoding + packet-copy AVI muxing, no camera access.

OpenCV's imencode uses the installed JPEG encoder; FFmpeg only muxes those
packets (no decode or second encode). Blocking I/O stays in the video worker.
"""
import shutil
import subprocess
import tempfile


class FastMjpegWriter:
    backend_name = "jpeg_packet_copy"

    def __init__(self, path, fps, size, cv2_module, quality=90):
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is unavailable")
        self._cv2 = cv2_module
        self._size = tuple(size)
        self._quality = max(1, min(100, int(quality)))
        self._closed = False
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen([
                executable, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-threads", "1", "-probesize", "32", "-analyzeduration", "0",
                "-f", "image2pipe", "-framerate", str(float(fps)),
                "-vcodec", "mjpeg", "-i", "pipe:0", "-map", "0:v:0",
                "-c:v", "copy", "-f", "avi", "-y", str(path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._stderr, bufsize=0, close_fds=True)
        except Exception:
            self._stderr.close()
            raise

    def isOpened(self):
        return not self._closed and self._process.poll() is None

    def write(self, image):
        if not self.isOpened():
            raise RuntimeError("JPEG muxer stopped")
        if (image.shape[1], image.shape[0]) != self._size:
            raise ValueError("video frame size changed")
        ok, packet = self._cv2.imencode(
            ".jpg", image, [self._cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        data = memoryview(packet).cast("B")
        while data:
            count = self._process.stdin.write(data)
            if not count:
                raise RuntimeError("JPEG muxer pipe closed")
            data = data[count:]

    def release(self):
        if self._closed:
            return
        self._closed = True
        error = None
        try:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self._process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1.)
                raise RuntimeError("JPEG muxer close timed out")
            if code:
                self._stderr.seek(0)
                error = self._stderr.read(2048).decode("utf-8", errors="replace")
        finally:
            self._stderr.close()
        if error is not None:
            raise RuntimeError("JPEG muxer failed: " + error)
