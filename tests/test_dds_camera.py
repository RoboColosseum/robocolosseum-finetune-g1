import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from g1_deploy.dds_camera import FrameBuffer, decode_image
from g1_deploy.wbc_client import CameraReader


def message(width=640, stamp=1):
    image = np.zeros((480, width, 3), np.uint8)
    image[:] = (0, 0, 255)
    if width == 1280:
        image[:, 640:] = (255, 0, 0)
    return SimpleNamespace(data=cv2.imencode('.png', image)[1].tobytes(), format='png',
                           header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nanosec=0)))


class CameraTests(unittest.TestCase):
    def test_rgb_and_model_head_semantics(self):
        data = message(1280).data
        left = decode_image(data, 'head', {'head_mode': 'left_eye'})
        np.testing.assert_array_equal(left[100, 500], [255, 0, 0])
        stereo = decode_image(data, 'head', {'head_mode': 'stereo_squash'})
        self.assertEqual(stereo.shape, (480, 640, 3))
        np.testing.assert_array_equal(stereo[100, 500], [0, 0, 255])
        self.assertEqual(decode_image(data, 'head', {'head_mode': 'stereo_full'}).shape, (480, 1280, 3))
        with self.assertRaises(ValueError):
            decode_image(message().data, 'head', {'head_mode': 'stereo_squash'})
        with self.assertRaises(ValueError):
            decode_image(data, 'left_wrist', {})
        with self.assertRaises(ValueError):
            decode_image(b'broken', 'head', {})

    def test_missing_stale_skew_and_oldest_timestamp(self):
        buf = FrameBuffer({'max_skew_s': 0.15})
        msg = message()
        with patch('g1_deploy.dds_camera.time.monotonic', return_value=10):
            buf.update('head', msg, 9.99)
            with self.assertRaisesRegex(RuntimeError, 'left_wrist'):
                buf.snapshot(0.5)
            buf.update('left_wrist', msg, 9.0)
            buf.update('right_wrist', msg, 9.98)
            with self.assertRaisesRegex(RuntimeError, 'left_wrist'):
                buf.snapshot(0.5)
            with self.assertRaisesRegex(RuntimeError, 'skew'):
                buf.snapshot(2)
            buf.update('left_wrist', message(stamp=2), 9.95)
            images, stamp = buf.snapshot(0.5)
            self.assertEqual(stamp, 9.95)
            images['head'][:] = 0
            self.assertTrue(buf.frames['head'][0].any())
            buf.update('left_wrist', message(stamp=2), 10)
            self.assertEqual(buf.frames['left_wrist'][1], 9.95)

    def test_bad_frame_fails_until_recovered(self):
        buf = FrameBuffer({})
        for name in ('head', 'left_wrist', 'right_wrist'):
            buf.update(name, message())
        bad = message(stamp=2)
        bad.data = b'bad'
        buf.update('head', bad)
        with self.assertRaisesRegex(RuntimeError, 'decode'):
            buf.snapshot(1)
        buf.update('head', message(stamp=3))
        buf.snapshot(1)

    def test_backend_selection(self):
        with patch('g1_deploy.dds_camera.DDSCameraReader') as reader:
            self.assertIs(CameraReader({'backend': 'dds'}), reader.return_value)
        with self.assertRaises(ValueError):
            CameraReader({'backend': 'typo'})


if __name__ == '__main__':
    unittest.main()
