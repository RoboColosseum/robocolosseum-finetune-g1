import base64
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

from g1_deploy.multi_policy import UniversalBackend
from g1_deploy.policy_server import decode_observation


class FullStereoTests(unittest.TestCase):
    def test_full_head_transport_and_training_resize(self):
        head = np.random.default_rng(42).integers(0, 256, (480, 1280, 3), dtype=np.uint8)
        wrist = head[:, :640].copy()
        payload = dict(state=[0] * 29, task='carrying box', images={
            k: base64.b64encode(cv2.imencode('.png', v[:, :, ::-1])[1]).decode()
            for k, v in [('head', head), ('left_wrist', wrist), ('right_wrist', wrist)]})
        obs = decode_observation(payload, 'stereo_full')
        np.testing.assert_array_equal(obs['observation.images.head'], head)
        backend = UniversalBackend.__new__(UniversalBackend)
        backend.spec = {'family': 'groot'}
        backend.head_mode = 'stereo_full'
        backend.torch = torch
        actual = backend.image_tensor(obs['observation.images.head'])
        expected = torch.from_numpy(cv2.resize(head, (640, 480), interpolation=cv2.INTER_LINEAR)).permute(2, 0, 1).float() / 255
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(backend.image_tensor(wrist).shape), (3, 480, 640))
        self.assertEqual(obs['observation.images.head'].shape, (480, 1280, 3))
        backend.spec = {'family': 'pi05'}
        self.assertEqual(tuple(backend.image_tensor(head).shape), (3, 480, 1280))


if __name__ == '__main__':
    unittest.main()
