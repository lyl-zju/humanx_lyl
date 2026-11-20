import subprocess
import sys
import textwrap
import unittest


class HOIGravityTerminationTests(unittest.TestCase):
    def test_reference_tilt_allows_bends_and_yaw_but_resets_large_errors(self):
        # Isaac Gym must load before Torch, which other tests may have imported.
        script = textwrap.dedent(
            """
            import isaacgym
            import math
            from types import SimpleNamespace

            import torch
            from isaac_utils.rotations import quat_from_euler_xyz, quat_rotate_inverse
            from humanoidverse.envs.humanx.hoi import LeggedRobotMotionTrackingHOI

            env = LeggedRobotMotionTrackingHOI.__new__(LeggedRobotMotionTrackingHOI)
            env.config = SimpleNamespace(
                termination=SimpleNamespace(
                    terminate_by_contact=False,
                    terminate_by_gravity=True,
                    terminate_by_low_height=False,
                    terminate_when_close_to_dof_pos_limit=False,
                    terminate_when_close_to_dof_vel_limit=False,
                    terminate_when_close_to_torque_limit=False,
                ),
                termination_scales=SimpleNamespace(
                    termination_gravity_x=0.8, termination_gravity_y=0.8,
                ),
            )
            zeros = torch.zeros(4)
            actual_pitch = torch.full((4,), 80.0).deg2rad()
            reference_pitch = torch.tensor([80.0, 80.0, 0.0, 30.0]).deg2rad()
            reference_yaw = torch.tensor([0.0, 120.0, 0.0, 0.0]).deg2rad()
            actual_root = quat_from_euler_xyz(zeros, actual_pitch, zeros)
            env.ref_root_rot = quat_from_euler_xyz(zeros, reference_pitch, reference_yaw)
            env.gravity_vec = torch.tensor([0.0, 0.0, -1.0]).repeat(4, 1)
            env.projected_gravity = quat_rotate_inverse(actual_root, env.gravity_vec, w_last=True)
            env.reset_buf = torch.zeros(4, dtype=torch.long)
            env.time_out_buf = torch.zeros(4, dtype=torch.bool)
            env.log_dict = {}

            env._gravity_reference_cos_threshold = math.cos(math.radians(60.0))
            env._update_reset_buf()
            assert env.reset_buf.tolist() == [0, 0, 1, 0]
            assert not env.time_out_buf.any()
            assert env.log_dict["term_first_gravity_frac"].item() == 0.25

            env._gravity_reference_cos_threshold = math.cos(math.radians(45.0))
            assert env._gravity_termination_mask().tolist() == [False, False, True, True]

            env._gravity_reference_cos_threshold = None
            assert env._gravity_termination_mask().tolist() == [True, True, True, True]
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
