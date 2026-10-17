"""One interface over the policy families used in the graded sweep:
    adapter.new_episode(instruction, seed)
    adapter.chunk(obs) -> list of 7-D env actions (one forward pass; chunked policies return several)
OFT and OpenVLA must be run from their repo roots (experiments.robot.*); Octo needs its calibration file from plus_5."""
import json

import numpy as np


class OFTAdapter:
    name = "openvla_oft"

    def __init__(self, ckpt=None):
        from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
        from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
        from plus_common import OFT_CKPT, SUITE
        ckpt = ckpt or OFT_CKPT[SUITE]                    # one released checkpoint per suite
        self.cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name=SUITE, center_crop=True)
        set_seed_everywhere(self.cfg.seed)
        self.parts = initialize_model(self.cfg)
        self.resize = get_image_resize_size(self.cfg)
        self.instr = ""

    def new_episode(self, instruction, seed):
        self.instr = instruction

    def chunk(self, obs):
        from experiments.robot.libero.run_libero_eval import prepare_observation, process_action
        from experiments.robot.robot_utils import get_action
        model, action_head, proprio_projector, noisy_action_projector, processor = self.parts
        observation, _ = prepare_observation(obs, self.resize)
        acts = get_action(self.cfg, model, observation, self.instr, processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                          use_film=self.cfg.use_film)
        return [np.asarray(process_action(a, self.cfg.model_family), float) for a in acts]


class OpenVLAAdapter:
    name = "openvla"

    def __init__(self, ckpt="openvla/openvla-7b-finetuned-libero-10"):
        import inspect
        from experiments.robot.libero.libero_utils import get_libero_image
        from experiments.robot.libero.run_libero_eval import GenerateConfig
        from experiments.robot.openvla_utils import get_processor
        from experiments.robot.robot_utils import get_image_resize_size, get_model, set_seed_everywhere
        self.cfg = GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name="libero_10", center_crop=True)
        set_seed_everywhere(self.cfg.seed)
        self.model = get_model(self.cfg)
        self.cfg.unnorm_key = self.cfg.task_suite_name
        if self.cfg.unnorm_key not in self.model.norm_stats and f"{self.cfg.unnorm_key}_no_noops" in self.model.norm_stats:
            self.cfg.unnorm_key = f"{self.cfg.unnorm_key}_no_noops"
        self.processor = get_processor(self.cfg)
        self.resize = get_image_resize_size(self.cfg)
        self.img_takes_resize = len(inspect.signature(get_libero_image).parameters) >= 2
        self.instr = ""

    def new_episode(self, instruction, seed):
        self.instr = instruction

    def chunk(self, obs):
        from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
        from experiments.robot.robot_utils import get_action, invert_gripper_action, normalize_gripper_action
        img = get_libero_image(obs, self.resize) if self.img_takes_resize else get_libero_image(obs)
        observation = {"full_image": img,
                       "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                                obs["robot0_gripper_qpos"]))}
        a = get_action(self.cfg, self.model, observation, self.instr, processor=self.processor)
        a = invert_gripper_action(normalize_gripper_action(a, binarize=True))
        return [np.asarray(a, float)]


class OctoAdapter:
    name = "octo"

    def __init__(self, calib_file, step=30000):
        from eval_octo import load_model
        self.model, self.stats = load_model(step)
        cal = json.load(open(calib_file))
        self.flip, self.gmode = bool(cal["flip"]), cal["gmode"]
        self.task, self.rng = None, None

    def new_episode(self, instruction, seed):
        import jax
        self.task = self.model.create_tasks(texts=[instruction])
        self.rng = jax.random.PRNGKey(seed)

    def chunk(self, obs):
        import jax
        from PIL import Image
        from eval_octo import to_env_gripper
        img = obs["agentview_image"]
        img = img[::-1, ::-1] if self.flip else img
        img = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((256, 256)))
        o = {"image_primary": img[None, None], "timestep_pad_mask": np.ones((1, 1), dtype=bool)}
        self.rng, key = jax.random.split(self.rng)
        acts = np.asarray(self.model.sample_actions(o, self.task, unnormalization_statistics=self.stats, rng=key))[0]
        out = []
        for a in acts:
            a = np.asarray(a, dtype=np.float64).copy()
            a[6] = to_env_gripper(float(a[6]), self.gmode)
            out.append(a)
        return out


class LeRobotAdapter:
    """pi0.5 / pi0 through LeRobot's own pipeline (verified against lerobot 0.6.1 lerobot_eval.rollout): the raw
    robosuite observation is formatted exactly like envs/libero.LiberoEnv._format_raw_obs, then
    preprocess_observation -> env preprocessor (LiberoProcessorStep: 180 deg image flip, eef pos + axis-angle +
    gripper qpos state) -> policy preprocessor -> select_action -> postprocessor -> env postprocessor.
    select_action keeps its own action queue, so chunk() returns one action per call."""

    def __init__(self, policy_path):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        self.name = "pi05" if "pi05" in policy_path else ("pi0" if "pi0" in policy_path else "lerobot")
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = False
        env_cfg = LiberoEnv(task="libero_10")
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(self.policy.config.device)},
                                    "rename_observations_processor": {"rename_map": {}}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        self.torch = torch
        self.instr = ""

    def new_episode(self, instruction, seed):
        self.instr = instruction
        self.policy.reset()
        self.torch.manual_seed(seed)

    def chunk(self, obs):
        from lerobot.envs.utils import preprocess_observation
        b = lambda x: np.asarray(x, np.float64)[None]
        raw = {"pixels": {"image": np.ascontiguousarray(obs["agentview_image"])[None],
                          "image2": np.ascontiguousarray(obs["robot0_eye_in_hand_image"])[None]},
               "robot_state": {"eef": {"pos": b(obs["robot0_eef_pos"]), "quat": b(obs["robot0_eef_quat"]),
                                       "mat": np.eye(3)[None]},
                               "gripper": {"qpos": b(obs["robot0_gripper_qpos"]), "qvel": b(obs["robot0_gripper_qvel"])},
                               "joints": {"pos": b(obs["robot0_joint_pos"]), "vel": b(obs["robot0_joint_vel"])}}}
        o = preprocess_observation(raw)
        o["task"] = [self.instr]
        o = self.pre(self.env_pre(o))
        with self.torch.inference_mode():
            a = self.policy.select_action(o)
        a = self.post(a)
        a = self.env_post({"action": a})["action"]
        return [np.asarray(a.to("cpu").numpy(), float)[0]]


class RemoteAdapter:
    """A policy that lives in another process (policy_server.py): for machines where the simulator's pins and the
    policy's pins cannot share one Python. Same interface as every adapter."""
    KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "robot0_gripper_qvel", "robot0_joint_pos", "robot0_joint_vel")

    def __init__(self, name, port=6006):
        import os
        from multiprocessing.connection import Client
        self.name = name
        self.conn = Client(("127.0.0.1", int(os.environ.get("POLICY_PORT", port))), authkey=b"self-motion")

    def new_episode(self, instruction, seed):
        self.conn.send(("new", instruction, int(seed)))
        self.conn.recv()

    def chunk(self, obs):
        self.conn.send(("chunk", {k: np.ascontiguousarray(obs[k]) for k in self.KEYS}))
        return [np.asarray(a, float) for a in self.conn.recv()]


def make_adapter(family, octo_calib=""):
    if family == "smolvla":                 # served by policy_server.py --policy lerobot/smolvla_libero
        return RemoteAdapter("smolvla")
    if family == "pi05":
        return LeRobotAdapter("lerobot/pi05_libero_finetuned_v044")
    if family == "pi0":
        return LeRobotAdapter("lerobot/pi0_libero_finetuned_v044")
    if family == "oft":
        return OFTAdapter()
    if family == "openvla":
        return OpenVLAAdapter()
    if family == "octo":
        assert octo_calib, "--octo_calib is required for octo (run the plus_5 calibration cell first)"
        return OctoAdapter(octo_calib)
    raise ValueError(family)
