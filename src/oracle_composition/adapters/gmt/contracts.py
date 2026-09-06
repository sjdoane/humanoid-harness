"""Pinned contracts for the admitted GMT G1 deployment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

GMT_UPSTREAM_COMMIT = "2a590de25a1eb08e47491977a738549c22f16e1f"
GMT_CHECKPOINT_SHA256 = "bf965da77571a5bb0dbb291b349b387a258c1b34cb86173cc7502f9ec2015755"
GMT_CHECKPOINT_SIZE = 8_291_216
GMT_ARCHIVE_PREFIX = "may2-run3-datav3-samerun2-airtime-distill-fixlr-40500-jit/"
GMT_ARCHIVE_MEMBER_COUNT = 114
GMT_ARCHIVE_UNCOMPRESSED_SIZE = 8_292_696
GMT_SUPPORT_FILE_SHA256 = {
    "LICENSE": "209a5f19d0d9f024d558a99de5995b458874956caaa3f03e3738744b876ddbde",
    "README.md": "8ae6728345c1605b335c754957004713b44a4ded9409bf6b1dc4a137fb86d52b",
    "sim2sim.py": "10e8cac2e2cd4895ee4b229db4a06aad7f467d1a14399355a7254d789b8e1b5f",
    "utils/motion_lib.py": "c54d8d9f543c6eaf0b9fcd04ad3c8c7447068eece00d83be40741904c1b2f5d1",
    "utils/torch_utils.py": "7bf1941c721cbfe58bb8d73d0cbe937a7f56a9b0db8976bce98e485677d405c1",
    "assets/robots/g1/g1.xml": ("7013cd256c89796b2844613d24dda2a13410df7cd32ddd4d59b1741f85094304"),
}
GMT_G1_MESH_NAMES = (
    "head_link.STL",
    "left_ankle_pitch_link.STL",
    "left_ankle_roll_link.STL",
    "left_elbow_link.STL",
    "left_hip_pitch_link.STL",
    "left_hip_roll_link.STL",
    "left_hip_yaw_link.STL",
    "left_knee_link.STL",
    "left_rubber_hand.STL",
    "left_shoulder_pitch_link.STL",
    "left_shoulder_roll_link.STL",
    "left_shoulder_yaw_link.STL",
    "left_wrist_pitch_link.STL",
    "left_wrist_roll_link.STL",
    "left_wrist_yaw_link.STL",
    "logo_link.STL",
    "pelvis.STL",
    "pelvis_contour_link.STL",
    "right_ankle_pitch_link.STL",
    "right_ankle_roll_link.STL",
    "right_elbow_link.STL",
    "right_hip_pitch_link.STL",
    "right_hip_roll_link.STL",
    "right_hip_yaw_link.STL",
    "right_knee_link.STL",
    "right_rubber_hand.STL",
    "right_shoulder_pitch_link.STL",
    "right_shoulder_roll_link.STL",
    "right_shoulder_yaw_link.STL",
    "right_wrist_pitch_link.STL",
    "right_wrist_roll_link.STL",
    "right_wrist_yaw_link.STL",
    "torso_link_rev_1_0.STL",
    "waist_roll_link_rev_1_0.STL",
    "waist_yaw_link_rev_1_0.STL",
)
GMT_G1_MESH_TREE_SHA256 = "d8366a1f0c1e64d47c3710dfe7fd01d136ee462fda77d3cb571ab7dcecc967f2"
GENERATED_OPERATOR_ALLOWLIST = (
    "ops.prim.NumToTensor",
    "torch._convolution",
    "torch.add",
    "torch.cat",
    "torch.clamp",
    "torch.div",
    "torch.flatten",
    "torch.layer_norm",
    "torch.linear",
    "torch.mul",
    "torch.permute",
    "torch.reshape",
    "torch.silu",
    "torch.size",
    "torch.slice",
    "torch.sub",
    "torch.to",
)

OBSERVATION_DIM = 2_154
ACTION_DIM = 23
REFERENCE_HORIZON = 20
REFERENCE_FRAME_DIM = 30
PROPRIOCEPTION_DIM = 74
NORMALIZER_EPSILON = 1.0e-4
LAYER_NORM_EPSILON = 1.0e-5
PARAMETER_COUNT = 2_052_271

REFERENCE_SLICE = slice(0, 600)
CURRENT_REFERENCE_SLICE = slice(0, 30)
CURRENT_PROPRIOCEPTION_SLICE = slice(600, 674)
PROPRIOCEPTION_HISTORY_SLICE = slice(674, 2_154)

CONTROL_DT_SECONDS = 0.02
SIMULATION_DT_SECONDS = 0.001
SIMULATION_DECIMATION = 20
ACTION_SCALE = 0.5
RAW_ACTION_MIN = -10.0
RAW_ACTION_MAX = 10.0
ANGULAR_VELOCITY_SCALE = 0.25
DOF_POSITION_SCALE = 1.0
DOF_VELOCITY_SCALE = 0.05
HISTORY_LENGTH = 20
SURVIVAL_ROOT_HEIGHT_MIN = 0.5
REFERENCE_OFFSETS = (
    1,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
)
ZEROED_DOF_VELOCITY_INDICES = (4, 5, 10, 11)

JOINT_NAMES = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
)
DEFAULT_DOF_POSITION = (
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.4,
    0.0,
    1.2,
    0.0,
    -0.4,
    0.0,
    1.2,
)
STIFFNESS = (
    100,
    100,
    100,
    150,
    40,
    40,
    100,
    100,
    100,
    150,
    40,
    40,
    150,
    150,
    150,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
)
DAMPING = (
    2,
    2,
    2,
    4,
    2,
    2,
    2,
    2,
    2,
    4,
    2,
    2,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
)
TORQUE_LIMITS = (
    88,
    139,
    88,
    139,
    50,
    50,
    88,
    139,
    88,
    139,
    50,
    50,
    88,
    50,
    50,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
    25,
)


@dataclass(frozen=True)
class TensorSpec:
    storage: int
    key: str
    shape: tuple[int, ...]
    dtype: str = "<f4"

    @property
    def nbytes(self) -> int:
        itemsize = 8 if self.dtype == "<i8" else 4
        return prod(self.shape) * itemsize

    @property
    def strides(self) -> tuple[int, ...]:
        strides: list[int] = []
        stride = 1
        for width in reversed(self.shape):
            strides.append(stride)
            stride *= width
        return tuple(reversed(strides))


TENSOR_SPECS = (
    TensorSpec(0, "history_encoder.input_linear.weight", (30, 74)),
    TensorSpec(1, "history_encoder.input_linear.bias", (30,)),
    TensorSpec(2, "history_encoder.conv1.weight", (20, 30, 6)),
    TensorSpec(3, "history_encoder.conv1.bias", (20,)),
    TensorSpec(4, "history_encoder.conv2.weight", (10, 20, 4)),
    TensorSpec(5, "history_encoder.conv2.bias", (10,)),
    TensorSpec(6, "history_encoder.output_linear.weight", (64, 30)),
    TensorSpec(7, "history_encoder.output_linear.bias", (64,)),
    TensorSpec(8, "motion_encoder.input_linear.weight", (60, 30)),
    TensorSpec(9, "motion_encoder.input_linear.bias", (60,)),
    TensorSpec(10, "motion_encoder.conv1.weight", (40, 60, 6)),
    TensorSpec(11, "motion_encoder.conv1.bias", (40,)),
    TensorSpec(12, "motion_encoder.conv2.weight", (20, 40, 4)),
    TensorSpec(13, "motion_encoder.conv2.bias", (20,)),
    TensorSpec(14, "motion_encoder.output_linear.weight", (128, 60)),
    TensorSpec(15, "motion_encoder.output_linear.bias", (128,)),
    TensorSpec(16, "actor_backbone.0.weight", (1024, 296)),
    TensorSpec(17, "actor_backbone.0.bias", (1024,)),
    TensorSpec(18, "actor_backbone.2.weight", (1024, 1024)),
    TensorSpec(19, "actor_backbone.2.bias", (1024,)),
    TensorSpec(20, "actor_backbone.4.weight", (512, 1024)),
    TensorSpec(21, "actor_backbone.4.bias", (512,)),
    TensorSpec(22, "actor_backbone.6.weight", (256, 512)),
    TensorSpec(23, "actor_backbone.6.bias", (256,)),
    TensorSpec(24, "actor_backbone.7.weight", (256,)),
    TensorSpec(25, "actor_backbone.7.bias", (256,)),
    TensorSpec(26, "actor_backbone.9.weight", (23, 256)),
    TensorSpec(27, "actor_backbone.9.bias", (23,)),
    TensorSpec(28, "normalizer_count", (1,), "<i8"),
    TensorSpec(29, "normalizer_mean", (2_154,)),
    TensorSpec(30, "normalizer_std", (2_154,)),
)

GENERATED_MEMBER_SHA256 = {
    "data.pkl": "8dd1dedfd489456869fa8c9d4cbbca8661968f6feb846a3121c7a72f81d46df5",
    "constants.pkl": "47f716c83f559b1bf4d31da5b9705d397d0d7ad32b3ae93b666a2e5e8ba27871",
    "constants/0": "244daf676b076b34d5be3f723196f6a9a2fa7e81a05c781394a14fa645e1d6d8",
    "constants/1": "0c27700c82d333aa295692f1814040a962d7bc530253af661d97635dd5ed7af9",
    "version": "7de1555df0c2700329e815b93b32c571c3ea54dc967b89e81ab73b9972b72d1d",
    "byteorder": "180ca01b95f0dfdd36fbb600e51cf6e46c8ef468de56b017847886fefaf7b6f9",
    "code/__torch__.py": "db6f67013ea703730a69d46d83cb6e9516ef0a6f50fea22838699656d8bfeedf",
    "code/__torch__/rsl_rl/modules/dagger_actor.py": (
        "ac6affdf7fdba77760eb843a740f4f121fc86c81ddbd695aa1ba2591b434e46c"
    ),
    "code/__torch__/rsl_rl/modules/actor_critic_mimic.py": (
        "384e6697cb7c0bc3e12272ef3fb02b20f796ebde7e3448c19bfeaa28cd470222"
    ),
    "code/__torch__/rsl_rl/utils/normalizer.py": (
        "846262d26450312dc0370f2d66e98ccbea7bd9cdf9b3a07cc85db20e87e5e5e3"
    ),
    "code/__torch__/torch/nn/modules/container/___torch_mangle_26.py": (
        "d04e5d45648bbf0503f8effa7838833d7c32052a37d2976b77ab520613a4268e"
    ),
    "code/__torch__/torch/nn/modules/normalization.py": (
        "2bcd3dc30d99526e665d463b8f54cbf375ab09acec837a381eafffa19561fd41"
    ),
    "code/__torch__/torch/nn/modules/conv.py": (
        "deb67f2cc4bf04fe2dd91954619011512d9548105bdc4de58df579ebe3f86639"
    ),
    "code/__torch__/torch/nn/modules/conv/___torch_mangle_2.py": (
        "4c86d0f6a6e0be5c72eae7f25c09b056ae4197094ef8e29d48e887834c35dff1"
    ),
    "code/__torch__/torch/nn/modules/conv/___torch_mangle_10.py": (
        "b3037c340557ff51fe1b0bf4d127af093bb0c1747fa5db9a52b319eaaa27062d"
    ),
    "code/__torch__/torch/nn/modules/conv/___torch_mangle_12.py": (
        "670d0d62a2cde6793e5900a2849f689d2b6acbc2bc372a52f152e32677b9cc6a"
    ),
}


@dataclass(frozen=True)
class MotionArraySpan:
    key: str
    offset: int
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * 4


@dataclass(frozen=True)
class MotionSpec:
    name: str
    sha256: str
    size: int
    frames: int
    fps: float
    fps_opcode: str
    arrays: tuple[MotionArraySpan, ...]


def _motion_arrays(
    frames: int, root: int, rotation: int, joints: int
) -> tuple[MotionArraySpan, ...]:
    return (
        MotionArraySpan("root_pos", root, (frames, 3)),
        MotionArraySpan("root_rot", rotation, (frames, 4)),
        MotionArraySpan("dof_pos", joints, (frames, 23)),
    )


MOTION_SPECS = {
    "airkick_stand": MotionSpec(
        "airkick_stand",
        "39f62b0eb6bda3afbe8d81b5523f582a5dd4c81f47618ee26dcb70dbab505755",
        116_345,
        200,
        29.887359198998748,
        "BINFLOAT",
        _motion_arrays(200, 176, 2_622, 5_867),
    ),
    "basic_walk": MotionSpec(
        "basic_walk",
        "9e60b415c56edf163ea657dcf2120238c202dd82fb6b993387239d1987266a5d",
        676_840,
        1_173,
        29.980814325303772,
        "BINFLOAT",
        _motion_arrays(1_173, 177, 14_300, 33_114),
    ),
    "crouchwalk_stand": MotionSpec(
        "crouchwalk_stand",
        "08107069ebd416115fcc1fdb8a47e727975098e98905ace6b8d2f8fc1da99cb6",
        131_897,
        227,
        29.933774834437088,
        "BINFLOAT",
        _motion_arrays(227, 176, 2_946, 6_623),
    ),
    "dance": MotionSpec(
        "dance",
        "c64f96d2ca77937675210e29628d6b2e78b979a276874e347752d8f67eed973e",
        398_591,
        690,
        30.0,
        "BININT1",
        _motion_arrays(690, 170, 8_497, 19_583),
    ),
    "dance_waltz": MotionSpec(
        "dance_waltz",
        "c9e6b02c4fc5a301c24a280ff7787ad5f22726998321ab3b76efb5db9cb1e3cc",
        156_127,
        269,
        33.25062034739454,
        "BINFLOAT",
        _motion_arrays(269, 177, 3_452, 7_802),
    ),
    "kick_walk": MotionSpec(
        "kick_walk",
        "bb4ade92fc55ca10e064c96ec530453b0f100c1600a86d1c992c65c313fb7f15",
        168_765,
        291,
        29.974160206718345,
        "BINFLOAT",
        _motion_arrays(291, 177, 3_716, 8_418),
    ),
    "squat": MotionSpec(
        "squat",
        "c350086e22dc2cfce1a8c17b7cab027cd0e152e24c9f96d5820e85c529e767ef",
        176_287,
        304,
        59.9999885559082,
        "BINFLOAT",
        _motion_arrays(304, 177, 3_872, 8_782),
    ),
    "walk_stand": MotionSpec(
        "walk_stand",
        "13610cc7bca2fca0d0c5b2ad7bcbca61538556bdd82f2645b39980d1eeee7b75",
        129_051,
        222,
        29.932279909706544,
        "BINFLOAT",
        _motion_arrays(222, 176, 2_886, 6_483),
    ),
}
