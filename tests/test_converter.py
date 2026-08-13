from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_core():
    module_name = "remap_lora_28_to_40_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "remap_lora_28_to_40.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def install_comfy_stubs(lora_root: Path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_folder_paths = lambda category: [str(lora_root)]
    folder_paths.get_filename_list = lambda category: [
        str(path.relative_to(lora_root)).replace("\\", "/")
        for path in sorted(lora_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in (".safetensors", ".sft", ".ckpt", ".pt")
    ]

    def get_full_path_or_raise(category, name):
        path = lora_root / name
        if not path.is_file():
            raise FileNotFoundError(name)
        return str(path)

    folder_paths.get_full_path_or_raise = get_full_path_or_raise

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_utils = types.ModuleType("comfy.utils")
    comfy.utils = comfy_utils
    return {
        "folder_paths": folder_paths,
        "comfy": comfy,
        "comfy.utils": comfy_utils,
    }, comfy_utils


def load_backend(lora_root: Path):
    stubs, comfy_utils = install_comfy_stubs(lora_root)
    package_name = "anima_converter_testpkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT)]
    modules = {**stubs, package_name: package}

    with patch.dict(sys.modules, modules):
        core_name = f"{package_name}.remap_lora_28_to_40"
        core_spec = importlib.util.spec_from_file_location(
            core_name, ROOT / "remap_lora_28_to_40.py"
        )
        core_module = importlib.util.module_from_spec(core_spec)
        sys.modules[core_name] = core_module
        assert core_spec.loader is not None
        core_spec.loader.exec_module(core_module)

        backend_name = f"{package_name}.anima_lora_converter"
        backend_spec = importlib.util.spec_from_file_location(
            backend_name, ROOT / "anima_lora_converter.py"
        )
        backend_module = importlib.util.module_from_spec(backend_spec)
        sys.modules[backend_name] = backend_module
        assert backend_spec.loader is not None
        backend_spec.loader.exec_module(backend_module)

    return backend_module, comfy_utils


ANIMA_STATE = {
    f"lora_unet_blocks_{index}_self_attn_q_proj.lora_down.weight": object()
    for index in range(28)
}


class RemapCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_complete_old_to_new_mapping(self):
        self.assertEqual(len(self.core.OLD_TO_NEW), 28)
        self.assertEqual(self.core.OLD_TO_NEW[0], 0)
        self.assertEqual(self.core.OLD_TO_NEW[2], 3)
        self.assertEqual(self.core.OLD_TO_NEW[14], 20)
        self.assertEqual(self.core.OLD_TO_NEW[27], 39)
        self.assertTrue(
            set(self.core.OLD_TO_NEW.values()).isdisjoint(self.core.INSERTION_POSITIONS)
        )

    def test_remap_does_not_create_inserted_layer_keys(self):
        remapped = self.core.remap_lora_state_dict(ANIMA_STATE, source_name="test")
        mapped_indices = {
            self.core.find_main_block(key)[1]
            for key in remapped
            if self.core.find_main_block(key)[1] is not None
        }
        self.assertEqual(len(remapped), len(ANIMA_STATE))
        self.assertTrue(mapped_indices.isdisjoint(self.core.INSERTION_POSITIONS))

    def test_passthrough_key_is_preserved(self):
        passthrough = object()
        state = {
            "lora_unet_blocks_0_self_attn_q_proj.alpha": object(),
            "ss.some_global_value": passthrough,
        }
        remapped = self.core.remap_lora_state_dict(state)
        self.assertIs(remapped["ss.some_global_value"], passthrough)

    def test_invalid_high_layer_raises(self):
        with self.assertRaisesRegex(self.core.LoraRemapError, "不支持的主干层 28"):
            self.core.remap_lora_state_dict(
                {"lora_unet_blocks_28_self_attn_q_proj.alpha": object()},
                source_name="invalid.safetensors",
            )

    def test_missing_main_layer_raises(self):
        with self.assertRaisesRegex(self.core.LoraRemapError, "未包含可识别"):
            self.core.remap_lora_state_dict({"metadata.only": object()})

    def test_collision_raises(self):
        with self.assertRaisesRegex(self.core.LoraRemapError, "键名冲突"):
            self.core.remap_lora_state_dict(
                {"lora_unet_blocks_0_x": object(), "lora_unet_blocks_00_x": object()}
            )

    def test_net_prefix_remaps_too(self):
        remapped = self.core.remap_lora_state_dict(
            {"net.blocks.14.self_attn.output_proj.weight": object()}
        )
        self.assertIn("net.blocks.20.self_attn.output_proj.weight", remapped)

    def test_diffusion_model_prefix_remaps_too(self):
        remapped = self.core.remap_lora_state_dict(
            {"diffusion_model.blocks.14.self_attn.output_proj.lora_A.weight": object()}
        )
        self.assertIn(
            "diffusion_model.blocks.20.self_attn.output_proj.lora_A.weight", remapped
        )

    def test_diffusion_model_high_layer_raises_with_hint(self):
        with self.assertRaisesRegex(self.core.LoraRemapError, "可能已经是 40 层结构"):
            self.core.remap_lora_state_dict(
                {"diffusion_model.blocks.28.self_attn.q.lora_A.weight": object()},
                source_name="already40.safetensors",
            )


class ConverterNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.backend, self.comfy_utils = load_backend(self.root)

        self.written: dict[str, tuple[dict, dict | None]] = {}
        self.load_table: dict[str, tuple[dict, dict | None]] = {}

        def load_torch_file(path, **kwargs):
            key = str(path)
            if key not in self.load_table:
                raise OSError(f"missing {key}")
            return self.load_table[key]

        def save_torch_file(sd, ckpt, metadata=None):
            self.written[str(ckpt)] = (sd, metadata)

        self.comfy_utils.load_torch_file = load_torch_file
        self.comfy_utils.save_torch_file = save_torch_file

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_lora(self, name: str, state=None, metadata=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
        self.load_table[str(path)] = (
            state if state is not None else dict(ANIMA_STATE), metadata
        )

    def convert_node(self, **kwargs):
        defaults = {
            "source_mode": "names",
            "lora_names": "",
            "source_subfolder": "",
            "filter": "*",
            "output_subfolder": "anima_40",
            "name_suffix": "_40",
            "overwrite": False,
        }
        defaults.update(kwargs)
        return self.backend.Anima28To40BatchConverter().convert(**defaults)

    def test_split_names_handles_separators_and_duplicates(self):
        names, duplicates = self.backend.split_names(
            "a.safetensors\nb.safetensors, a.safetensors;; c.safetensors"
        )
        self.assertEqual(names, ["a.safetensors", "b.safetensors", "c.safetensors"])
        self.assertEqual(duplicates, ["a.safetensors"])

    def test_names_mode_converts_and_saves(self):
        self.add_lora("a.safetensors")
        self.add_lora("b.safetensors")
        summary = self.convert_node(lora_names="a.safetensors, b.safetensors")

        self.assertEqual(len(self.written), 2)
        out_a = self.written[str(self.root / "anima_40" / "a_40.safetensors")]
        self.assertIsInstance(out_a[1], dict)
        self.assertEqual(out_a[1][self.backend.METADATA_MARKER], "28to40")
        keys = list(out_a[0])
        self.assertIn("lora_unet_blocks_3_self_attn_q_proj.lora_down.weight", keys)  # 2 -> 3
        self.assertIn("lora_unet_blocks_39_self_attn_q_proj.lora_down.weight", keys)  # 27 -> 39
        self.assertNotIn("lora_unet_blocks_2_self_attn_q_proj.lora_down.weight", keys)
        self.assertIn("[成功] a.safetensors → anima_40/a_40.safetensors", summary)
        self.assertIn("成功: 2", summary)

    def test_names_mode_metadata_roundtrip(self):
        self.add_lora("a.safetensors", metadata={"ss_sd_model_name": "anima v10"})
        self.convert_node(lora_names="a.safetensors")
        _, metadata = self.written[str(self.root / "anima_40" / "a_40.safetensors")]
        self.assertEqual(metadata["ss_sd_model_name"], "anima v10")
        self.assertEqual(metadata[self.backend.METADATA_MARKER], "28to40")

    def test_names_mode_missing_file_aborts_before_writing(self):
        self.add_lora("a.safetensors")
        with self.assertRaisesRegex(ValueError, "无法定位 LoRA"):
            self.convert_node(lora_names="a.safetensors, missing.safetensors")
        self.assertEqual(self.written, {})

    def test_target_exists_skips_unless_overwrite(self):
        self.add_lora("a.safetensors")
        target = self.root / "anima_40" / "a_40.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing")

        # 全部已存在 → 整体报错并说明原因
        with self.assertRaisesRegex(ValueError, "目标文件已存在"):
            self.convert_node(lora_names="a.safetensors")
        self.assertEqual(self.written, {})

        # 部分已存在 → 跳过已存在的，转换其余
        self.add_lora("b.safetensors")
        summary = self.convert_node(lora_names="a.safetensors, b.safetensors")
        self.assertIn("[跳过] a.safetensors: 目标文件已存在", summary)
        self.assertIn("成功: 1", summary)
        self.assertIn(str(self.root / "anima_40" / "b_40.safetensors"), self.written)

        summary = self.convert_node(lora_names="a.safetensors", overwrite=True)
        self.assertIn("成功: 1", summary)
    def test_same_stem_different_extension_conflicts(self):
        self.add_lora("x.safetensors")
        self.add_lora("x.sft")
        with self.assertRaisesRegex(ValueError, "冲突"):
            self.convert_node(lora_names="x.safetensors, x.sft")
        self.assertEqual(self.written, {})

    def test_names_mode_keeps_subfolder_structure(self):
        self.add_lora("anima/sub/foo.safetensors")
        self.convert_node(lora_names="anima/sub/foo.safetensors")
        self.assertIn(
            str(self.root / "anima_40" / "anima" / "sub" / "foo_40.safetensors"),
            self.written,
        )

    def test_folder_mode_skips_non_anima_and_converts_good(self):
        self.add_lora("anima/good.safetensors")
        self.add_lora("anima/junk.pt", {"not_a_lora.alpha": object()})
        (self.root / "anima" / "readme.txt").write_text("skip by extension")

        summary = self.convert_node(source_mode="folder", source_subfolder="anima")
        self.assertIn("成功: 1", summary)
        self.assertIn("[跳过] anima/junk.pt", summary)
        self.assertIn(str(self.root / "anima_40" / "good_40.safetensors"), self.written)

    def test_folder_mode_strips_scan_root_prefix(self):
        self.add_lora("anima/deep/a.safetensors")
        self.convert_node(source_mode="folder", source_subfolder="anima")
        self.assertIn(str(self.root / "anima_40" / "deep" / "a_40.safetensors"), self.written)

    def test_folder_mode_filter_matches(self):
        self.add_lora("anima/keep_foo.safetensors")
        self.add_lora("anima/drop_bar.safetensors")
        summary = self.convert_node(
            source_mode="folder", source_subfolder="anima", filter="foo"
        )
        self.assertIn("成功: 1", summary)
        self.assertNotIn("drop_bar", summary)

    def test_folder_mode_load_failure_is_skipped(self):
        self.add_lora("anima/good.safetensors")
        self.add_lora("anima/broken.safetensors")
        broken_path = str(self.root / "anima" / "broken.safetensors")
        comfy_utils = self.comfy_utils
        old_load = comfy_utils.load_torch_file

        def load_torch_file(path, **kwargs):
            if str(path) == broken_path:
                raise OSError("corrupt file")
            return old_load(path, **kwargs)

        comfy_utils.load_torch_file = load_torch_file
        summary = self.convert_node(source_mode="folder", source_subfolder="anima")
        self.assertIn("成功: 1", summary)
        self.assertIn("[跳过] anima/broken.safetensors", summary)

    def test_folder_mode_all_skipped_raises(self):
        self.add_lora("anima/junk.pt", {"not_a_lora.alpha": object()})
        with self.assertRaisesRegex(ValueError, "全部文件都被跳过"):
            self.convert_node(source_mode="folder", source_subfolder="anima")

    def test_suffix_not_duplicated(self):
        self.add_lora("a_40.safetensors")
        self.convert_node(lora_names="a_40.safetensors")
        self.assertIn(str(self.root / "anima_40" / "a_40.safetensors"), self.written)

    def test_names_mode_invalid_anima_aborts(self):
        self.add_lora("bad.safetensors", {"not_a_lora.alpha": object()})
        with self.assertRaisesRegex(ValueError, "加载或映射失败"):
            self.convert_node(lora_names="bad.safetensors")

    def test_empty_names_raises(self):
        with self.assertRaisesRegex(ValueError, "没有找到可转换"):
            self.convert_node(lora_names="")

    def test_sanitize_suffix(self):
        self.assertEqual(self.backend.sanitize_suffix("a/b"), "a_b")
        self.assertEqual(self.backend.sanitize_suffix("x:y*z?"), "x_y_z_")
        self.assertEqual(self.backend.sanitize_suffix(""), "")

    def test_path_traversal_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能包含 '..'"):
            self.convert_node(lora_names="..\\escape.safetensors")


if __name__ == "__main__":
    unittest.main()
