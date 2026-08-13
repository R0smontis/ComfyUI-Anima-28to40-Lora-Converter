"""ComfyUI 节点：批量把 Anima 28 层 LoRA 转换为 40 层并保存到 loras 文件夹。

两种输入方式：
- names：在文本框里逐行（或用逗号/分号分隔）填写 loras 下的相对路径，严格模式，任何错误都会中止。
- folder：扫描 loras 下某个子文件夹（支持递归），按 filter 过滤文件名，宽容模式，无法转换的文件跳过并记入摘要。

输出统一写到 models/loras/<output_subfolder> 下，保留源文件的相对子目录结构，
文件名追加 name_suffix（已带后缀则不重复追加），始终输出 .safetensors。
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import comfy.utils
import folder_paths

from .remap_lora_28_to_40 import LoraRemapError, remap_lora_state_dict

SUPPORTED_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt")
METADATA_MARKER = "ss_anima_remap"
METADATA_MARKER_VALUE = "28to40"

NAME_SEPARATORS = re.compile(r"[\n,;]+")
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"|?*\\/]+')
ABSOLUTE_PATH_PREFIX = re.compile(r"^([A-Za-z]:[\\/]|[\\/])")




def split_names(text: str | None) -> tuple[list[str], list[str]]:
    """按换行/逗号/分号切分名称；返回 (去重后名称, 被丢弃的重复项)。"""
    raw = [part.strip() for part in NAME_SEPARATORS.split(text or "") if part.strip()]
    seen: set[str] = set()
    names: list[str] = []
    duplicates: list[str] = []
    for name in raw:
        if name in seen:
            duplicates.append(name)
        else:
            seen.add(name)
            names.append(name)
    return names, duplicates


def validate_relative_name(name: str) -> None:
    """名称必须是 loras 文件夹下的相对路径，禁止绝对路径与 .. 逃逸。"""
    if name != name.strip():
        raise ValueError(f"LoRA 名称 {name!r} 首尾不能有空白字符")
    if ABSOLUTE_PATH_PREFIX.match(name):
        raise ValueError(f"LoRA 名称 {name!r} 必须是 loras 下的相对路径，不能是绝对路径")
    if ".." in Path(name).parts:
        raise ValueError(f"LoRA 名称 {name!r} 不能包含 '..' 路径片段")


def sanitize_suffix(suffix: str) -> str:
    return UNSAFE_FILENAME_CHARS.sub("_", suffix).strip(" .")


def matches_filter(filename: str, pattern: str) -> bool:
    """不含通配符时按大小写不敏感的子串匹配，否则按 fnmatch 匹配。"""
    lowered = filename.lower()
    if not re.search(r"[\*\?\[\]]", pattern):
        return pattern.lower() in lowered
    return fnmatch.fnmatch(lowered, pattern.lower())


def load_and_remap(path: Path, source_name: str) -> tuple[dict[str, Any], dict[str, str] | None]:
    """加载 LoRA 文件并重映射键名；返回 (重映射后的 state dict, 元数据)。"""

    loaded = _load_torch_file(path)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        state_dict, metadata = loaded
    else:
        state_dict, metadata = loaded, None
    if not isinstance(state_dict, dict):
        raise TypeError(f"加载结果不是 state dict，而是 {type(state_dict).__name__}")

    remapped = remap_lora_state_dict(state_dict, source_name=source_name)

    if metadata is None:
        metadata_out: dict[str, str] | None = {METADATA_MARKER: METADATA_MARKER_VALUE}
    else:
        try:
            metadata_out = {str(key): str(value) for key, value in metadata.items()}
        except Exception as error:
            raise TypeError(f"LoRA 元数据无法转换为字符串键值：{error}") from error
        metadata_out.setdefault(METADATA_MARKER, METADATA_MARKER_VALUE)
    return remapped, metadata_out


def _load_torch_file(path: Path) -> Any:
    """兼容支持及不支持 return_metadata 的 ComfyUI 版本。"""
    try:
        return comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    except TypeError as error:
        if "return_metadata" not in str(error):
            raise
        return comfy.utils.load_torch_file(str(path), safe_load=True)


def target_relative_path(
    source_rel: Path,
    scan_root: Path,
    output_subfolder: str,
    name_suffix: str,
) -> Path:
    """计算输出文件的相对路径：剥掉扫描根前缀，保留其余子目录，追加后缀。"""
    rel = source_rel.relative_to(scan_root) if str(scan_root) not in ("", ".") else source_rel
    stem = source_rel.stem
    if name_suffix and not stem.endswith(name_suffix):
        stem = f"{stem}{name_suffix}"
    return Path(output_subfolder) / rel.parent / f"{stem}.safetensors"


def format_summary(
    converted: list[tuple[str, str]],
    skipped: list[tuple[str, str]],
) -> str:
    lines = ["Anima 28→40 批量转换完成", f"成功: {len(converted)} | 跳过: {len(skipped)}"]
    for source, target in converted:
        lines.append(f"[成功] {source} → {target}")
    for source, reason in skipped:
        lines.append(f"[跳过] {source}: {reason}")
    return "\n".join(lines)


class Anima28To40BatchConverter:
    """把一批 28 层 Anima LoRA 重映射为 40 层，并批量写入 loras 文件夹。"""

    CATEGORY = "loaders/Anima"
    FUNCTION = "convert"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    DESCRIPTION = (
        "批量把 Anima 28 层 LoRA 转换为 40 层 key 结构并保存到 loras 文件夹。"
        "names 模式严格（出错即中止）；folder 模式宽容（跳过无法转换的文件）。"
    )
    IS_CHANGED = float("NaN")  # 工具节点：每次队列执行都重新运行

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "source_mode": (["names", "folder"], {"default": "names"}),
                "lora_names": (
                    "STRING",
                    {"default": "", "multiline": True, "placeholder": "loras 下的相对路径，一行一个，可用逗号分隔\n例如：anima\\foo.safetensors"},
                ),
                "source_subfolder": ("STRING", {"default": "anima", "placeholder": "anima"}),
                "filter": ("STRING", {"default": "*", "placeholder": "*.safetensors 或关键词"}),
                "output_subfolder": ("STRING", {"default": "anima_40"}),
                "name_suffix": ("STRING", {"default": "_40"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
        }

    def _resolve_source(self, name: str) -> Path:
        try:
            return Path(folder_paths.get_full_path_or_raise("loras", name)).resolve()
        except Exception as error:
            raise FileNotFoundError(f"无法定位 LoRA {name!r}: {error}") from error

    def _collect_sources(
        self,
        source_mode: str,
        lora_names: str,
        source_subfolder: str,
        pattern: str,
    ) -> tuple[list[tuple[str, Path]], Path, list[str]]:
        """返回 (来源列表, 扫描根目录(loras 根下相对路径), 收集阶段错误)。"""
        loras_root = Path(folder_paths.get_folder_paths("loras")[0]).resolve()
        errors: list[str] = []

        if source_mode == "names":
            names, duplicates = split_names(lora_names)
            for duplicate in duplicates:
                errors.append(f"重复的名称已忽略：{duplicate}")
            sources: list[tuple[str, Path]] = []
            for name in names:
                try:
                    validate_relative_name(name)
                    sources.append((name, self._resolve_source(name)))
                except (ValueError, FileNotFoundError) as error:
                    errors.append(str(error))
            return sources, Path("."), errors

        scan_root_name = source_subfolder.strip().strip("/\\") if source_subfolder else ""
        if scan_root_name:
            validate_relative_name(scan_root_name)
        scan_dir = (loras_root / scan_root_name) if scan_root_name else loras_root
        if not scan_dir.is_dir():
            return [], Path(scan_root_name), [f"源文件夹不存在：{scan_dir}"]

        found: list[tuple[str, Path]] = []
        for path in sorted(scan_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if not matches_filter(path.name, pattern):
                continue
            found.append((str(path.relative_to(loras_root)).replace("\\", "/"), path))
        return found, Path(scan_root_name), errors
    def convert(
        self,
        source_mode: str = "names",
        lora_names: str = "",
        source_subfolder: str = "",
        filter: str = "*",
        output_subfolder: str = "anima_40",
        name_suffix: str = "_40",
        overwrite: bool = False,
        _lora_catalog: str | None = None,
        **kwargs: Any,
    ) -> str:
        """执行批量转换；返回摘要字符串。失败时抛出 ValueError。"""
        loras_root = Path(folder_paths.get_folder_paths("loras")[0]).resolve()
        validate_relative_name(output_subfolder)
        suffix = sanitize_suffix(name_suffix)

        sources, scan_root, collection_errors = self._collect_sources(
            source_mode, lora_names, source_subfolder, filter
        )
        names_mode = source_mode == "names"
        if not sources:
            details = "\n".join(f"  - {error}" for error in collection_errors) or "  没有匹配任何 LoRA 文件"
            raise ValueError("没有找到可转换的 LoRA。\n" + details)

        # —— 计划阶段：先算清所有目标并检查冲突，任何问题都在写文件之前整体中止 ——
        plan: list[tuple[str, Path, Path]] = []
        target_owners: dict[Path, str] = {}
        plan_errors: list[str] = list(collection_errors)
        for name, source_path in sources:
            target_rel = target_relative_path(
                Path(name.replace("\\", "/")), scan_root, output_subfolder, suffix
            )
            target_abs = (loras_root / target_rel).resolve()
            if target_abs == source_path:
                plan_errors.append(
                    f"{name}: 输出路径与源文件相同（请修改 output_subfolder 或 name_suffix）"
                )
                continue
            if target_abs in target_owners:
                plan_errors.append(
                    f"{name}: 输出目标 {target_rel} 与 {target_owners[target_abs]} 冲突"
                )
                continue
            target_owners[target_abs] = name
            plan.append((name, source_path, target_abs))

        if plan_errors:
            raise ValueError(
                "批量转换中止：\n  " + "\n  ".join(plan_errors)
            )

        # —— 执行阶段：逐文件转换并写出 ——
        converted: list[tuple[str, str]] = []
        skipped: list[tuple[str, str]] = []
        for name, source_path, target_abs in plan:
            if target_abs.exists() and not overwrite:
                skipped.append((name, "目标文件已存在（设置 overwrite=true 可覆盖）"))
                continue
            target_rel = str(target_abs.relative_to(loras_root)).replace("\\", "/")
            try:
                remapped, metadata = load_and_remap(source_path, source_name=name)
            except (LoraRemapError, OSError, TypeError, ValueError) as error:
                message = f"加载或映射失败：{error}"
                if names_mode:
                    raise ValueError(
                        f"{name}: {message}（此前已成功写出 {len(converted)} 个文件）"
                    ) from error
                skipped.append((name, message))
                continue
            try:
                target_abs.parent.mkdir(parents=True, exist_ok=True)
                comfy.utils.save_torch_file(remapped, str(target_abs), metadata=metadata)
            except Exception as error:
                message = f"写入失败：{error}"
                if names_mode:
                    raise ValueError(
                        f"{name}: {message}（此前已成功写出 {len(converted)} 个文件）"
                    ) from error
                skipped.append((name, message))
                continue
            converted.append((name, target_rel))

        if not converted and skipped:
            raise ValueError(
                "全部文件都被跳过：\n  "
                + "\n  ".join(f"{name}: {reason}" for name, reason in skipped)
            )
        return format_summary(converted, skipped)


NODE_CLASS_MAPPINGS = {
    "Anima28To40BatchConverter": Anima28To40BatchConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anima28To40BatchConverter": "Anima 28→40 批量 LoRA 转换器",
}

__all__ = [
    "Anima28To40BatchConverter",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
