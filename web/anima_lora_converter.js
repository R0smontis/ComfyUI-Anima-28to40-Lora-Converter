import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

const NODE_CLASS = "Anima28To40BatchConverter";
const MODE_WIDGETS = {
    names: ["lora_names"],
    folder: ["source_subfolder", "filter"],
};
const TOGGLEABLE = ["lora_names", "source_subfolder", "filter"];
const SEARCH_LIMIT = 500;

function setWidgetHidden(widget, hide) {
    if (!widget) {
        return;
    }
    if (hide) {
        if (widget.type !== "hidden") {
            widget.__animaOriginalType = widget.type;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
        }
    } else if (widget.type === "hidden") {
        widget.type = widget.__animaOriginalType || "text";
        delete widget.computeSize;
    }
}

function refreshVisibility(node) {
    const modeWidget = node.widgets?.find((widget) => widget.name === "source_mode");
    const mode = modeWidget?.value || "names";
    const visible = MODE_WIDGETS[mode] || [];
    for (const name of TOGGLEABLE) {
        setWidgetHidden(
            node.widgets?.find((widget) => widget.name === name),
            !visible.includes(name)
        );
    }
    node.setSize(node.computeSize());
}

function currentNames(node) {
    const widget = node.widgets?.find((item) => item.name === "lora_names");
    return String(widget?.value || "")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
}

async function openBrowseDialog(node) {
    let catalog;
    try {
        const response = await api.fetchApi("/models/loras");
        catalog = (await response.json() || [])
            .map((name) => String(name))
            .filter(Boolean);
    } catch (error) {
        app.alert?.(`获取 LoRA 文件列表失败：${error}`);
        return;
    }
    if (!catalog.length) {
        app.alert?.("LoRA 目录为空");
        return;
    }
    const selected = new Set(catalog.filter((name) => {
        const existing = currentNames(node);
        return !existing.includes(name);
    }));

    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:fixed;inset:0;background:rgba(0,0,0,0.45);z-index:9000;" +
        "display:flex;align-items:center;justify-content:center;";
    const dialog = document.createElement("div");
    dialog.style.cssText =
        "background:#2b2b2b;color:#ddd;border:1px solid #555;border-radius:10px;" +
        "padding:12px;width:520px;max-width:92vw;max-height:78vh;display:flex;" +
        "flex-direction:column;gap:8px;font:13px sans-serif;box-shadow:0 8px 32px rgba(0,0,0,0.6);";

    const title = document.createElement("div");
    title.textContent = "浏览添加 LoRA 文件";
    title.style.cssText = "font-weight:bold;color:#fff;";
    dialog.appendChild(title);

    const search = document.createElement("input");
    search.type = "text";
    search.placeholder = "搜索文件名或子文件夹（不区分大小写）";
    search.style.cssText =
        "background:#1e1e1e;color:#ddd;border:1px solid #555;border-radius:6px;padding:6px 8px;outline:none;";
    dialog.appendChild(search);

    const list = document.createElement("div");
    list.style.cssText =
        "overflow-y:auto;min-height:220px;max-height:52vh;background:#1e1e1e;" +
        "border:1px solid #444;border-radius:6px;padding:4px;display:flex;flex-direction:column;";
    dialog.appendChild(list);

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex;justify-content:space-between;align-items:center;gap:8px;";
    const count = document.createElement("span");
    count.style.color = "#9a9a9a";
    footer.appendChild(count);

    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:8px;";
    const mkButton = (label, onClick, primary) => {
        const button = document.createElement("button");
        button.textContent = label;
        button.style.cssText =
            "background:" + (primary ? "#3d6b99" : "#3a3a3a") + ";color:#fff;border:1px solid " +
            (primary ? "#5b8cc4" : "#555") + ";border-radius:6px;padding:6px 12px;cursor:pointer;";
        button.addEventListener("click", onClick);
        return button;
    };

    const render = () => {
        const query = search.value.trim().toLowerCase();
        const matches = catalog.filter((name) => !query || name.toLowerCase().includes(query));
        list.replaceChildren();
        const limit = Math.min(matches.length, SEARCH_LIMIT);
        for (const name of matches.slice(0, limit)) {
            const row = document.createElement("label");
            row.style.cssText =
                "display:flex;align-items:center;gap:8px;padding:4px 6px;cursor:pointer;border-radius:4px;";
            row.addEventListener("mouseenter", () => (row.style.background = "#333"));
            row.addEventListener("mouseleave", () => (row.style.background = ""));
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = selected.has(name);
            checkbox.addEventListener("change", () => {
                checkbox.checked ? selected.add(name) : selected.delete(name);
                updateCount();
            });
            const label = document.createElement("span");
            label.textContent = name;
            label.style.cssText = "user-select:none;word-break:break-all;";
            row.append(checkbox, label);
            list.appendChild(row);
        }
        if (matches.length > limit) {
            const note = document.createElement("div");
            note.textContent = `… 还有 ${matches.length - limit} 个文件未显示，请用搜索缩小范围`;
            note.style.cssText = "padding:6px;color:#9a9a9a;";
            list.appendChild(note);
        }
        updateCount();
    };
    const updateCount = () => {
        count.textContent = `已选 ${selected.size} 个`;
    };

    search.addEventListener("input", render);

    const allButton = mkButton("全选", () => {
        catalog.forEach((name) => selected.add(name));
        render();
    });
    const clearButton = mkButton("清空", () => {
        selected.clear();
        render();
    });
    const cancelButton = mkButton("取消", () => overlay.remove());
    const addButton = mkButton(`添加选中 (${selected.size})`, () => {
        const merged = [...new Set([...currentNames(node), ...selected])];
        const widget = node.widgets?.find((item) => item.name === "lora_names");
        if (widget) {
            widget.value = merged.join("\n");
        }
        const modeWidget = node.widgets?.find((item) => item.name === "source_mode");
        if (modeWidget && modeWidget.value !== "names") {
            modeWidget.value = "names";
        }
        refreshVisibility(node);
        node.graph?.setDirtyCanvas?.(true, true);
        overlay.remove();
    }, true);
    addButton.style.cssText += ";font-weight:bold;";
    updateCount();

    actions.append(allButton, clearButton, cancelButton, addButton);
    footer.append(count, actions);
    dialog.append(footer);
    overlay.appendChild(dialog);

    const close = (event) => {
        if (event.key === "Escape") {
            overlay.remove();
            document.removeEventListener("keydown", close);
        }
    };
    document.addEventListener("keydown", close);
    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) {
            overlay.remove();
            document.removeEventListener("keydown", close);
        }
    });

    document.body.appendChild(overlay);
    search.focus();
    render();
}

app.registerExtension({
    name: "anima.28-to-40-batch-lora-converter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) {
            return;
        }

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            this.title = "Anima 28→40 批量 LoRA 转换器";


            this.addWidget(
                "button",
                "＋ 浏览添加 LoRA…",
                null,
                () => openBrowseDialog(this),
                { serialize: false }
            );

            const modeWidget = this.widgets?.find((widget) => widget.name === "source_mode");
            if (modeWidget) {
                const originalCallback = modeWidget.callback;
                const nodeInstance = this;
                modeWidget.callback = function (value, canvas, nodeArg, pos, event) {
                    const result = originalCallback?.call(this, value, canvas, nodeArg, pos, event);
                    refreshVisibility(nodeArg || nodeInstance);
                    return result;
                };
            }
            refreshVisibility(this);
            this.setSize(this.computeSize());
            return result;
        };
    },
});
