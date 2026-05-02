"""Lung-RADS color-coded result card.

A demo-grade visualization that shows each detected nodule as a colored card,
with the Lung-RADS band, malignancy probability, and recommended action.
Replaces the plain-text result list. Designed so that during a defense / demo,
a viewer can read the diagnosis at a glance.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


# Lung-RADS band -> (background hex, accent text hex, light-text)
_LUNG_RADS_PALETTE = {
    "2":  ("#1e293b", "#86efac", "#cbd5e1"),    # green band  - benign
    "3":  ("#1e293b", "#7dd3fc", "#cbd5e1"),    # blue band   - low suspicion
    "4A": ("#1e293b", "#fcd34d", "#cbd5e1"),    # yellow band - moderate
    "4B": ("#1e293b", "#fb923c", "#cbd5e1"),    # orange band - high
    "4X": ("#7f1d1d", "#fee2e2", "#fee2e2"),    # red band    - very high
}


class LungRadsCard(QFrame):
    """One colored card for one detected nodule."""

    def __init__(self, nodule: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        bg, accent, text = _LUNG_RADS_PALETTE.get(nodule["lung_rads"], _LUNG_RADS_PALETTE["3"])
        self.setObjectName("lung_rads_card")
        self.setStyleSheet(
            f"""
            QFrame#lung_rads_card {{
                background-color: {bg};
                border-left: 5px solid {accent};
                border-radius: 8px;
                padding: 12px;
                margin: 4px 0;
            }}
            QLabel[role="band"]      {{ color: {accent}; font-size: 24px; font-weight: 800; }}
            QLabel[role="label"]     {{ color: {accent}; font-size: 14px; font-weight: 700; }}
            QLabel[role="probability"] {{ color: {accent}; font-size: 32px; font-weight: 800; }}
            QLabel[role="meta"]      {{ color: {text}; font-size: 11px; }}
            QLabel[role="action"]    {{ color: {text}; font-size: 13px; font-weight: 600; padding-top: 6px; }}
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # Top row: nodule index | Lung-RADS band | malignancy probability
        top = QHBoxLayout()
        idx_label = QLabel(f"結節 #{nodule['idx']}")
        idx_label.setProperty("role", "meta")
        top.addWidget(idx_label)
        top.addStretch()

        band_label = QLabel(f"Lung-RADS {nodule['lung_rads']}")
        band_label.setProperty("role", "band")
        top.addWidget(band_label)
        top.addStretch()

        prob_label = QLabel(f"{nodule['mal_prob'] * 100:.1f}%")
        prob_label.setProperty("role", "probability")
        prob_label.setAlignment(Qt.AlignRight)
        top.addWidget(prob_label)

        outer.addLayout(top)

        # Middle: clinical category + slice count
        middle = QHBoxLayout()
        cat_label = QLabel(nodule["label"])
        cat_label.setProperty("role", "label")
        middle.addWidget(cat_label)
        middle.addStretch()
        slice_label = QLabel(f"{nodule['n_slices']} slice 聚合")
        slice_label.setProperty("role", "meta")
        middle.addWidget(slice_label)
        outer.addLayout(middle)

        # Bottom: recommended action
        action_label = QLabel(f"建議: {nodule['action']}")
        action_label.setProperty("role", "action")
        action_label.setWordWrap(True)
        outer.addWidget(action_label)


class LungRadsPanel(QWidget):
    """Container that holds all nodule cards stacked vertically."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._placeholder = QLabel("執行分類後將顯示 Lung-RADS 評估結果")
        self._placeholder.setStyleSheet(
            "color: #475569; font-size: 12px; padding: 24px;"
            "border: 1px dashed #334155; border-radius: 6px;"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)
        self._layout.addStretch()

    def render_nodules(self, nodules: list[dict]) -> None:
        # Clear current children
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not nodules:
            self._layout.addWidget(self._placeholder)
            self._layout.addStretch()
            return
        # Sort: highest malignancy probability first (demo impact)
        for n in sorted(nodules, key=lambda x: -x["mal_prob"]):
            self._layout.addWidget(LungRadsCard(n))
        self._layout.addStretch()
