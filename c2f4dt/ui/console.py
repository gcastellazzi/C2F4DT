from __future__ import annotations

import codeop
import re
from typing import Callable, Dict, Optional, List

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor


class PythonHighlighter(QSyntaxHighlighter):
    """Very lightweight Python syntax highlighter for the console."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rules = []

        def fmt(color_hex: str, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color_hex))
            if bold:
                f.setFontWeight(QtGui.QFont.Bold)
            return f

        # Keywords
        kw_fmt = fmt("#C586C0", bold=True)
        keywords = [
            'False','class','finally','is','return','None','continue','for','lambda','try',
            'True','def','from','nonlocal','while','and','del','global','not','with',
            'as','elif','if','or','yield','assert','else','import','pass','break','except','in','raise'
        ]
        for kw in keywords:
            self.rules.append((QtCore.QRegularExpression(r"\b" + kw + r"\b"), kw_fmt))

        # Strings
        str_fmt = fmt("#CE9178")
        self.rules.append((QtCore.QRegularExpression(r"'[^'\\]*(?:\\.[^'\\]*)*'"), str_fmt))
        self.rules.append((QtCore.QRegularExpression(r'"[^"\\]*(?:\\.[^"\\]*)*"'), str_fmt))

        # Comments
        com_fmt = fmt("#6A9955")
        self.rules.append((QtCore.QRegularExpression(r"#.*$"), com_fmt))

    def highlightBlock(self, text: str) -> None:
        for rx, form in self.rules:
            it = rx.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), form)


class ConsoleWidget(QtWidgets.QPlainTextEdit):
    """A simple Abaqus-like interactive Python console with TAB autocomplete.

    Features
    --------
    - Single widget, terminal-like: input appears inline with a ``>>>`` prompt.
    - Multi-line paste/edit supported; continuation prompts use ``...``.
    - History navigation with Up/Down on the current prompt.
    - TAB shows autocomplete suggestions; if a single match, completes inline.
    - Executes code in a controlled namespace provided by ``context_provider``.

    Notes
    -----
    This widget protects previous output: editing is only allowed after the
    current prompt position. The console uses ``codeop`` to detect whether the
    current buffer is a complete Python block or needs continuation.
    """

    sigExecuted = QtCore.Signal(str)

    def __init__(self, context_provider: Callable[[], Dict[str, object]], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._context_provider = context_provider

        # Visuals: monospace, no line wrap, focus at end
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        font.setPointSize(font.pointSize() + 1)
        self.setFont(font)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setUndoRedoEnabled(False)

        # Prompts & state
        self._ps1 = ">>> "
        self._ps2 = "... "
        self._compiler = codeop.CommandCompiler()
        self._buffer_lines: List[str] = []  # current multi-line input buffer
        self._history: List[str] = []
        self._hist_idx: int = -1
        self._prompt_pos: int = 0  # document position of current prompt start

        self._write("C2F4DT Python Console. Type Python and press Enter. TAB for suggestions.\n")
        self._insert_prompt(primary=True)

        # Persistent execution namespace (survives between commands)
        self._ns: Dict[str, object] = {}
        self._refresh_namespace()

        # Accept drag&drop/paste of multi-line text
        self.setAcceptDrops(True)
        self.installEventFilter(self)

        # Syntax highlighting
        self._highlighter = PythonHighlighter(self.document())

        # Popup completer (TAB)
        self._completer = QtWidgets.QCompleter([], self)
        self._completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        self._completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self._completer.setWidget(self)
        self._completer.activated.connect(self._apply_completion)

        # Precompiled regex for completion contexts (name/attr/item)
        self._rx_item = re.compile(r"([\w\.]+)\[(?:'|\")(?:([\w\-.]*))?$")
        self._rx_attr = re.compile(r"([\w\.]+)\.([\w\-.]*)$")
        self._rx_name = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)$")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _write(self, text: str) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def _insert_prompt(self, primary: bool) -> None:
        prompt = self._ps1 if primary else self._ps2
        self._write(prompt)
        self._prompt_pos = self.textCursor().position()

    def _current_input_text(self) -> str:
        """Return text from current prompt to end of document."""
        doc = self.document()
        return doc.toPlainText()[self._prompt_pos:]

    def _set_current_input_text(self, text: str) -> None:
        cursor = self.textCursor()
        cursor.setPosition(self._prompt_pos)
        cursor.movePosition(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(text)
        self.setTextCursor(cursor)

    def _refresh_namespace(self) -> None:
        """Seed/refresh the persistent namespace with the latest context.

        Keeps previously defined symbols (variables/functions created in console).
        """
        try:
            ctx = self._context_provider() or {}
        except Exception:
            ctx = {}
        # Ensure builtins present for eval/exec
        self._ns.setdefault("__builtins__", __builtins__)
        # Update with dynamic context (window, mct, mcts, etc.)
        self._ns.update(ctx)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        # Prevent mouse from placing cursor before prompt
        if event.type() == QtCore.QEvent.MouseButtonPress:
            pos = event.position() if hasattr(event, "position") else event.pos()
            return False  # allow default, we'll clamp in keyPressEvent
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        mod = event.modifiers()

        # Ensure cursor never goes before current prompt
        if key in (QtCore.Qt.Key_Home,):
            # Move to after prompt on Home
            cur = self.textCursor()
            cur.setPosition(self._prompt_pos)
            self.setTextCursor(cur)
            return

        if key in (QtCore.Qt.Key_Backspace,):
            if self.textCursor().position() <= self._prompt_pos:
                return  # block backspace before prompt

        # History navigation (only if caret at end line region)
        if key in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
            cur = self.textCursor()
            if cur.position() >= self._prompt_pos:
                if key == QtCore.Qt.Key_Up:
                    self._history_prev()
                else:
                    self._history_next()
                return

        # Autocomplete on TAB
        if key == QtCore.Qt.Key_Tab:
            self._autocomplete()
            return

        # Execute on Enter/Return
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and not (mod & QtCore.Qt.ShiftModifier):
            self._on_enter()
            return

        # Allow explicit newline with Shift+Enter
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter) and (mod & QtCore.Qt.ShiftModifier):
            super().keyPressEvent(event)
            return

        # Block editing before prompt: if cursor is before prompt, jump to end
        if self.textCursor().position() < self._prompt_pos:
            cur = self.textCursor()
            cur.setPosition(self.document().characterCount() - 1)
            self.setTextCursor(cur)

        # If completer popup is visible, let it handle navigation keys
        if hasattr(self, "_completer") and self._completer.popup() and self._completer.popup().isVisible():
            if key in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return, QtCore.Qt.Key_Escape,
                    QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab,
                    QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                    QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown):
                event.ignore()
                return
    
        super().keyPressEvent(event)


    # ------------------------ Autocomplete helpers ------------------------
    def _current_line_text(self) -> str:
        doc = self.document().toPlainText()
        return doc[self._prompt_pos:]

    def _completion_context(self) -> dict:
        """Analyze current buffer; return kind/prefix/replacement anchor."""
        cursor = self.textCursor()
        abs_pos = cursor.position()
        upto = self.document().toPlainText()[self._prompt_pos:abs_pos]

        # 1) Dict/item access: foo['pre   or   foo["pre
        m = self._rx_item.search(upto)
        if m:
            base = m.group(1)
            prefix = m.group(2) or ""
            replace_from = abs_pos - len(prefix)
            return {"kind": "item", "base_expr": base, "prefix": prefix, "replace_from": replace_from}

        # 2) Attribute access: obj.pre
        m = self._rx_attr.search(upto)
        if m:
            base = m.group(1)
            prefix = m.group(2) or ""
            replace_from = abs_pos - len(prefix)
            return {"kind": "attr", "base_expr": base, "prefix": prefix, "replace_from": replace_from}

        # 3) Plain name
        m = self._rx_name.search(upto)
        if m:
            prefix = m.group(1)
            replace_from = abs_pos - len(prefix)
            return {"kind": "name", "base_expr": None, "prefix": prefix, "replace_from": replace_from}

        return {"kind": "name", "base_expr": None, "prefix": "", "replace_from": abs_pos}

    def _apply_completion_at(self, completion: str, replace_from: int) -> None:
        cur = self.textCursor()
        cur.setPosition(replace_from)
        cur.movePosition(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        cur.insertText(completion)
        self.setTextCursor(cur)

    def _apply_completion(self, text: str) -> None:
        rf = getattr(self._completer, 'replace_from', None)
        if rf is None:
            rf = self.textCursor().position()
        self._apply_completion_at(text, rf)
        
    # ------------------------------------------------------------------
    # History & autocomplete
    # ------------------------------------------------------------------
    def _history_prev(self) -> None:
        if not self._history:
            return
        self._hist_idx = max(0, self._hist_idx - 1) if self._hist_idx >= 0 else len(self._history) - 1
        self._set_current_input_text(self._history[self._hist_idx])

    def _history_next(self) -> None:
        if not self._history:
            return
        if self._hist_idx < len(self._history) - 1:
            self._hist_idx += 1
            self._set_current_input_text(self._history[self._hist_idx])
        else:
            self._hist_idx = len(self._history)
            self._set_current_input_text("")

    def _autocomplete(self) -> None:
        """Autocomplete using popup; supports name/attr/dict-key."""
        ctx_info = self._completion_context()
        prefix = ctx_info["prefix"]
        replace_from = ctx_info["replace_from"]
        kind = ctx_info["kind"]

        candidates: List[str] = []
        try:
            if kind == "name":
                names = sorted(set(list(self._ns.keys()) + dir(__builtins__)))
                candidates = [n for n in names if n.startswith(prefix)] if prefix else names
            else:
                base_expr = ctx_info["base_expr"]
                try:
                    base_obj = eval(base_expr, self._ns, self._ns)
                except Exception:
                    return
                if kind == "attr":
                    attrs = dir(base_obj)
                    if prefix and not prefix.startswith('_'):
                        attrs = [a for a in attrs if not a.startswith('_')]
                    candidates = [a for a in attrs if a.startswith(prefix)] if prefix else attrs
                elif kind == "item":
                    keys = []
                    try:
                        keys = list(base_obj.keys())  # may raise if not dict-like
                    except Exception:
                        keys = dir(base_obj)
                    keys = [k for k in keys if isinstance(k, str)]
                    candidates = [k for k in keys if k.startswith(prefix)] if prefix else keys
        except Exception:
            candidates = []

        if not candidates:
            return

        if len(candidates) == 1:
            self._apply_completion_at(candidates[0], replace_from)
            return

        model = QtCore.QStringListModel(sorted(candidates), self._completer)
        self._completer.setModel(model)
        self._completer.replace_from = replace_from  # type: ignore[attr-defined]
        cr = self.cursorRect()
        cr.setWidth(300)
        self._completer.complete(cr)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _on_enter(self) -> None:
        """Handle Enter: accept current line into buffer; execute when complete."""
        # Echo a newline
        self._write("\n")

        # Grab current input (may be multi-line already)
        line = self._current_input_text()
        self._buffer_lines.append(line)
        src = "\n".join(self._buffer_lines)

        # Determine if the code block is complete
        try:
            code_obj = self._compiler(src)
        except (OverflowError, SyntaxError, ValueError):
            code_obj = True  # Force execution to raise actual syntax error

        if code_obj is None:
            # Need more input -> print continuation prompt
            self._insert_prompt(primary=False)
            return

        code_to_run = src

        stripped = code_to_run.strip()
        if stripped.startswith('%'):
            self._refresh_namespace()
            import os, glob
            parts = stripped.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ''
            if cmd == '%run':
                window = self._ns.get('window')
                if window is not None and hasattr(window, '_exec_script_file'):
                    try:
                        window._exec_script_file(arg.strip())
                        self._write(f"[ran] {arg.strip()}\n")
                    except Exception as ex:
                        self._write(f"Error in %run: {ex}\n")
                else:
                    self._write("Error: window runner not available\n")
            elif cmd == '%pwd':
                self._write(os.getcwd() + "\n")
            elif cmd == '%cd':
                try:
                    os.chdir(arg.strip() or os.path.expanduser('~'))
                except Exception as ex:
                    self._write(f"cd: {ex}\n")
            elif cmd == '%ls':
                pat = arg.strip() or '*'
                try:
                    for name in sorted(glob.glob(pat)):
                        self._write(name + "\n")
                except Exception as ex:
                    self._write(f"ls: {ex}\n")
            elif cmd == '%clear':
                self.clear()
            else:
                self._write(f"Unknown magic: {cmd}\n")
            self._buffer_lines.clear()
            self._insert_prompt(primary=True)
            return

        # (Removed redundant %run handler block)

        # We have a complete block: execute
        self._history.append(code_to_run)
        self._hist_idx = len(self._history)
        self._buffer_lines.clear()

        # Refresh dynamic objects (window, mct/mcts, etc.) into the persistent namespace
        self._refresh_namespace()
        try:
            # Try eval first for simple expressions, using the same dict for globals/locals
            result = None
            try:
                result = eval(code_to_run, self._ns, self._ns)  # nosec: local dev console
            except SyntaxError:
                exec(code_to_run, self._ns, self._ns)  # nosec: local dev console
                result = None
            if result is not None:
                self._write(repr(result) + "\n")
            self.sigExecuted.emit(code_to_run)
        except Exception as ex:  # noqa: BLE001
            self._write(f"Error: {ex}\n")

        cur = self.textCursor()
        cur.movePosition(QtGui.QTextCursor.End)
        self.setTextCursor(cur)

        # New primary prompt
        self._insert_prompt(primary=True)
