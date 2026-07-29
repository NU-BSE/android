"""Policy-safe chat advisor for semantic Android navigation plans.

DRLQN still predicts semantic subgoals, but this module deliberately keeps the
last mile human-operated. It converts only the *next* subgoal into a concise
chat instruction, asks the user to confirm completion, and never exposes an
API that can dispatch a gesture or AccessibilityNodeInfo action.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence

from .core import AndroidState
from .window_graph import PlannerSubgoal, SubgoalType


class AdviceStatus(str, Enum):
    READY = "ready"
    WAITING_FOR_USER = "waiting_for_user"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 96
    temperature: float = 0.15
    top_p: float = 0.9
    stop: tuple[str, ...] = ()


class LocalChatModel(Protocol):
    """Minimal interface implemented by a LiteRT-LM or ONNX Runtime wrapper."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        options: GenerationOptions,
    ) -> str:
        ...


@dataclass(frozen=True)
class AdvisorConfig:
    locale: str = "ru"
    max_visible_labels: int = 48
    max_label_chars: int = 80
    max_instruction_chars: int = 320
    redact_sensitive_text: bool = True
    include_package_activity: bool = True
    generation: GenerationOptions = field(default_factory=GenerationOptions)


@dataclass(frozen=True)
class UserActionStep:
    step_index: int
    total_steps: int
    subgoal: PlannerSubgoal
    instruction: str
    expected_result: str
    confirmation_question: str
    target_label: str | None = None
    caution: str | None = None
    status: AdviceStatus = AdviceStatus.WAITING_FOR_USER
    generated_by_llm: bool = False


@dataclass(frozen=True)
class AdvisorReply:
    step: UserActionStep | None
    remaining_plan: tuple[PlannerSubgoal, ...]
    status: AdviceStatus
    message: str = ""


class AdvisorPolicyError(ValueError):
    """Raised when a model response attempts to cross the human-action boundary."""


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{6,}(?!\d)")
_OTP_HINT = re.compile(r"\b(?:otp|one[- ]time|код|парол|pin|cvv|cvc)\b", re.I)

# These strings indicate executable automation, not a user-facing instruction.
_FORBIDDEN_AUTOMATION = (
    "adb shell",
    "input tap",
    "input swipe",
    "dispatchgesture",
    "performaction",
    "accessibilitynodeinfo.action",
    "uiautomator",
    "am start",
    "monkeyrunner",
    "coordinates:",
    "node_id:",
    "x=",
    "y=",
)


SYSTEM_PROMPT_RU = """Ты — локальный помощник по навигации Android.
DRLQN уже выбрал семантическую подцель. Объясни пользователю ровно ОДНО
следующее действие. Пользователь выполняет действие сам.

Жёсткие ограничения:
- не утверждай, что ты нажал, ввёл, открыл или изменил что-либо;
- не выдавай координаты, node_id, команды ADB, shell, Accessibility API или код;
- не планируй и не выполняй автономную последовательность;
- используй только видимые подписи интерфейса из контекста;
- не повторяй персональные данные, коды, пароли и номера;
- попроси пользователя подтвердить результат перед следующим шагом.

Верни только JSON:
{"instruction":"...","expected_result":"...","confirmation_question":"...",
 "target_label":null,"caution":null}
"""

SYSTEM_PROMPT_EN = """You are an on-device Android navigation advisor.
DRLQN has already selected a semantic subgoal. Explain exactly ONE next action
for the user to perform manually.

Hard constraints:
- never claim that you tapped, typed, opened, or changed anything;
- never output coordinates, node ids, ADB/shell/Accessibility API commands, or code;
- never autonomously plan and execute a sequence;
- use only visible UI labels from the supplied context;
- do not repeat personal data, passwords, codes, or long numbers;
- ask the user to confirm the observed result before continuing.

Return JSON only:
{"instruction":"...","expected_result":"...","confirmation_question":"...",
 "target_label":null,"caution":null}
"""


class SensitiveTextRedactor:
    """Small deterministic privacy filter used before local prompting/logging."""

    @staticmethod
    def redact(text: str) -> str:
        value = _EMAIL.sub("[email]", text or "")
        value = _PHONE.sub("[phone]", value)
        if _OTP_HINT.search(value):
            value = _LONG_NUMBER.sub("[secret]", value)
        else:
            value = _LONG_NUMBER.sub("[number]", value)
        return " ".join(value.split())


class PlannerAdvisor:
    """Converts semantic subgoals into one-at-a-time chat guidance.

    There is intentionally no ``execute`` method. The caller must obtain an
    explicit user confirmation before dropping the first element of the plan.
    """

    def __init__(self, config: AdvisorConfig | None = None):
        self.config = config or AdvisorConfig()

    def advise(
        self,
        plan: Sequence[PlannerSubgoal],
        state: AndroidState,
        user_goal: str,
        *,
        llm: LocalChatModel | None = None,
        completed_steps: int = 0,
    ) -> AdvisorReply:
        plan = tuple(plan)
        if not plan:
            message = (
                "Задача завершена." if self.config.locale.startswith("ru")
                else "The task is complete."
            )
            return AdvisorReply(None, (), AdviceStatus.DONE, message)

        subgoal = plan[0]
        if subgoal.kind is SubgoalType.DONE:
            message = (
                "По плану дополнительных действий не требуется."
                if self.config.locale.startswith("ru")
                else "The plan requires no further action."
            )
            return AdvisorReply(None, plan[1:], AdviceStatus.DONE, message)

        deterministic = self._deterministic_step(
            subgoal,
            state,
            completed_steps + 1,
            len(plan) + completed_steps,
        )
        if llm is None:
            return AdvisorReply(deterministic, plan, AdviceStatus.WAITING_FOR_USER)

        try:
            prompt = self.build_messages(subgoal, state, user_goal)
            raw = llm.generate(prompt, self.config.generation)
            step = self._parse_llm_step(raw, deterministic, state)
        except (ValueError, TypeError, json.JSONDecodeError, AdvisorPolicyError):
            # A malformed or unsafe local-model response must degrade to a
            # deterministic instruction rather than crossing the policy boundary.
            step = deterministic
        return AdvisorReply(step, plan, AdviceStatus.WAITING_FOR_USER)

    def confirm_step(
        self,
        plan: Sequence[PlannerSubgoal],
        *,
        user_confirmed: bool,
        observed_expected_result: bool = True,
    ) -> tuple[PlannerSubgoal, ...]:
        """Advance only after explicit user confirmation and expected evidence."""
        plan = tuple(plan)
        if not plan:
            return ()
        if user_confirmed and observed_expected_result:
            return plan[1:]
        return plan

    def build_messages(
        self,
        subgoal: PlannerSubgoal,
        state: AndroidState,
        user_goal: str,
    ) -> tuple[ChatMessage, ...]:
        labels = self._visible_labels(state)
        redactor = SensitiveTextRedactor.redact
        if self.config.redact_sensitive_text:
            labels = [redactor(label) for label in labels]
            goal = redactor(user_goal)
            argument = redactor(subgoal.argument or "")
        else:
            goal = user_goal
            argument = subgoal.argument or ""

        context = {
            "goal": goal[:240],
            "subgoal": subgoal.kind.value,
            "argument": argument[:120] or None,
            "visible_labels": labels,
        }
        if self.config.include_package_activity:
            context["package"] = state.package
            context["activity"] = state.activity

        system = SYSTEM_PROMPT_RU if self.config.locale.startswith("ru") else SYSTEM_PROMPT_EN
        return (
            ChatMessage("system", system),
            ChatMessage("user", json.dumps(context, ensure_ascii=False, separators=(",", ":"))),
        )

    def _visible_labels(self, state: AndroidState) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for node in state.accessibility.nodes:
            label = " ".join((node.label or "").split())
            if not node.visible or not label:
                continue
            label = label[: self.config.max_label_chars]
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(label)
            if len(output) >= self.config.max_visible_labels:
                break
        return output

    def _find_target_label(self, subgoal: PlannerSubgoal, state: AndroidState) -> str | None:
        argument = (subgoal.argument or "").strip().casefold()
        labels = self._visible_labels(state)
        if argument:
            exact = next((label for label in labels if label.casefold() == argument), None)
            if exact:
                return exact
            partial = next((label for label in labels if argument in label.casefold()), None)
            if partial:
                return partial
        if subgoal.kind in (SubgoalType.OPEN_MENU, SubgoalType.CONFIRM):
            keywords = {
                SubgoalType.OPEN_MENU: ("menu", "меню", "more", "ещё", "options"),
                SubgoalType.CONFIRM: ("ok", "готово", "подтверд", "сохран", "apply", "done"),
            }[subgoal.kind]
            for label in labels:
                if any(word in label.casefold() for word in keywords):
                    return label
        return None

    def _deterministic_step(
        self,
        subgoal: PlannerSubgoal,
        state: AndroidState,
        index: int,
        total: int,
    ) -> UserActionStep:
        ru = self.config.locale.startswith("ru")
        target = self._find_target_label(subgoal, state)
        argument = subgoal.argument or target or "нужный элемент"

        if ru:
            instruction, expected, question, caution = self._template_ru(subgoal, argument, target)
        else:
            instruction, expected, question, caution = self._template_en(subgoal, argument, target)

        return UserActionStep(
            step_index=index,
            total_steps=total,
            subgoal=subgoal,
            instruction=instruction[: self.config.max_instruction_chars],
            expected_result=expected,
            confirmation_question=question,
            target_label=target,
            caution=caution,
            generated_by_llm=False,
        )

    @staticmethod
    def _template_ru(subgoal, argument, target):
        quoted = f"«{target or argument}»"
        kind = subgoal.kind
        if kind is SubgoalType.OPEN_MENU:
            return (f"Откройте меню {quoted if target else 'на текущем экране'}.",
                    "Должен появиться список дополнительных пунктов.",
                    "Меню открылось?", None)
        if kind is SubgoalType.FIND_SETTING:
            return (f"Найдите пункт «{argument}» и нажмите его.",
                    f"Должен открыться экран или раздел «{argument}».",
                    "Нужный раздел открылся?", None)
        if kind is SubgoalType.ENTER_VALUE:
            return (f"Коснитесь нужного поля и введите значение «{argument}».",
                    "Введённое значение должно появиться в поле.",
                    "Значение отображается правильно?",
                    "Не отправляйте в чат пароль, PIN, одноразовый код или платёжные данные.")
        if kind is SubgoalType.CONFIRM:
            return (f"Нажмите {quoted if target else 'кнопку подтверждения или сохранения'}.",
                    "Изменение должно сохраниться, а экран — обновиться.",
                    "Изменение сохранилось?", None)
        if kind is SubgoalType.TAP_TARGET:
            return (f"Нажмите {quoted}.", "Интерфейс должен перейти к следующему состоянию.",
                    "Экран изменился ожидаемым образом?", None)
        if kind is SubgoalType.SCROLL:
            return ("Прокрутите текущий список, чтобы увидеть следующие пункты.",
                    "На экране должны появиться новые элементы списка.",
                    "Появился нужный пункт?", None)
        if kind is SubgoalType.DRAG_SLIDER:
            return (f"Переместите ползунок {quoted if target else ''} до нужного значения.".strip(),
                    "Рядом с ползунком должно отображаться требуемое значение.",
                    "Ползунок установлен правильно?", None)
        if kind is SubgoalType.BACK:
            return ("Вернитесь на предыдущий экран системной кнопкой или жестом «Назад».",
                    "Должен открыться предыдущий экран.", "Предыдущий экран открылся?", None)
        if kind is SubgoalType.WAIT:
            return ("Подождите, пока экран перестанет обновляться.",
                    "Индикатор загрузки должен исчезнуть, а элементы — стать доступными.",
                    "Загрузка завершилась?", None)
        return ("Проверьте текущий экран и выполните следующий понятный шаг вручную.",
                "Экран должен соответствовать ожидаемому результату.",
                "Можно продолжать?", None)

    @staticmethod
    def _template_en(subgoal, argument, target):
        quoted = f'“{target or argument}”'
        kind = subgoal.kind
        if kind is SubgoalType.OPEN_MENU:
            return (f"Open {quoted if target else 'the menu on the current screen'}.",
                    "A list of additional options should appear.", "Did the menu open?", None)
        if kind is SubgoalType.FIND_SETTING:
            return (f"Find and tap “{argument}”.", f"The “{argument}” section should open.",
                    "Did the requested section open?", None)
        if kind is SubgoalType.ENTER_VALUE:
            return (f"Tap the relevant field and enter “{argument}”.",
                    "The entered value should be visible in the field.",
                    "Is the value shown correctly?",
                    "Do not share passwords, PINs, one-time codes, or payment data in chat.")
        if kind is SubgoalType.CONFIRM:
            return (f"Tap {quoted if target else 'the confirm or save button'}.",
                    "The change should be saved and the screen should update.",
                    "Was the change saved?", None)
        if kind is SubgoalType.TAP_TARGET:
            return (f"Tap {quoted}.", "The interface should move to the next state.",
                    "Did the screen change as expected?", None)
        if kind is SubgoalType.SCROLL:
            return ("Scroll the current list to reveal more items.",
                    "New list items should become visible.", "Can you see the needed item?", None)
        if kind is SubgoalType.DRAG_SLIDER:
            return (f"Move {quoted if target else 'the slider'} to the required value.",
                    "The required value should be displayed beside the slider.",
                    "Is the slider set correctly?", None)
        if kind is SubgoalType.BACK:
            return ("Use the system Back button or gesture.", "The previous screen should open.",
                    "Did the previous screen open?", None)
        if kind is SubgoalType.WAIT:
            return ("Wait until the screen stops updating.",
                    "The loading indicator should disappear and controls should become available.",
                    "Has loading finished?", None)
        return ("Review the current screen and perform the next clear step manually.",
                "The screen should match the expected result.", "Can we continue?", None)

    def _parse_llm_step(
        self,
        raw: str,
        fallback: UserActionStep,
        state: AndroidState,
    ) -> UserActionStep:
        payload = json.loads(self._extract_json(raw))
        required = ("instruction", "expected_result", "confirmation_question")
        if any(not isinstance(payload.get(key), str) or not payload[key].strip() for key in required):
            raise ValueError("missing advisor fields")

        strings = [payload.get(key) or "" for key in (*required, "target_label", "caution")]
        normalized = " ".join(strings).casefold()
        if any(token in normalized for token in _FORBIDDEN_AUTOMATION):
            raise AdvisorPolicyError("local model produced executable automation content")
        if re.search(r"\b(?:x|y)\s*[:=]\s*\d", normalized):
            raise AdvisorPolicyError("coordinate-like output is not allowed")

        # The model saw redacted labels, so accept either form before falling back.
        visible: set[str] = set()
        for label in self._visible_labels(state):
            visible.add(label.casefold())
            if self.config.redact_sensitive_text:
                visible.add(SensitiveTextRedactor.redact(label).casefold())
        if fallback.target_label:
            visible.add(fallback.target_label.casefold())
        target = payload.get("target_label")
        if target is not None:
            if not isinstance(target, str):
                raise ValueError("target_label must be a string or null")
            if visible and target.casefold() not in visible:
                target = fallback.target_label

        return UserActionStep(
            step_index=fallback.step_index,
            total_steps=fallback.total_steps,
            subgoal=fallback.subgoal,
            instruction=payload["instruction"].strip()[: self.config.max_instruction_chars],
            expected_result=payload["expected_result"].strip()[:240],
            confirmation_question=payload["confirmation_question"].strip()[:160],
            target_label=target or fallback.target_label,
            caution=(payload.get("caution") or fallback.caution),
            generated_by_llm=True,
        )

    @staticmethod
    def _extract_json(raw: str) -> str:
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
            raw = re.sub(r"\s*```$", "", raw)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise json.JSONDecodeError("JSON object not found", raw, 0)
        return raw[start : end + 1]
